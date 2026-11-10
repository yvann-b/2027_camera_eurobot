"""
hardware/camera_daheng.py

Source de frames synchrone sur la caméra Daheng réelle (SDK gxipy).

Interface volontairement symétrique à VideoFileSimulation.get_frame() :
get_frame() -> (frame, frame_id, timestamp_ns) ou None. Le process appelant
(process_image_acquisition) reste ainsi identique, que la source soit la
vraie caméra ou une vidéo de simulation — remap de distorsion et écriture
dans le ring buffer se font UNE SEULE FOIS, dans ce process, pas ici.

Pas de thread, pas de ring, pas de stop_event dans cette classe : elle ne
fait qu'une chose, "donne-moi la prochaine frame", de façon synchrone.
Le LIFO ("toujours la plus fraîche, jamais de queue bloquante") est garanti
par le SDK lui-même via StreamBufferHandlingMode=NewestOnly (voir
_force_freerun_settings), pas par un thread + verrou côté Python : si la
boucle appelante est plus lente que le capteur, get_image() renvoie quand
même la dernière image, le driver jette le reste tout seul.

Point d'attention corrigé par rapport à l'ancien code (LIAISON_classes.py,
classe DahengCamera) : l'ancien __init__ ne touchait jamais ExposureAuto ni
ExposureTime. La caméra héritait donc silencieusement de l'état laissé par
la dernière session GalaxyView (ou un essai précédent en exposition
manuelle figée), sans aucune garantie que cet état soit celui voulu au
prochain lancement. Ici, TriggerMode, AcquisitionMode, ExposureAuto,
GainAuto et StreamBufferHandlingMode sont forcés explicitement à chaque
instanciation.
"""

from __future__ import annotations

import logging
import time

import cv2

import gxipy as gx
import numpy as np

from config.loader import CameraConfig

logger = logging.getLogger(__name__)


class DahengCameraError(Exception):
    """Erreur d'initialisation ou d'exécution spécifique à la caméra Daheng."""


class DahengCamera:
    """
    Source de frames synchrone sur une caméra Daheng en freerun.

    Interface symétrique à VideoFileSimulation : start() / get_frame() /
    stop(). __init__ ouvre le device et fige les réglages (doit être fait
    avant stream_on) ; start() ne fait que déclencher le flux.

    Usage (dans process_image_acquisition, boucle et écriture au ring
    possédées par l'appelant, pas par cette classe) :
        camera = DahengCamera(camera_config)
        camera.start()
        while not stop_event.is_set():
            result = camera.get_frame()
            if result is None:
                continue
            frame, frame_id, timestamp_ns = result
            frame = cv2.remap(frame, map1, map2, cv2.INTER_NEAREST)
            frame_ring.write_image(frame, frame_id, timestamp_ns)
        camera.stop()
    """

    def __init__(self, camera_cfg: CameraConfig) -> None:
        self._cfg = camera_cfg
        self._frame_id = 0

        # --- 1. Énumération et ouverture ---
        self._device_manager = gx.DeviceManager()
        dev_num, _ = self._device_manager.update_device_list()
        if dev_num == 0:
            raise DahengCameraError("Aucune caméra Daheng détectée.")

        # ATTENTION (hypothèse à valider avec le matériel en main) :
        # gxipy est en index 1-based (manuel §2.2.3 : "index [1, 2, 3...]"),
        # alors que camera.device_id (pipeline_config.yaml) suit la
        # convention 0-based partagée avec cv2.VideoCapture / la simulation
        # vidéo. D'où le +1 ici. Si une seule caméra est branchée,
        # device_id=0 -> index Daheng 1, ce qui correspond à l'ancien code
        # (open_device_by_index(1)).
        daheng_index = self._cfg.device_id + 1
        try:
            self._cam = self._device_manager.open_device_by_index(daheng_index)
        except Exception as exc:
            raise DahengCameraError(
                f"Impossible d'ouvrir la caméra Daheng à l'index {daheng_index} "
                f"(device_id={self._cfg.device_id} dans la config) : {exc}"
            ) from exc

        self._remote = self._cam.get_remote_device_feature_control()

        # --- 2. Réglages explicites et reproductibles (voir docstring module) ---
        self._force_freerun_settings()

        # --- 3. Conversion de format ---
        self._converter = self._device_manager.create_image_format_convert()
        self._converter.set_dest_format(gx.GxPixelFormatEntry.RGB8)

        # Buffer de sortie = np.ndarray déjà dans la forme attendue par
        # FrameRing.write_image, réutilisé à chaque frame. Le SDK écrit
        # directement dedans via .ctypes.data (adresse mémoire brute).
        self._out_frame = np.empty(
            (self._cfg.height, self._cfg.width, self._cfg.channels), dtype=np.uint8
        )
        self._daheng_index = daheng_index  # gardé pour le log dans start()

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Déclenche le flux. Doit être appelé après __init__ (réglages déjà
        figés) et avant le premier get_frame()."""
        self._cam.stream_on()
        logger.info(
            "Daheng: flux démarré (index=%d, %dx%d, freerun, "
            "ExposureAuto/GainAuto=Continuous, StreamBufferHandlingMode=NewestOnly).",
            self._daheng_index, self._cfg.width, self._cfg.height,
        )

    # ------------------------------------------------------------------
    def _force_freerun_settings(self) -> None:
        """Force un état caméra explicite, indépendant de l'état résiduel
        laissé par une session précédente (GalaxyView, essai mode photo...).

        Choix acté : ExposureAuto = Continuous (comportement connu et
        validé jusqu'ici).
        """
        if self._remote.is_implemented("TriggerMode"):
            self._remote.get_enum_feature("TriggerMode").set("Off")
        else:
            logger.warning("TriggerMode non implémenté sur ce device.")

        if self._remote.is_implemented("AcquisitionMode"):
            self._remote.get_enum_feature("AcquisitionMode").set("Continuous")

        if self._remote.is_implemented("ExposureAuto"):
            self._remote.get_enum_feature("ExposureAuto").set("Continuous")
        else:
            logger.warning(
                "ExposureAuto non implémenté sur ce device : "
                "impossible de forcer un état d'exposition explicite."
            )

        if self._remote.is_implemented("GainAuto"):
            self._remote.get_enum_feature("GainAuto").set("Continuous")

        # Garantit le LIFO côté driver : get_image() renvoie toujours la
        # frame la plus récente et jette les plus anciennes si le
        # consommateur (process_image_acquisition) est plus lent que le
        # capteur. C'est ce qui remplace le thread + verrou "latest_frame"
        # de l'ancien code, sans le réimplémenter en Python.
        # NOTE : StreamBufferHandlingMode vit sur le stream, pas sur le
        # remote device -> feature control différent (get_stream, index
        # 1-based lui aussi d'après le manuel §3.3.2.15).
        try:
            stream_features = self._cam.get_stream(1).get_feature_control()
            if stream_features.is_implemented("StreamBufferHandlingMode"):
                stream_features.get_enum_feature("StreamBufferHandlingMode").set("NewestOnly")
            else:
                logger.warning(
                    "StreamBufferHandlingMode non implémenté : le LIFO n'est "
                    "plus garanti par le driver si le consommateur est plus "
                    "lent que le capteur. À vérifier avec le matériel en main."
                )
        except Exception:
            logger.exception("Impossible de configurer StreamBufferHandlingMode.")

    # ------------------------------------------------------------------
    def get_frame(self) -> tuple[np.ndarray, int, int] | None:
        """Une capture synchrone. Retourne None sur timeout/frame invalide
        (l'appelant doit alors simplement reboucler) — jamais de blocage
        indéfini grâce au timeout de get_image()."""
        try:
            raw_image = self._cam.data_stream[0].get_image(timeout=1000)
            if raw_image is None or raw_image.get_status() != gx.GxFrameStatusList.SUCCESS:
                return None

            # Horodatage pris ici, au plus tôt côté logiciel (juste après la
            # remise de la frame par le SDK), pas raw_image.get_timestamp()
            # (horloge interne du capteur, pas alignée sur l'epoch système) :
            # tous les consommateurs relisent le même ring, donc la
            # corrélation ne dépend pas d'une horloge matérielle absolue.
            timestamp_ns = time.time_ns()

            img_w = raw_image.get_width()
            img_h = raw_image.get_height()
            if (img_w, img_h) != (self._cfg.width, self._cfg.height):
                logger.error(
                    "Daheng: dimensions reçues (%dx%d) != config (%dx%d), frame ignorée.",
                    img_w, img_h, self._cfg.width, self._cfg.height,
                )
                return None

            # Dématriçage Bayer -> RGB8, écrit directement dans self._out_frame.
            actual_size = self._converter.get_buffer_size_for_conversion(raw_image)
            self._converter.convert(raw_image, self._out_frame.ctypes.data, actual_size, False)

            if self._cfg.color_order == "BGR":
                frame = cv2.cvtColor(self._out_frame, cv2.COLOR_RGB2BGR)
            else:
                frame = self._out_frame.copy()  # copie : self._out_frame sera réécrit au prochain appel

            frame_id = self._frame_id
            self._frame_id += 1
            return frame, frame_id, timestamp_ns

        except Exception:
            logger.exception("Daheng: erreur pendant la capture, frame ignorée.")
            return None

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._cam.stream_off()
            self._cam.close_device()
        except Exception:
            logger.exception("Daheng: erreur pendant la fermeture du device.")

    stop = close  # alias pour coller au vocabulaire du process appelant


# ---------------------------------------------------------------------------
# Smoke test de bring-up matériel, indépendant du reste du pipeline.
#   python -m hardware.camera_daheng config/pipeline_config.yaml
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from config.loader import load_config
    from ring.frame_ring import FrameRing

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/pipeline_config.yaml"
    logging.basicConfig(level=logging.INFO)

    cfg = load_config(config_path)
    ring = FrameRing(
        cam=cfg.camera,
        header=cfg.ring_buffer_header,
        data_slot=cfg.ring_buffers["camera"],
        is_creating=True,
    )

    camera = DahengCamera(cfg.camera)
    camera.start()
    print("Capture en cours, 'q' ou Ctrl+C pour arrêter...")
    last_id = -1
    t0, n_frames = time.time(), 0
    try:
        while True:
            result = camera.get_frame()
            if result is not None:
                frame, frame_id, timestamp_ns = result
                ring.write_image(frame, frame_id, timestamp_ns)
                last_id = frame_id
                n_frames += 1
                cv2.imshow("Daheng - smoke test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if time.time() - t0 >= 1.0:
                print(f"~{n_frames} fps (frame_id courant = {last_id})")
                n_frames, t0 = 0, time.time()
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()
        ring.close()
        ring.unlink()