"""
config/loader.py

Chargement et validation de la configuration générale du pipeline.

Ce module NE couvre PAS la configuration du bus ZMQ interne/robot
(voir external_bus/robot_bus_config.py) : ici on ne stocke que
l'interrupteur external_bus.enabled.

Usage basique :
    from config import load_config
    cfg = load_config("config/pipeline_config.yaml")
    print(cfg.summary())

Multiprocessing :
    PipelineConfig est un dataclass "frozen" composé uniquement de types
    picklables (int, float, str, bool, Path, dict de dataclasses simples).
    Deux façons de le distribuer aux sous-process :
      1) le passer directement en argument de multiprocessing.Process(...) ;
      2) passer seulement le chemin du yaml et appeler load_config(path)
         dans chaque sous-process (recommandé avec le start method
         'spawn' : plus robuste, pas d'état implicitement partagé).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from p_math.changement_repere import from_coordinate_to_pixel
import numpy as np
import yaml

logger = logging.getLogger(__name__)

SUPPORTED_VERSIONS = (1,)

# Tailles en octets des dtypes reconnus (header ring buffer + camera)
_DTYPE_SIZES: dict[str, int] = {
    "uint8": 1,
    "uint16": 2,
    "int64": 8,
    "float32": 4,
    "float64": 8,
}

_DTYPE_STRUCT_CODES = {
    'uint8': 'B', 'uint16': 'H', 'uint32': 'I', 'uint64': 'Q',
    'int32': 'i', 'int64': 'q', 'float32': 'f', 'float64': 'd',
}


_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_VALID_COLOR_ORDERS = {"BGR", "RGB"}
_VALID_ZONE_TYPES = {"depot", "ramassage"}


class ConfigError(Exception):
    """Fichier de config manquant, mal formé, ou valeurs incohérentes."""


# ---------------------------------------------------------------------------
# Sous-sections
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Schémas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameMessage:
    frame_id : int
    frame_timestamp_ns : int
    frame : np.ndarray

@dataclass(frozen=True)
class ZoneObstructionMessage:
    frame_id: int
    timestamp_ns: int
    is_blocked: np.ndarray  # shape (n_zones,), dtype bool — obstruction par robot, PAS le FoV caméra

@dataclass(frozen=True)
class CalibrationData:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    roi: tuple

@dataclass
class Zone:
    header: ZoneHeader
    coordinates_center_pixel: tuple[float, float]
    is_visible: bool
    is_blocked : bool | None = None
    """
    -----------
    Pas défini quand placé dans le ring 
    """
    is_full: bool | None = None          # None légitime ici : pas encore vu de frame
    number_blue : int = 0
    number_yellow : int = 0
    dominant_color: str | None = None    # idem



# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeConfig:
    standalone: bool
    target_fps: float
    log_level: str
    yellow_rob_height : int
    blue_rob_height : int


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    channels: int
    dtype: str
    color_order: str
    device_id: int
    video_name_simulation : str
    is_camera_used : bool

    @property
    def frame_bytes(self) -> int:
        """Taille en octets d'une frame brute complète."""
        return self.width * self.height * self.channels * _DTYPE_SIZES[self.dtype]


@dataclass(frozen=True)
class CalibrationConfig :
    number_frame : int
    npz_path : Path
    output_calib_file : Path
    number_try_to_calib : int


@dataclass(frozen=True)
class RingBufferHeaderSpec:
    """Layout du header partagé par tous les slots de tous les ring buffers."""
    frame_id_dtype: str
    timestamp_dtype: str

    @property
    def header_bytes(self) -> int:
        return _DTYPE_SIZES[self.frame_id_dtype] + _DTYPE_SIZES[self.timestamp_dtype]

    @property
    def struct_format(self) -> str:
        return '<' + _DTYPE_STRUCT_CODES[self.frame_id_dtype] + _DTYPE_STRUCT_CODES[self.timestamp_dtype]


@dataclass(frozen=True)
class RingBufferStreamSpec :
    name: str
    n_slots: int
    shm_name: str
    payload_bytes: int
    header_bytes: int

    @property
    def slot_bytes(self) -> int:
        return self.header_bytes + self.payload_bytes

    @property
    def total_bytes(self) -> int:
        return self.n_slots * self.slot_bytes


@dataclass(frozen=True)
class MemoryConfig:
    max_ram_gb: float
    warn_threshold_ratio: float


@dataclass(frozen=True)
class PathsConfig:
    models_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class ExternalBusConfig:
    enabled: bool
    ip : str
    port : int
    topic : bytes
    sndhwm : int
    linger_ms : int

@dataclass(frozen=True)
class Homographie_config :
    width : int
    height : int
    security_margin : float
    ratio_mm_per_pixel : int
    number_channel_cam : int
    dtype: str

    @property
    def frame_bytes(self) -> int:
        """Taille en octets d'une frame brute complète."""
        return int((self.width + 2*self.security_margin) /self.ratio_mm_per_pixel) * int((self.height + 2*self.security_margin)/self.ratio_mm_per_pixel) * self.number_channel_cam * _DTYPE_SIZES[self.dtype]

@dataclass(frozen=True)
class Subframe_Configuration:
    name: str
    corners: tuple[tuple[float, float, float], ...]
    homo_config : Homographie_config

    def __post_init__(self) -> None:
        n = len(self.corners)
        cx = sum(c[0] for c in self.corners) / n
        cy = sum(c[1] for c in self.corners) / n
        cz = sum(c[2] for c in self.corners) / n
        object.__setattr__(self, "center", (cx, cy, cz))

@dataclass(frozen=True)
class YOLO_config:
    model : str
    confiance : float
    target_class : list[str]
    height : int

@dataclass(frozen=True)
class Kalman_config :
    threshold_nohits : float
    threshold_fusion : float
    initial_confidence : np.ndarray
    covariance_model : np.ndarray
    covariance_mesurement : np.ndarray
    distance : float

_VALID_ZONE_TYPES = {"depot", "ramassage"}


@dataclass(frozen = True)
class Jenga :
    height : int
    length : int
    width : int


@dataclass(frozen=True)
class ZoneHeader:
    """Définition statique d'une zone : id, nom, type, géométrie.
    Ne change jamais après le chargement — l'état dynamique (visible ou
    non) est publié séparément sur le ring buffer zone_state."""
    id: int
    name: str
    type: str
    corners: tuple[tuple[float, float, float], ...]
    center: tuple[float, float, float] = field(init=False)
    is_visible : bool = None
    slots : tuple[tuple[float, float, float], ...] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        n = len(self.corners)
        cx = sum(c[0] for c in self.corners) / n
        cy = sum(c[1] for c in self.corners) / n
        cz = sum(c[2] for c in self.corners) / n
        object.__setattr__(self, "center", (cx, cy, cz))


        if self.type == "ramassage":
            corner1 = self.corners[0]  # bas-gauche
            corner2 = self.corners[1]  # haut-gauche
            corner4 = self.corners[3]  # bas-droite

            largeur = corner4[0] - corner1[0]
            hauteur = corner2[1] - corner1[1]

            is_horizontal = largeur >= hauteur

            centers = np.zeros((4, 3))

            if is_horizontal:
                centers[:, 1] = corner1[1] + hauteur / 2

                offset_initial = largeur / 8
                pas = largeur / 4
                centers[:, 0] = corner1[0] + offset_initial + np.arange(4) * pas

            else:
                centers[:, 0] = corner1[0] + largeur / 2

                offset_initial = hauteur / 8
                pas = hauteur / 4
                centers[:, 1] = corner1[1] + offset_initial + np.arange(4) * pas

            object.__setattr__(self, "slots", tuple(map(tuple, centers)))

# ---------------------------------------------------------------------------
# Config racine
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    version: int
    runtime: RuntimeConfig
    camera: CameraConfig
    calibration : CalibrationConfig
    ring_buffer_header: RingBufferHeaderSpec
    ring_buffers: dict[str, RingBufferStreamSpec]
    memory: MemoryConfig
    paths: PathsConfig
    external_bus: ExternalBusConfig
    source_path: Path
    zones_header : dict[int, ZoneHeader]
    homographie_config : Homographie_config
    subframes_dico : dict[str, Subframe_Configuration]
    yolo_config : YOLO_config
    kalman_config : Kalman_config
    jenga : Jenga

    # -- dérivés ------------------------------------------------------------

    def estimated_ring_buffer_gb(self) -> float:
        total = sum(s.total_bytes for s in self.ring_buffers.values())
        return total / 1e9

    def summary(self) -> str:
        lines = [
            f"PipelineConfig (v{self.version}) chargé depuis {self.source_path}",
            f"  standalone={self.runtime.standalone}  target_fps={self.runtime.target_fps}",
            f"  camera: {self.camera.width}x{self.camera.height}x{self.camera.channels} "
            f"({self.camera.dtype}, {self.camera.color_order}) -> "
            f"{self.camera.frame_bytes / 1e6:.1f} MB/frame",
        ]
        for name, spec in self.ring_buffers.items():
            lines.append(
                f"  ring_buffer[{name}]: {spec.n_slots} slots x {spec.slot_bytes} B "
                f"(header={spec.header_bytes}B + payload={spec.payload_bytes}B) "
                f"= {spec.total_bytes / 1e6:.1f} MB"
            )
        lines.append(
            f"  ring buffers total ~= {self.estimated_ring_buffer_gb():.2f} GB "
            f"/ budget {self.memory.max_ram_gb:.1f} GB"
        )
        lines.append(f"  external_bus.enabled={self.external_bus.enabled}")
        return "\n".join(lines)

    # -- chargement -----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Fichier de config introuvable : {path}")

        with path.open("r", encoding="utf-8") as f:
            try:
                raw = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise ConfigError(f"YAML invalide dans {path} : {exc}") from exc

        cfg = cls._from_dict(raw, source_path=path)
        cfg._validate()
        return cfg

    @classmethod
    def _from_dict(cls, raw: dict[str, Any], source_path: Path) -> "PipelineConfig":
        try:
            version = int(raw["version"])

            r = raw["runtime"]
            runtime = RuntimeConfig(
                standalone=bool(r["standalone"]),
                target_fps=float(r["target_fps"]),
                log_level=str(r["log_level"]).upper(),
                yellow_rob_height=int(r["yellow_robot_height"]),
                blue_rob_height=int(r["blue_robot_height"]),
            )

            c = raw["camera"]
            camera = CameraConfig(
                width=int(c["width"]),
                height=int(c["height"]),
                channels=int(c["channels"]),
                dtype=str(c["dtype"]),
                color_order=str(c["color_order"]).upper(),
                device_id=int(c["device_id"]),
                video_name_simulation = str(c["video_name_simulation"]).upper(),
                is_camera_used = bool(c["is_camera_used"]),
            )

            t = raw["table"]
            homographie_config = Homographie_config(
                width=int(t["width"]),
                height=int(t["height"]),
                security_margin=float(t["security_margin"]),
                ratio_mm_per_pixel=int(t["ratio_mm_per_pixel"]),
                number_channel_cam=camera.channels,
                dtype=camera.dtype,
            )


            cal = raw["calibration"]
            calibration = CalibrationConfig(
                number_frame=int(cal["number_frame"]),
                output_calib_file=Path(cal["output_path"]),
                number_try_to_calib=int(cal["number_try_to_calib"]),
                npz_path=Path(cal["calibration_distorsion_file"]),
            )


            rb = raw["ring_buffer"]
            h = rb["header"]
            header_spec = RingBufferHeaderSpec(
                frame_id_dtype=str(h["frame_id_dtype"]),
                timestamp_dtype=str(h["timestamp_dtype"]),
            )

            ring_buffers: dict[str, RingBufferStreamSpec] = {}
            for stream_name, s in rb["streams"].items():
                payload_bytes = s.get("payload_bytes")
                if payload_bytes is None:
                    if stream_name != "camera" and stream_name != "frame_redresse":
                        raise ConfigError(
                            f"ring_buffer.streams.{stream_name}.payload_bytes est requis "
                            f"(seul le flux 'camera' et 'frame_redresse' peut être déduit automatiquement)."
                        )
                    if stream_name == "camera" :
                        payload_bytes = camera.frame_bytes
                    elif stream_name == "frame_redresse":
                        payload_bytes = homographie_config.frame_bytes
                else:
                    payload_bytes = int(payload_bytes)

                ring_buffers[stream_name] = RingBufferStreamSpec(
                    name=stream_name,
                    n_slots=int(s["n_slots"]),
                    shm_name=str(s["shm_name"]),
                    payload_bytes=payload_bytes,
                    header_bytes=header_spec.header_bytes,
                )

            m = raw["memory"]
            memory = MemoryConfig(
                max_ram_gb=float(m["max_ram_gb"]),
                warn_threshold_ratio=float(m.get("warn_threshold_ratio", 0.8)),
            )

            p = raw["paths"]
            paths = PathsConfig(
                models_dir=Path(p["models_dir"]),
                logs_dir=Path(p["logs_dir"]),
            )

            eb = raw["external_bus"]
            external_bus = ExternalBusConfig(   enabled=bool(eb["enabled"]),
                                                ip = str(eb["ROBOT_BUS_IP"]),
                                                port = int(eb["ROBOT_BUS_PORT"]),
                                                topic=str(eb["ROBOT_BUS_TOPIC"]).encode(),                                                sndhwm = int(eb["ROBOT_BUS_SNDHWM"]),
                                                linger_ms = int(eb["ROBOT_BUS_LINGER_MS"])
                                             )

            zones_header: dict[int, ZoneHeader] = {}
            for z in raw["zones"]:
                zone_id = int(z["id"])
                if zone_id in zones_header:
                    raise ConfigError(
                        f"zones: id {zone_id} dupliqué (utilisé par '{zones_header[zone_id].name}' "
                        f"et '{z['name']}')"
                    )
                corners = tuple(tuple(float(v) for v in corner) for corner in z["corners"])
                zones_header[zone_id] = ZoneHeader(
                    id=zone_id, name=str(z["name"]), type=str(z["type"]), corners=corners,
                )

            subframes_dico : dict[str,Subframe_Configuration] = {}
            for s in raw["subframe"]:
                name = s["name"]
                corners = tuple(tuple(float(v) for v in corner) for corner in s["corners"])

                subframes_dico[name] =Subframe_Configuration(
                    name=name,
                    corners=corners,
                    homo_config=homographie_config)


            y = raw["YOLO"]
            YOLO = YOLO_config(
                model = y["models"],
                confiance=y["confiance"],
                target_class=y["target_class"],
                height = y["object_height"]
            )

            k = raw["Kalman"]
            initial_confidence = np.array(k["initial_confidence"], dtype=float)
            covariance_model = np.array(k["covariance_model"], dtype=float)
            covariance_mesurement = np.array(k["covariance_mesurement"], dtype=float)
            Kalman = Kalman_config(
                initial_confidence=initial_confidence,
                covariance_model=covariance_model,
                covariance_mesurement=covariance_mesurement,
                threshold_fusion=k["threshold_fusion"],
                threshold_nohits=k["threshold_nohits"],
                distance = k["distance"],
            )

            j = raw["Jenga"]
            jenga = Jenga(
                height = j["height"],
                length = j["length"],
                width = j["width"]
            )


        except KeyError as exc:
            raise ConfigError(f"Clé de configuration manquante : {exc}") from exc

        return cls(
            version=version,
            runtime=runtime,
            camera=camera,
            calibration=calibration,
            ring_buffer_header=header_spec,
            ring_buffers=ring_buffers,
            memory=memory,
            paths=paths,
            external_bus=external_bus,
            source_path=source_path,
            zones_header=zones_header,
            homographie_config=homographie_config,
            subframes_dico=subframes_dico,
            yolo_config=YOLO,
            kalman_config=Kalman,
            jenga = jenga,
        )

    # -- validation -----------------------------------------------------------

    def _validate(self) -> None:
        if self.version not in SUPPORTED_VERSIONS:
            raise ConfigError(
                f"version={self.version} non supportée (supportées : {SUPPORTED_VERSIONS})"
            )

        if self.runtime.target_fps <= 0:
            raise ConfigError("runtime.target_fps doit être > 0")
        if self.runtime.log_level not in _VALID_LOG_LEVELS:
            raise ConfigError(f"runtime.log_level invalide : {self.runtime.log_level}")

        if self.camera.width <= 0 or self.camera.height <= 0 or self.camera.channels <= 0:
            raise ConfigError("camera.width/height/channels doivent être > 0")
        if self.camera.dtype not in _DTYPE_SIZES:
            raise ConfigError(f"camera.dtype inconnu : {self.camera.dtype}")
        if self.camera.color_order not in _VALID_COLOR_ORDERS:
            raise ConfigError(f"camera.color_order invalide : {self.camera.color_order}")

        if self.ring_buffer_header.frame_id_dtype not in _DTYPE_SIZES:
            raise ConfigError("ring_buffer.header.frame_id_dtype inconnu")
        if self.ring_buffer_header.timestamp_dtype not in _DTYPE_SIZES:
            raise ConfigError("ring_buffer.header.timestamp_dtype inconnu")

        if not self.calibration.npz_path.is_file():
            raise ConfigError(f"Fichier de calibration introuvable : {self.calibration.npz_path}")

        if not self.ring_buffers:
            raise ConfigError("ring_buffer.streams est vide")

        shm_names = [s.shm_name for s in self.ring_buffers.values()]
        if len(shm_names) != len(set(shm_names)):
            raise ConfigError(f"shm_name dupliqués entre ring buffers : {shm_names}")

        for name, spec in self.ring_buffers.items():
            if spec.n_slots <= 0:
                raise ConfigError(f"ring_buffer.streams.{name}.n_slots doit être > 0")
            if spec.payload_bytes <= 0:
                raise ConfigError(f"ring_buffer.streams.{name}.payload_bytes doit être > 0")

        if self.memory.max_ram_gb <= 0:
            raise ConfigError("memory.max_ram_gb doit être > 0")
        if not (0 < self.memory.warn_threshold_ratio <= 1):
            raise ConfigError("memory.warn_threshold_ratio doit être dans (0, 1]")

        if not self.zones_header:
            raise ConfigError("zones est vide")
        for zone_id, z in self.zones_header.items():
            if z.id < 0:
                raise ConfigError(f"zones: id négatif interdit ({z.id})")
            if z.type not in _VALID_ZONE_TYPES:
                raise ConfigError(f"zones[{z.id}].type invalide : {z.type}")
            if len(z.corners) < 3:
                raise ConfigError(f"zones[{z.id}] doit avoir au moins 3 coins")
            if not z.name:
                raise ConfigError(f"zones[{z.id}].name est vide")
        names = [z.name for z in self.zones_header.values()]
        if len(names) != len(set(names)):
            raise ConfigError(f"noms de zones dupliqués : {names}")

        estimated = self.estimated_ring_buffer_gb()
        budget = self.memory.max_ram_gb
        if estimated > budget:
            raise ConfigError(
                f"Les ring buffers seuls ({estimated:.2f} GB) dépassent le budget "
                f"mémoire cible ({budget:.1f} GB). Réduire n_slots ou payload_bytes."
            )
        if estimated > budget * self.memory.warn_threshold_ratio:
            logger.warning(
                "Ring buffers ~= %.2f GB, proche du budget cible %.1f GB (seuil %.0f%%).",
                estimated, budget, self.memory.warn_threshold_ratio * 100,
            )


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "config/pipeline_config.yaml") -> PipelineConfig:
    """Charge, valide et retourne la configuration du pipeline."""
    cfg = PipelineConfig.from_yaml(path)
    logger.info("Configuration chargée :\n%s", cfg.summary())
    return cfg


def configure_logging(cfg: PipelineConfig) -> None:
    """Configure le logging racine à partir de runtime.log_level."""
    logging.basicConfig(
        level=getattr(logging, cfg.runtime.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


if __name__ == "__main__":
    import sys

    _path = sys.argv[1] if len(sys.argv) > 1 else "config/pipeline_config.yaml"
    _config = load_config(_path)
    print(_config.summary())
