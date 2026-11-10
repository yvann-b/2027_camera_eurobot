"""
calibration/calibration.py

TableCalibration : calcule la pose caméra -> table (R_cam, t_cam) à partir
des 4 tags de référence fixes (20-23), moyennés sur plusieurs frames pour
lisser le bruit de détection. Persiste le résultat sur disque.

Cas limite documenté : on acquiert jusqu'à MIN_VALID_FRAMES frames VALIDES
(les 4 tags de ref vus), pas MIN_VALID_FRAMES frames brutes — une frame où
un tag est occulté/flou ne compte pas.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from p_aruco.tag_extractor import TagExtractor, TagCategory


# Positions théoriques des tags de référence sur la table (mm), centre du tag.
# Origine (0,0,0) = coin bas-gauche de la table.
REF_TAGS_DEFINITION: dict[int, np.ndarray] = {
    23: np.array([2400.0, 600.0, 0.0]),
    22: np.array([600.0, 600.0, 0.0]),
    20: np.array([600.0, 1400.0, 0.0]),
    21: np.array([2400.0, 1400.0, 0.0]),
}

REF_TAG_IDS = tuple(REF_TAGS_DEFINITION.keys())
MIN_VALID_FRAMES = 100  # frames VALIDES requises (pas des frames brutes)


@dataclass
class CameraPose:
    """Pose caméra -> table, résultat figé de la calibration (persistable)."""
    R_cam: np.ndarray  # (3, 3)
    t_cam: np.ndarray  # (3, 1)

    def save(self, path: str) -> None:
        np.savez(path, R_cam=self.R_cam, t_cam=self.t_cam)

    @classmethod
    def load(cls, path: str) -> "CameraPose":
        data = np.load(path)
        return cls(R_cam=data["R_cam"], t_cam=data["t_cam"])


class TableCalibration:
    """
    Accumule les détections des tags de référence sur N frames, moyenne
    les coins pixels pour lisser le bruit, puis résout une pose caméra
    unique via solvePnP (SQPNP) sur les 4 tags x 4 coins.
    """

    def __init__(
        self,
        tag_extractor: TagExtractor,
        cam_mtx: np.ndarray,
        dist_coeffs: np.ndarray,
        ref_marker_size_mm: float = 100.0,
    ):
        self.tag_extractor = tag_extractor  # composition, pas héritage
        self.cam_mtx = cam_mtx
        self.dist_coeffs = dist_coeffs
        self.ref_marker_size_mm = ref_marker_size_mm

        self._ref_corners_3d = self._precompute_ref_corners_3d()
        self._calib_buffer: dict[int, list[np.ndarray]] = {tid: [] for tid in REF_TAG_IDS}

    def _precompute_ref_corners_3d(self) -> dict[int, np.ndarray]:
        """4 coins 3D (mm) par tag de ref, ordre ArUco TL, TR, BR, BL."""
        s = self.ref_marker_size_mm / 2.0
        corners_dict = {}
        for tag_id, center in REF_TAGS_DEFINITION.items():
            cx, cy, cz = center
            corners_dict[tag_id] = np.array([
                [cx - s, cy + s, cz],
                [cx + s, cy + s, cz],
                [cx + s, cy - s, cz],
                [cx - s, cy - s, cz],
            ], dtype=np.float32)
        return corners_dict

    def analyse_frame(self, frame: np.ndarray) -> bool:
        """
        Détecte les tags de ref sur `frame` et accumule leurs coins dans
        le buffer interne. Retourne True si les 4 tags de ref ont été vus
        sur CETTE frame précise (indicatif, pas une condition d'arrêt en
        soi — voir valid_frame_count pour l'état cumulé).
        """
        detected = self.tag_extractor.extract(frame, categories={TagCategory.TABLE_REF})

        seen_ids = set()
        for tag in detected:
            self._calib_buffer[tag.id].append(tag.corners)
            seen_ids.add(tag.id)

        return seen_ids == set(REF_TAG_IDS)

    @property
    def valid_frame_count(self) -> int:
        """
        Nombre de frames exploitables pour la calibration : le tag de ref
        le moins vu fait foi (il faut les 4 pour résoudre la pose).
        """
        return min(len(self._calib_buffer[tid]) for tid in REF_TAG_IDS)

    def finalize(self) -> CameraPose:
        """
        Moyenne les coins accumulés, résout la pose table (SQPNP), puis
        dérive et retourne la pose caméra (R_cam, t_cam).

        Lève ValueError si pas assez de données pour un des tags de ref,
        RuntimeError si solvePnP échoue.
        """
        final_obj_points = []
        final_img_points = []

        for tag_id in REF_TAG_IDS:
            corners_list = self._calib_buffer[tag_id]
            if len(corners_list) < MIN_VALID_FRAMES:
                raise ValueError(
                    f"Tag ref {tag_id}: seulement {len(corners_list)} frames valides "
                    f"(minimum requis: {MIN_VALID_FRAMES})"
                )
            avg_corners = np.mean(np.array(corners_list), axis=0)  # (4, 2)
            final_img_points.extend(avg_corners.reshape(-1, 2))
            final_obj_points.extend(self._ref_corners_3d[tag_id])

        success, rvec, tvec = cv2.solvePnP(
            np.array(final_obj_points, dtype=np.float32),
            np.array(final_img_points, dtype=np.float32),
            self.cam_mtx,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        if not success:
            raise RuntimeError("Échec solvePnP lors de la finalisation de la calibration")

        R_table, _ = cv2.Rodrigues(rvec)
        t_table = tvec.reshape(3, 1)

        # Inversion pose table->caméra en pose caméra->table
        R_cam = R_table.T
        t_cam = -R_cam @ t_table

        return CameraPose(R_cam=R_cam, t_cam=t_cam)


