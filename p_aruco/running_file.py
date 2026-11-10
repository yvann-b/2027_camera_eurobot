"""
p_aruco/robot_tracking.py

RobotTagTracking : localisation des robots (tags 1-10) sur la table, en
régime permanent (après calibration).

Décision archi (point 1) : au lieu de refaire une intersection rayon/plan
par pixel à chaque frame (ce que faisait Robot.compute_table_position dans
l'ancien code), on dérive UNE FOIS, à la construction, une homographie H_h
à partir de la pose caméra (R_cam, t_cam) produite par TableCalibration.
En régime permanent, localiser un tag devient un simple produit matriciel
+ division homogène sur ses 4 coins pixels.

Deux hauteurs de tag : le règlement autorise l'ajout d'un déflecteur entre
le tag et le sommet du robot, ce qui peut changer la hauteur du tag selon
l'équipe (bleue / jaune). On dérive donc DEUX homographies (une par
hauteur), mais la détection ArUco elle-même (l'étape coûteuse) ne se fait
qu'UNE SEULE FOIS par frame, pour tous les robots confondus. Le choix de
l'homographie par tag se fait après coup, via son ID -> équipe -> hauteur,
un simple lookup dict — pas de second passage de détection.

Contrat : frame -> dict[int, RobotPose].
Aucun état conservé d'une frame à l'autre (pas d'historique de trajectoire
ici — cf. décision archi : le suivi temporel est le rôle du Kalman filter
en aval du pipeline, pas de ce module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from p_aruco.tag_extractor import TagExtractor, TagCategory
from p_aruco.calibration import CameraPose

# Pour le débug :
import cv2



ROBOT_MAST_HEIGHT_MM = 430.0  # Hauteur réglementaire par défaut (sans déflecteur)

DEFAULT_HEIGHTS_MM: dict[str, float] = {
    "BLUE": ROBOT_MAST_HEIGHT_MM,
    "YELLOW": ROBOT_MAST_HEIGHT_MM,
}


@dataclass(frozen=True)
class RobotPose:
    id: int
    team: str          # "BLUE" (1-5) / "YELLOW" (6-10)
    x_mm: float
    y_mm: float
    angle_deg: float


def _team_for_id(robot_id: int) -> str:
    # Règlement : Blue = 1-5, Yellow = 6-10
    if 1 <= robot_id <= 5:
        return "BLUE"
    if 6 <= robot_id <= 10:
        return "YELLOW"
    return "UNKNOWN"


class RobotTagTracking:
    """
    Localisation des tags robot (1-10) sur la table, en régime permanent.

    IMPORTANT : `cam_mtx` doit être la MÊME matrice intrinsèque que celle
    utilisée pour la calibration (new_camera_mtx, i.e. calib_frame.new_mtx
    dans main.py) — puisque les frames du ring buffer sont déjà
    undistordues via map1/map2 en amont. Utiliser la matrice caméra brute
    ici introduirait une erreur systématique de projection.
    """

    def __init__(
        self,
        tag_extractor: TagExtractor,
        cam_mtx: np.ndarray,
        camera_pose: CameraPose,
        heights_mm: Optional[dict[str, float]] = None,
    ):
        self.tag_extractor = tag_extractor  # composition, pas héritage

        self.heights_mm: dict[str, float] = dict(DEFAULT_HEIGHTS_MM)
        if heights_mm:
            self.heights_mm.update(heights_mm)

        # Dédoublonnage : si les deux équipes ont la même hauteur (pas de
        # déflecteur cette année), on ne dérive qu'UNE SEULE homographie
        # au lieu de calculer deux fois la même matrice.
        distinct_heights = set(self.heights_mm.values())
        self._homography_by_height: dict[float, np.ndarray] = {
            h: self._derive_homography(cam_mtx, camera_pose, h) for h in distinct_heights
        }

    @staticmethod
    def _derive_homography(cam_mtx: np.ndarray, camera_pose: CameraPose, height_mm: float) -> np.ndarray:
        """
        Dérive H (3x3) telle que, pour un pixel homogène q = [u, v, 1]^T :
            p = H @ q ;  X = p[0]/p[2] ;  Y = p[1]/p[2]
        donne directement (X, Y) en mm sur la table, au plan z=height_mm.

        Dérivation (équivalente à l'intersection rayon/plan de l'ancien
        compute_table_position, mais réduite à une forme fermée, vérifiée
        numériquement équivalente à moins de 1e-10mm d'écart) :
        soit d = K^-1 @ q la direction du rayon caméra pour le pixel q,
        et e = R_cam @ d cette direction exprimée dans le repère table.
        L'intersection avec le plan z = height_mm donne :
            X = (height_mm - t_cam_z) * e[0]/e[2] + t_cam_x
            Y = (height_mm - t_cam_z) * e[1]/e[2] + t_cam_y
        Ce qui se réécrit comme une unique homographie H = D @ R_cam @ K^-1,
        D étant la matrice de mise à l'échelle/translation ci-dessous.
        """
        K_inv = np.linalg.inv(cam_mtx)
        M = camera_pose.R_cam @ K_inv  # (3, 3)

        t_cam = camera_pose.t_cam.flatten()
        tcx, tcy, tcz = t_cam[0], t_cam[1], t_cam[2]

        D = np.array([
            [height_mm - tcz, 0.0, tcx],
            [0.0, height_mm - tcz, tcy],
            [0.0, 0.0, 1.0],
        ])

        H = D @ M

        # Fail-fast : une CameraPose non initialisée (ex: R_cam resté à
        # zéro parce que le fichier de calibration n'a jamais été chargé)
        # produit une homographie dégénérée -> divisions par zéro -> NaN
        # silencieux publiés sur le bus. On préfère planter ici, au
        # démarrage du process, plutôt que de laisser passer des poses
        # NaN vers le Kalman filter en aval.
        if abs(np.linalg.det(H)) < 1e-9:
            raise ValueError(
                f"Homographie dégénérée pour height_mm={height_mm} (det≈0). "
                f"CameraPose non initialisée (R_cam nul ?) ou cam_mtx invalide."
            )

        return H

    @staticmethod
    def _apply_homography(pixels: np.ndarray, H: np.ndarray) -> np.ndarray:
        """pixels: (N, 2) pixels -> (N, 2) mm sur la table, via l'homographie H."""
        n = pixels.shape[0]
        homog_px = np.hstack([pixels, np.ones((n, 1))])  # (N, 3)
        projected = homog_px @ H.T                        # (N, 3)
        w = projected[:, 2:3]
        return projected[:, :2] / w

    def track(self, frame: np.ndarray, is_printing : bool) -> dict[int, RobotPose]:
        """
        Détecte les tags robot sur `frame` (UNE SEULE extraction ArUco,
        tous robots confondus) et retourne leur position table, chacun
        projeté avec l'homographie correspondant à la hauteur de SON
        équipe. Dict vide si aucun robot vu (jamais de None).
        """
        detected = self.tag_extractor.extract(frame, categories={TagCategory.ROBOT})

        poses: dict[int, RobotPose] = {}
        for tag in detected:
            team = _team_for_id(tag.id)
            height = self.heights_mm.get(team, ROBOT_MAST_HEIGHT_MM)
            H = self._homography_by_height[height]


            table_corners = self._apply_homography(tag.corners, H)  # (4, 2) mm

            center = table_corners.mean(axis=0)

            # Angle : vecteur du coin TL vers TR, projeté sur la table
            delta = table_corners[1] - table_corners[0]
            angle_deg = np.degrees(np.arctan2(delta[1], delta[0]))

            poses[tag.id] = RobotPose(
                id=tag.id,
                team=team,
                x_mm=float(center[0]),
                y_mm=float(center[1]),
                angle_deg=float(angle_deg),
            )

            # 1. Récupérer les coins et les convertir en entiers (int32)
            corners = np.array(tag.corners, dtype=np.int32)

            # 2. Dessiner le cadre (polygone) reliant les 4 points
            # isClosed=True permet de relier le dernier point au premier
            cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=3)

            # 3. Dessiner chaque point (les 4 coins)
            for pt in corners:
                # pt est un tableau [x, y], on le convertit en tuple (x, y) pour OpenCV
                cv2.circle(frame, tuple(pt), radius=8, color=(0, 0, 255), thickness=-1)


        if is_printing :
            frame = cv2.resize(frame, (720, int(720 * 2048/2448)), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Debug Tag", frame)
            cv2.waitKey(1)
        return poses