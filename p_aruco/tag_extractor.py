"""
core/tag_extractor.py

TagExtractor : détection ArUco pure.

Contrat : frame (pixels) -> liste de (id, coins pixels, taille physique mm).

Les tags 36/47/41 (Jenga/bases) ne sont plus catalogués : détection ArUco
abandonnée au profit de YOLO. Si un besoin réapparaît, les rajouter dans
TAG_CATALOG avec une nouvelle TagCategory suffit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np


class TagCategory(Enum):
    TABLE_REF = auto()   # Tags fixes de calibration table (20-23)
    ROBOT = auto()        # Tags mât robot (1-10)


@dataclass(frozen=True)
class TagSpec:
    category: TagCategory
    size_mm: float


@dataclass
class DetectedTag:
    id: int
    corners: np.ndarray      # (4, 2) pixels, ordre ArUco : TL, TR, BR, BL
    size_mm: float
    category: TagCategory


class TagExtractor:
    """
    Détection ArUco pure. Ne connaît que : quels IDs sont valides, leur
    taille physique (mm), et les paramètres de détection.
    """

    # Règlement : robots ID 1-5 = équipe bleue, 6-10 = équipe jaune.
    # Tags de référence table : 20-23 (positions théoriques dans
    # calibration/calibration.py, pas ici — TagExtractor ne connaît que
    # la taille physique, pas la position sur la table).
    TAG_CATALOG: dict[int, TagSpec] = {
        **{i: TagSpec(TagCategory.TABLE_REF, 100.0) for i in (20, 21, 22, 23)},
        **{i: TagSpec(TagCategory.ROBOT, 70.0) for i in range(1, 11)},
    }

    def __init__(self, dictionary: int = cv2.aruco.DICT_4X4_50):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self.parameters = cv2.aruco.DetectorParameters()

        # --- Paramètres tunés empiriquement sur la table Eurobot ---
        # (repris tels quels de l'ancien code, ne pas retoucher sans raison)
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.polygonalApproxAccuracyRate = 0.03
        self.parameters.minMarkerPerimeterRate = 0.03
        self.parameters.errorCorrectionRate = 0.3

        # Un seul détecteur, réutilisé (l'ancien code en recréait un par
        # appel dans process_frame / calibration_table_analyse_frame).
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def extract(
        self,
        frame: np.ndarray,
        categories: Optional[set[TagCategory]] = None,
    ) -> list[DetectedTag]:
        """
        Détecte les tags ArUco sur `frame` et retourne uniquement ceux
        appartenant au catalogue connu (tout le reste = bruit, ignoré).

        categories : si fourni, restreint le retour à ces catégories
                     (ex : {TagCategory.ROBOT} en régime permanent,
                     {TagCategory.TABLE_REF} pendant la calibration).
                     None = toutes les catégories connues.
        """
        corners, ids, _rejected = self.detector.detectMarkers(frame)

        if ids is None:
            return []

        detected: list[DetectedTag] = []
        for i, id_val in enumerate(ids.flatten()):
            id_val = int(id_val)
            spec = self.TAG_CATALOG.get(id_val)

            if spec is None:
                continue  # tag non catalogué -> bruit

            if categories is not None and spec.category not in categories:
                continue

            detected.append(DetectedTag(
                id=id_val,
                corners=corners[i][0],  # (1,4,2) -> (4,2)
                size_mm=spec.size_mm,
                category=spec.category,
            ))

        return detected