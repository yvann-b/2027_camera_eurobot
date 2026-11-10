from p_math.changement_repere import from_coordinate_to_pixel
from config.loader import Homographie_config
import numpy as np
import cv2

def calculer_homographie_via_pose(new_camera_mtx, R_cam, t_cam,homo_config : Homographie_config):
    """
        Calcule la matrice H pour le traitement YOLO.

        Mapping :
        - Coin Table (0,0) -> Pixel Image (0,0) [Haut-Gauche]
        - Axe X Table      -> Axe X Image (Vers la droite)
        - Axe Y Table      -> Axe Y Image (Vers le bas)
        """

    # Les 4 coins de la table en Mètres (Monde 3D)
    # Ordre : 0->1->2->3
    TABLE_X = homo_config.width
    TABLE_Y = homo_config.height
    RATIO_MM = homo_config.ratio_mm_per_pixel
    marge_secu = homo_config.security_margin


    # Largeur Totale = 3.0 + 0.1 (gauche) + 0.1 (droite) = 3.2m
    world_width_mm = (TABLE_X + 2*marge_secu)
    world_height_mm = (TABLE_Y + 2*marge_secu)

    # Calcul des dimensions PIXELS exactes
    W_out = int(world_width_mm / RATIO_MM)  # Devrait donner 1600 px
    H_out = int(world_height_mm / RATIO_MM)  # Devrait donner 1100 px


    coins_table_3d = [
        np.array([[-marge_secu], [-marge_secu], [0.0]]),  # Coin 0 (0,0)
        np.array([[TABLE_X + marge_secu], [-marge_secu], [0.0]]),  # Coin 1 (3,0)
        np.array([[TABLE_X + marge_secu], [TABLE_Y+marge_secu], [0.0]]),  # Coin 2 (3,2)
        np.array([[-marge_secu], [TABLE_Y+marge_secu], [0.0]])  # Coin 3 (0,2)
    ]
    # Projection des coins théoriques sur le capteur (Source)
    pts_source_px = []
    for P_monde in coins_table_3d:
        pts_source_px.append(        from_coordinate_to_pixel(   camera_mtx=new_camera_mtx,
                                    R_cam=R_cam,
                                    T_cam=t_cam,
                                    points_table=P_monde,
                                    )
)

    pts_source_px = np.array(pts_source_px, dtype=np.float32)

    # On force le Coin 0 à aller en (0,0)

    pts_dst_px = np.array([
        [0, 0],  # Coin 0 (0,0)   -> Pixel HAUT-GAUCHE
        [W_out, 0],  # Coin 1 (3,0)   -> Pixel HAUT-DROITE
        [W_out, H_out],  # Coin 2 (3,2)   -> Pixel BAS-DROITE
        [0, H_out]  # Coin 3 (0,2)   -> Pixel BAS-GAUCHE
    ], dtype=np.float32)

    # Calcul H
    H = cv2.getPerspectiveTransform(pts_source_px, pts_dst_px)

    return H, W_out, H_out