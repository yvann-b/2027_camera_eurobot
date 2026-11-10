"""
Ce fichier regroupe toutes les maths de changement de repère
Pour rappel il y a 3 types de repères :
image : plan du photorécepteur de la caméra, 2D, matrice frame directement
repère de la table : repère XYZ en 00 (coin bas gauche de la table) classique
repère de la caméra : repère XYZ en le centre optique de la caméra, simplement matrice de changement de repère (rotation + translation) (les R_cam et T_cam)
"""

import numpy as np


# ==============================================
# Changement de repère table <-> caméra (3D - 3D)
# ==============================================
def from_rcam_to_rtable(T_cam, R_cam, points_cam):
    """
    Passe un point (ou vecteur) du repère caméra au repère table.
    """
    pts, single = _as_column(points_cam)
    points_table = (R_cam @ pts) + T_cam
    return points_table.reshape(3) if single else points_table


def from_rtable_to_rcam(T_cam, R_cam, points_table):
    """
    Passe un point du repère table au repère caméra.
    """
    # Mathématiquement: P_cam = R_cam^-1 * (P_table - T_cam)
    # Pour une matrice de rotation, l'inverse est la transposée (.T)
    pts, single = _as_column(points_table)
    points_cam = R_cam.T @ (pts - T_cam)
    return points_cam.reshape(3) if single else points_cam


# ==============================================
# Pixel to mm / mm to Pixel
# ==============================================
def from_pixel_to_direction(camera_mtx, pixel, R_cam):
    """
    Calcule le VECTEUR DIRECTEUR du rayon lumineux dans le repère table.
    Note: On n'a pas besoin de T_cam ici car on oriente juste un vecteur.
    """
    augmented_state = np.array([pixel[0], pixel[1], 1.0])

    # 1. Vecteur direction dans le repère caméra (np.linalg.inv !)
    cam_vct_dir = np.linalg.inv(camera_mtx) @ augmented_state

    # 2. Vecteur direction orienté dans le repère table (seulement la rotation)
    table_vct_dir = R_cam @ cam_vct_dir
    return table_vct_dir


def from_pixel_to_coordinate(camera_mtx, pixel, R_cam, T_cam, height=0.0):
    """
    Trouve l'intersection du rayon lumineux avec un plan de hauteur donnée.
    """
    table_vct_dir = from_pixel_to_direction(camera_mtx, pixel, R_cam)

    # Equation paramétrique de la droite : P = T_cam + lamb * table_vct_dir
    # On isole lambda pour que P_z = height
    lamb = float((height - T_cam[2, 0]) / table_vct_dir[2])

    # On calcule le point d'intersection final
    P = T_cam.reshape(3) + lamb * table_vct_dir
    return P


def from_coordinate_to_pixel(camera_mtx, points_table, R_cam, T_cam):
    """
    Projette un point 3D de la table sur le capteur 2D de la caméra.
    """
    # 1. On ramène le point 3D dans le repère de la caméra
    p_cam = from_rtable_to_rcam(T_cam, R_cam, points_table)
    pts_cam, single = _as_column(p_cam)

    # 2. On projette avec la matrice intrinsèque
    augmented_state = camera_mtx @ pts_cam

    # 3. DIVISION PERSPECTIVE ! (Indispensable pour écraser la 3D vers la 2D)
    u = augmented_state[0] / augmented_state[2]
    v = augmented_state[1] / augmented_state[2]

    uv = np.stack([u, v])  # (2, N)
    return uv.reshape(2) if single else uv


def _as_column(points: np.ndarray, dim: int = 3) -> tuple[np.ndarray, bool]:
    """Normalise un point seul (dim,) ou un batch déjà en colonnes (dim, N).
    Retourne (array (dim, N), était_un_point_seul)."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        assert arr.shape == (dim,), f"point seul: attendu ({dim},), reçu {arr.shape}"
        return arr.reshape(dim, 1), True
    assert arr.shape[0] == dim, f"batch: attendu ({dim}, N), reçu {arr.shape}"
    return arr, False
