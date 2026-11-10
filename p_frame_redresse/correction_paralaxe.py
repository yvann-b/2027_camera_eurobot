import numpy as np

def correct_parallax(mesure_xy: np.ndarray, camera_xyz: np.ndarray, hauteur_objet: float) -> np.ndarray:
    """mesure_xy : (N, 2) positions brutes issues de l'homographie
    camera_xyz : (Cx, Cy, Cz), même repère/unités que slot_coords (mm, table)
    hauteur_objet : hauteur du point détecté au-dessus du plan table (mm)"""
    cam_xy = camera_xyz[:2]
    facteur = (camera_xyz[2] - hauteur_objet) / camera_xyz[2]  # < 1, rapproche le point de la caméra
    return cam_xy + (mesure_xy - cam_xy) * facteur

def parallax_list (mesure: list, camera_xyz: np.ndarray, hauteur_objet: float) :
    output = []
    camera_xyz_flat = camera_xyz.flatten()
    for detec in mesure:
        mesure_xy = np.array([float(detec["x_mm"]), float(detec["y_mm"])])
        new_xy = correct_parallax(mesure_xy, camera_xyz_flat, hauteur_objet)
        detec["x_mm"] = float(new_xy[0])
        detec["y_mm"] = float(new_xy[1])
        output.append(detec)
    return output