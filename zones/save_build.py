"""
La définition géométrique des zones est faite dans le .config
Ici c'est juste les fonctions pour sauvegarder/charger le fait que la zone soit visible ou pas (calculé par initialisation, exploitée par post-traitement)
"""

from config.loader import ZoneHeader, Zone
import numpy as np
from p_math.changement_repere import from_coordinate_to_pixel

def build_zones(headers: list[ZoneHeader], visibility_path: str) -> dict[int, Zone]:
    """Appelé une fois au démarrage de chaque process consommateur."""
    vis = load_zone_visibility(visibility_path)
    return {
        h.id: Zone(header=h, coordinates_center_pixel=vis[h.id][0], is_visible=vis[h.id][1])
        for h in headers
    }

def save_zone_visibility(headers: list[ZoneHeader], R_cam, T_cam, cam_mtx, roi, path: str) -> None:
    """Appelé une seule fois, par initialisation, juste avant qu'il termine."""
    ids = np.array([h.id for h in headers], dtype=np.int32)
    centers = np.array([h.center for h in headers], dtype=np.float64)  # (N, 3)
    pixels = np.array(
        [from_coordinate_to_pixel(cam_mtx, c, R_cam, T_cam) for c in centers]
    )  # (N, 2)

    x, y, w, h = roi
    visible = (x <= pixels[:, 0]) & (pixels[:, 0] < x + w) & \
              (y <= pixels[:, 1]) & (pixels[:, 1] < y + h)

    np.savez(path, zone_id=ids, coordinates_center_pixel=pixels, is_visible=visible)


def load_zone_visibility(path: str) -> dict[int, tuple[tuple[float, float], bool]]:
    data = np.load(path)
    for k in data.files:
        print(k, data[k].shape, data[k].dtype)

    return {
        int(zid): (tuple(px), bool(vis))
        for zid, px, vis in zip(data["zone_id"], data["coordinates_center_pixel"], data["is_visible"])
    }