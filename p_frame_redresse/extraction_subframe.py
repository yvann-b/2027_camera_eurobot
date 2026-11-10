import numpy as np

def extract_subframe_mm(frame: np.ndarray, corners_mm: list[tuple[float, float]],
                         security_margin: float, ratio_mm_per_pixel: float) -> np.ndarray:
    h_img, w_img = frame.shape[:2]

    xs_mm = [c[0] for c in corners_mm]
    ys_mm = [c[1] for c in corners_mm]

    xs_mm_shifted = [x + security_margin for x in xs_mm]
    ys_mm_shifted = [y + security_margin for y in ys_mm]

    xs_px = [x / ratio_mm_per_pixel for x in xs_mm_shifted]
    ys_px = [y / ratio_mm_per_pixel for y in ys_mm_shifted]

    x_min = max(0, int(np.floor(min(xs_px))))
    x_max = min(w_img, int(np.ceil(max(xs_px))))

    # frame_map est flippé (row 0 = y_mm max, cohérent avec Y-up table).
    # Donc y_mm max -> row min, y_mm min -> row max : on inverse ET on échange min/max.
    y_px_min = min(ys_px)   # correspond au bord "bas" de la zone en mm
    y_px_max = max(ys_px)   # correspond au bord "haut" de la zone en mm

    y_min = max(0, int(np.floor(h_img - y_px_max)))
    y_max = min(h_img, int(np.ceil(h_img - y_px_min)))

    if x_min >= x_max or y_min >= y_max:
        raise ValueError(f"Zone hors image ou dégénérée : x=[{x_min},{x_max}] y=[{y_min},{y_max}]")

    return frame[y_min:y_max, x_min:x_max]