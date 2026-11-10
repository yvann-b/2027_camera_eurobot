import time

from ring.estimation_ring import DetectionRingBuffer
import cv2
import numpy as np
import math
import multiprocessing as mp

class MapDebugger:
    def __init__(self, scale=0.3):
        """
        scale: Facteur d'échelle (0.3 = fenêtre de 900x600 pixels pour une carte 3000x2000)
        """
        self.scale = scale
        self.map_w = int(3000 * self.scale)
        self.map_h = int(2000 * self.scale)
        self.window_name = "Debug Map"

        # Création de la fenêtre (optionnel mais propre)
        cv2.namedWindow(self.window_name)

    def update(self, robots_data):
        """
        Met à jour et affiche la carte de manière NON BLOQUANTE.
        robots_data: list de dict [{"x_mm": ..., "y_mm": ..., "theta_r": ..., "color": ...}]
        """
        # Créer un fond gris foncé (léger en mémoire)
        img = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 40

        # Dessiner la grille (optionnel, tous les 500mm)
        step = int(500 * self.scale)
        for i in range(0, self.map_w, step):
            cv2.line(img, (i, 0), (i, self.map_h), (60, 60, 60), 1)
        for i in range(0, self.map_h, step):
            cv2.line(img, (0, i), (self.map_w, i), (60, 60, 60), 1)

        # Dessiner chaque robot/point
        for data in robots_data:
            if data is not None :
                # Conversion en pixels
                x_px = int(data["x_mm"] * self.scale)
                # Inversion de l'axe Y (le (0,0) d'une image est en haut à gauche, en math c'est en bas à gauche)
                y_px = self.map_h - int(data["y_mm"] * self.scale)

                theta = data["theta_r"]

                # Choix de la couleur au format BGR (Bleu, Vert, Rouge)
                color = (255, 0, 0) if data["color"] == "blue" else (0, 255, 255)  # blue ou yellow

                # Dessin du point central
                cv2.circle(img, (x_px, y_px), 6, color, -1)

                # Dessin de la ligne de direction (longueur visuelle : 30 pixels)
                line_len = 30
                end_x = int(x_px + line_len * math.cos(theta))
                end_y = int(y_px - line_len * math.sin(theta))  # - car Y inversé

                cv2.line(img, (x_px, y_px), (end_x, end_y), (255, 255, 255), 2)

            # Affichage
        cv2.imshow(self.window_name, img)

        # LA MAGIE NON BLOQUANTE EST ICI :
        # Laisse 1ms à OpenCV pour dessiner, puis rend la main au programme principal
        cv2.waitKey(1)

    # ==========================================


def process_plotting_kf(ring : DetectionRingBuffer, start_event : mp.Event, stop_event : mp.Event):

    while not(start_event.is_set()) :
        time.sleep(0.01)

    while not(stop_event.is_set()) :
        debugger = MapDebugger(scale=0.4)  # Fenêtre de 1200 x 800
        msg = ring.read_latest()
        if msg is not None:
            liste_estimate_state = msg["detections"]
            if len(liste_estimate_state) >0 :
                #print(liste_estimate_state)
                debugger.update(liste_estimate_state)

