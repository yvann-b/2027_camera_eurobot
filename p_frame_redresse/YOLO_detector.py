import numpy as np
from ultralytics import YOLO
from config.loader import Subframe_Configuration,YOLO_config


class YoloDetector:
    def __init__(self, YOLO_config):
        print(f"Chargement du modèle YOLO : {YOLO_config.model}...")
        self.model = YOLO(YOLO_config.model, task='obb')
        self.conf_threshold = YOLO_config.confiance

        # Classes cibles pour la 3D
        self.target_classes = YOLO_config.target_class
        self.object_height = YOLO_config.height  # Hauteur des objets

        # Modification dans YoloDetector
    def analyze_batch(self, dico_image : dict[Subframe_Configuration,np.ndarray]):
            """
            Traite une liste d'images en une seule passe GPU (Batching).
            """
            all_detected_objects = []  # Liste de listes (une liste d'objets par image)

            # Inférence sur la liste complète
            # verbose=False pour éviter le spam console
            configurations = list(dico_image.keys())
            images = list(dico_image.values())

            # Inférence sur la liste complète des images
            results = self.model.predict(images, imgsz=640, device="0", verbose=False)
            for config, r in zip(configurations, results):
                objects_in_current_frame = []

                if r.obb is not None:

                    offset_x_mm = config.corners[1][0]  # Coordonnées du coin haut gauche de la subframe
                    offset_y_mm = config.corners[1][1]

                    # Transfert sur GPU pour le filtrage
                    xywhr = r.obb.xywhr
                    conf = r.obb.conf
                    cls = r.obb.cls

                    mask = conf >= self.conf_threshold
                    valid_xywhr = xywhr[mask]

                    if len(valid_xywhr) > 0:
                        # Calcul avec l'offset spécifique à cette subframe
                        x_pixels = valid_xywhr[:, 0]
                        y_pixels = valid_xywhr[:, 1]
                        widths = valid_xywhr[:, 2]
                        heights = valid_xywhr[:, 3]
                        thetas_r = valid_xywhr[:, 4]  # L'angle en radians

                        # Convertir l'angle pour représenter la direction de la largeur (plus petit côté)
                        # YOLO retourne l'angle du côté le plus long, on ajoute π/2 si h > w
                        mask_height_gt_width = heights > widths
                        thetas_r[mask_height_gt_width] += np.pi / 2

                        # Inverser theta car on inverse l'axe Y (image Y down → table Y up)
                        thetas_r = -thetas_r

                        # Le calcul matriciel reste sur le GPU
                        x_table_mm = (x_pixels * config.homo_config.ratio_mm_per_pixel) + offset_x_mm
                        y_table_mm = offset_y_mm - (y_pixels * config.homo_config.ratio_mm_per_pixel)

                        x_cpu = x_table_mm.cpu().numpy()
                        y_cpu = y_table_mm.cpu().numpy()
                        theta_cpu = thetas_r.cpu().numpy()

                        # On n'oublie pas de récupérer les classes et confiances masquées
                        valid_cls = cls[mask].cpu().numpy().astype(int)
                        valid_conf = conf[mask].cpu().numpy()

                        # 2. On construit nos objets
                        for i in range(len(x_cpu)):
                            obj = {
                                "x_mm": float(x_cpu[i]),
                                "y_mm": float(y_cpu[i]),
                                "theta_rad": float(theta_cpu[i]),
                                "class": self.model.names[valid_cls[i]],
                                "confidence": float(valid_conf[i])
                            }

                            objects_in_current_frame.append(obj)

                        # On ajoute les objets de cette subframe à la liste GLOBALE
                        # On utilise .extend() et non .append() pour avoir une seule liste plate à la fin
                    all_detected_objects.extend(objects_in_current_frame)

            return all_detected_objects


