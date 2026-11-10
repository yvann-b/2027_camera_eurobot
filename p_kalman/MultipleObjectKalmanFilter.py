from p_kalman.JengaTracking import EstimatedObject
from config.loader import Kalman_config
import numpy as np
import math
from scipy.optimize import linear_sum_assignment

class MultipleObjectKalmanFilter:
    def __init__(self, kalman_config):
        self.liste_object = []                       # list [dico] : liste de DetectedObject
        self.buffer_tempo_object = []               # list[dico[dico : int] liste de dictionnaire dont les clés sont des obj et les values un int (le nombre d'occurence)
        self.cov_init = kalman_config.initial_confidence
        self.mes_inc = kalman_config.covariance_mesurement
        self.mod_inc = kalman_config.covariance_model
        self.threshold_nohits = kalman_config.threshold_nohits
        self.threshold_fusion = kalman_config.threshold_fusion
        self.distance = kalman_config.distance

    def predict_kf(self):
        for object in self.liste_object:
            object.predict()

    def update_kf(self, measurements: list[dict]) -> None:

        no_objects_measurement = []


        # ASSIGNATION DES OBJETS CONNUS
        if len(self.liste_object) != 0 and len(measurements) != 0:
            cost_matrix = np.zeros((len(self.liste_object), len(measurements)))

            for i, est_obj in enumerate(self.liste_object):
                for j, measure in enumerate(measurements):
                    x_est, y_est = est_obj.projected_state[0], est_obj.projected_state[1]
                    x_obj = measure["x_mm"]
                    y_obj = measure["y_mm"]
                    cost_matrix[i][j] = math.sqrt((x_est - x_obj) ** 2 + (y_est - y_obj) ** 2)

            rows, cols = linear_sum_assignment(cost_matrix)

            # On va stocker ici les index des objets et mesures qui ont matché
            matched_object_indices = set()
            matched_measure_indices = set()

            for i in range(len(rows)):
                r_idx = rows[i]
                c_idx = cols[i]
                if cost_matrix[r_idx][c_idx] < self.distance :  # Seuil strict
                    self.liste_object[r_idx].update(measurements[c_idx])
                    matched_object_indices.add(r_idx)
                    matched_measure_indices.add(c_idx)

            for j, measure in enumerate(measurements):
                if j not in matched_measure_indices:
                    no_objects_measurement.append(measure)

            for i, est_obj in enumerate(self.liste_object):
                if i not in matched_object_indices:
                    est_obj.update(None)
        else:
            no_objects_measurement = measurements
            for est_obj in self.liste_object:
                est_obj.update(None)

        # ---> NOUVEAU : Suppression des objets morts <---
        # On ne garde que les objets dont number_nohits est <= 10
        self.liste_object = [obj for obj in self.liste_object if obj.number_nohits <= 10]

        # 3. ASSIGNATION DU BUFFER TEMPORAIRE
        matched_buffer_indices = set()
        matched_measure_indices = set()
        indices_to_promote = []

        if len(self.buffer_tempo_object) != 0 and len(no_objects_measurement) != 0:
            cost_matrix_buf = np.zeros((len(self.buffer_tempo_object), len(no_objects_measurement)))

            for i, temp_dict in enumerate(self.buffer_tempo_object):
                temp_obj = temp_dict["mesure"]
                for j, obj in enumerate(no_objects_measurement):
                    x_temp = temp_obj["x_mm"]
                    y_temp = temp_obj["y_mm"]
                    x_meas = obj["x_mm"]
                    y_meas = obj["y_mm"]
                    cost_matrix_buf[i][j] = math.sqrt((x_temp - x_meas) ** 2 + (y_temp - y_meas) ** 2)

            rows_b, cols_b = linear_sum_assignment(cost_matrix_buf)

            for i in range(len(rows_b)):
                r_idx = rows_b[i]
                c_idx = cols_b[i]

                if cost_matrix_buf[r_idx][c_idx] < 10:
                    self.buffer_tempo_object[r_idx]["mesure"] = no_objects_measurement[c_idx]
                    self.buffer_tempo_object[r_idx]["hits"] += 1

                    matched_buffer_indices.add(r_idx)
                    matched_measure_indices.add(c_idx)

                    if self.buffer_tempo_object[r_idx]["hits"] >= 3:
                        indices_to_promote.append(r_idx)

        # 4. CRÉATION DES VRAIS OBJETS KF ET RECONSTRUCTION DU BUFFER
        for idx in indices_to_promote:
            mes = self.buffer_tempo_object[idx]["mesure"]
            nouvel_objet_kf = EstimatedObject(mes["class"], mes["x_mm"], mes["y_mm"], mes["theta_rad"], self.cov_init,
                                              self.mes_inc, self.mod_inc)
            self.liste_object.append(nouvel_objet_kf)

        nouveau_buffer = []

        for i, temp_obj in enumerate(self.buffer_tempo_object):
            if i in matched_buffer_indices and i not in indices_to_promote:
                nouveau_buffer.append(temp_obj)

        for j, measure in enumerate(no_objects_measurement):
            if j not in matched_measure_indices:
                nouveau_buffer.append({"mesure": measure, "hits": 1})

        self.buffer_tempo_object = nouveau_buffer

    def get_augmented_estimate_state(self):
        liste_state = []        # Liste de dictionnaire : {x_mm : ... y, t , color}
        for obj in self.liste_object:
            liste_state.append(obj.get_augmented_state())
        return liste_state