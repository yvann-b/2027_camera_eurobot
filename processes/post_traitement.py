import time

from bus_zmq.zmq_bus import Subscriber
from ring.estimation_ring import DetectionRingBuffer
from ring.zone_obstruction import ZoneObstructionRing
from p_post_traitement.merge_info import merge_branches
from config.loader import ZoneHeader, Jenga
from zones.save_build import build_zones
from config.loader import CalibrationConfig, Zone, ZoneHeader, ExternalBusConfig
from p_post_traitement.creation_cost_matrix import creation_cost_matrix
from p_post_traitement.p_get_information_dico import build_robot_payload

from scipy.optimize import linear_sum_assignment
from bus_zmq.robot_bus_client import RobotBusClient

import multiprocessing as mp
import time
import numpy as np
import os
from shapely.geometry import Polygon
import math

def process_post_traitement(
                            estimation_ring : DetectionRingBuffer,
                            obstruction_ring : ZoneObstructionRing,
                            start_event : mp.Event,
                            stop_event : mp.Event,
                            zones_header : dict[int, ZoneHeader],
                            param_calib : CalibrationConfig,
                            jenga : Jenga,
                            external_bus : ExternalBusConfig,


):



    """
    On va créer les sous-slots pour chaque zone de ramassage (pour les assigner à la zone si ça rentre)
    """


    while not(start_event.is_set()) :
        time.sleep(0.01)

    bus_externe = RobotBusClient(
        ip=external_bus.ip, port=external_bus.port, topic=external_bus.topic,
        enabled=external_bus.enabled, sndhwm=external_bus.sndhwm, linger_ms=external_bus.linger_ms,
    )
    os.makedirs(param_calib.output_calib_file, exist_ok=True)               # Pour créer le dossier s'il n'existe pas déjà
    chemin_complet1 = os.path.join(param_calib.output_calib_file, "zones_visiblity.npz")

    liste_zones_header = []
    for key in zones_header.keys():
        liste_zones_header.append(zones_header[key])

    dico_zones = build_zones(headers= liste_zones_header,visibility_path= chemin_complet1 )     # Dict[int , Zone]
    last_frame_id_aruco = None
    last_frame_id_detection = None


    while not(stop_event.is_set()) :
        zone_record, det_record = merge_branches(estimation_ring,obstruction_ring, int(1e9))
        if zone_record is not None and det_record is not None :


            data_zone = {"frame_id" : zone_record.frame_id,"timestamp_ns" : zone_record.timestamp_ns,"data" : zone_record.is_blocked}
            data_record = {"frame_id" : det_record["frame_id"],"timestamp_ns" : det_record["timestamp_ns"],"data" : det_record["detections"]}
            estimations = data_record["data"]
            temp_aruco = data_zone["frame_id"]
            temp_detection = data_record["frame_id"]
            if temp_aruco == last_frame_id_aruco and temp_detection == last_frame_id_detection:
                time.sleep(0.001)  # Small sleep to avoid busy-wait
                continue


            """
            {'frame_id': 1366, 'timestamp_ns': 1788359525920801100, 'data': array([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True])}
            {'frame_id': 1362, 'timestamp_ns': 1788359525882894100, 'data': array([(     1060.4,      768.31,   0.0035314, 'yellow'), (     1940.7,      769.36,    0.015322, 'yellow'), (     1786.4,      768.33,    0.026276, 'yellow'), (     1887.4,      776.07,   0.0099197, 'blue'), (     1165.7,      769.73,   0.0044111, 'yellow'), (     1113.1,      770.17,  -0.0087743, 'blue'),
       (     1837.4,      769.21,    0.017006, 'blue'), (       1217,      775.86,   -0.011985, 'blue'), (     1888.2,      142.61,    0.064629, 'yellow'), (     1838.8,      143.33,    0.064996, 'blue'), (     1991.6,      142.72,    0.030592, 'yellow'), (       1941,      147.42,    0.058868, 'blue'),
       (     158.32,      1104.3,   -0.020746, 'blue'), (     156.33,      1205.8,   0.0026278, 'yellow'), (     155.31,      1153.9,  -0.0048436, 'blue'), (     156.65,      1260.1, -0.00059587, 'yellow'), (     1166.9,      132.96,   -0.030291, 'yellow'), (     1115.3,      131.88,   -0.034582, 'blue'),
       (     1060.9,      132.61,   -0.027143, 'yellow'), (       1012,      133.17,   -0.051392, 'blue')], dtype=[('x_mm', '<f8'), ('y_mm', '<f8'), ('theta_r', '<f8'), ('color', '<U10')])}
            """

            for i,state in enumerate(data_zone["data"]):
                dico_zones[i].is_blocked = state
                """ 
                Dico zone est donc bien construit ici : 
                clé : id de la zone
                value : Zone, avec le champ is visible chargé depuis l'init et is_blocked récupéré depuis le raycasting    
                """
                if dico_zones[i].header.type == "depot":
                    dico_zones[i].number_blue = 0
                    dico_zones[i].number_yellow = 0
            slot_coords, slot_zone_id, slot_local_idx = [], [], []

            for zone_id, zone in sorted(dico_zones.items()):
                if zone.header.type == "ramassage":
                    assert len(zone.header.slots) == 4, \
                        f"zone {zone_id}: {len(zone.header.slots)} slots, attendu 4"
                    for local_idx, coord in enumerate(zone.header.slots):
                        slot_coords.append(coord)
                        slot_zone_id.append(zone_id)
                        slot_local_idx.append(local_idx)


            colors = np.array([d[3] for d in data_record["data"]])
            EMPTY = ""

            slot_coords = np.array(slot_coords)  # shape (n_slots, 2) — sert à construire le cost_matrix chaque frame
            slot_zone_id = np.array(slot_zone_id)  # shape (n_slots,)  — table de correspondance figée
            slot_local_idx = np.array(slot_local_idx)  # shape (n_slots,)

            estimations = data_record["data"]
            #print(f"{len(estimations)} détections cette frame")  # à garder un moment, pas juste pour ce crash

            if len(estimations) == 0:
                output = {zid: [EMPTY] * 4 for zid in np.unique(slot_zone_id)}
                unassigned_detections = []
            else:
                cost_mtx = creation_cost_matrix(dico_measurement=data_record, liste_slot=slot_coords)
                rows, cols = linear_sum_assignment(cost_mtx)
                dists = cost_mtx[rows, cols]
                valid_mask = dists < 25
                valid_r, valid_c = rows[valid_mask], cols[valid_mask]

                colors = np.array([d[3] for d in estimations])
                output = {zid: [EMPTY] * 4 for zid in np.unique(slot_zone_id)}
                for r, c in zip(valid_r, valid_c):
                    output[slot_zone_id[c]][slot_local_idx[c]] = colors[r]

                matched_measures = set(valid_r.tolist())
                unassigned_detections = [
                    estimations[i] for i in range(len(estimations)) if i not in matched_measures
                ]


                for idx in dico_zones.keys():
                    zone = dico_zones[idx]
                    if not(zone.is_visible) :
                        output[i] = ["unknown"] * 4

                for us_det in unassigned_detections:
                    length = jenga.length
                    width = jenga.width
                    x_center, y_center, theta = us_det[0], us_det[1], us_det[2]

                    cos_t, sin_t = math.cos(theta), math.sin(theta)
                    half_l, half_w = length / 2, width / 2

                    local_corners = [(-half_l, -half_w), (half_l, -half_w), (half_l, half_w), (-half_l, half_w)]
                    rect2 = Polygon([
                        (x_center + lx * cos_t - ly * sin_t, y_center + lx * sin_t + ly * cos_t)
                        for lx, ly in local_corners
                    ])

                    for zone in dico_zones.values():
                        if zone.header.type == "depot" :
                            rect1 = Polygon(zone.header.corners)
                            if rect1.intersects(rect2) :
                                if us_det[3] == "blue" :
                                    zone.number_blue += 1
                                elif us_det[3] == "yellow" :
                                    zone.number_yellow += 1

                for i, zone in enumerate(dico_zones.values()):
                    if zone.header.type == "depot":
                        if zone.is_blocked or not (
                        zone.is_visible):  # Si la zone n'est pas visible à la caméra ou si elle est masquée par un robot
                            zone.dominant_color = "Unknown"
                        elif zone.number_blue == 0 and zone.number_yellow == 0:
                            zone.dominant_color = "EMPTY"
                        elif zone.number_blue - zone.number_yellow == 0:
                            zone.dominant_color = "DRAW"
                        elif zone.number_blue > zone.number_yellow:
                            zone.dominant_color = "BLUE"
                        else:
                            zone.dominant_color = "YELLOW"
                payload = build_robot_payload(dico_zones, output, zone_record.frame_id, zone_record.timestamp_ns)
                bus_externe.publish(payload)