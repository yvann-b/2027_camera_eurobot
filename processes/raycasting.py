from config import RuntimeConfig
from config.loader import CalibrationConfig, ZoneHeader, Zone
from p_aruco.calibration import CameraPose
from p_raycasting.core_raycasting import intersection_droite_cylindre
from bus_zmq.zmq_bus import Subscriber, ReceivedMessage
from zones.save_build import build_zones
from p_aruco.running_file import RobotPose
import time
import os
import multiprocessing
import numpy as np
from ring.zone_obstruction import ZoneObstructionRing

def process_raycasting(sub :  Subscriber, start_event : multiprocessing.Event, stop_event : multiprocessing.Event, param_calib : CalibrationConfig, zones_header : dict[int, ZoneHeader], runtime : RuntimeConfig, zone_ring : ZoneObstructionRing) :

    while not (start_event.is_set()):
        time.sleep(0.1)


    os.makedirs(param_calib.output_calib_file, exist_ok=True)               # Pour créer le dossier s'il n'existe pas déjà
    chemin_complet1 = os.path.join(param_calib.output_calib_file, "zones_visiblity.npz")
    chemin_complet = os.path.join(param_calib.output_calib_file, "pose_cam.npz")
    cam_pose = CameraPose.load(chemin_complet)  # appel direct

    liste_zones_header = []
    for key in zones_header.keys():
        liste_zones_header.append(zones_header[key])

    dico_zones = build_zones(headers= liste_zones_header,visibility_path= chemin_complet1 )     # Dict[int , Zone]

    while not (stop_event.is_set()):
        msg = sub.receive(timeout=2.0)

        if msg is None:
            print("Message not received")
        else :
            number_of_zone_value = np.zeros(len(dico_zones), dtype=np.int8)
            if not msg.data == {} :
                for robot_id in msg.data.keys():
                    robot_pose = msg.data[robot_id]
                    center = [robot_pose["x_mm"], robot_pose["y_mm"]]
                    height = runtime.yellow_rob_height

                    if robot_pose["team"] == "BLUE":
                        height = runtime.blue_rob_height

                    for i in dico_zones.keys():
                        zone = dico_zones[i]

                        result = intersection_droite_cylindre(center=center,
                                                                       point_table=zone.header.center,
                                                                       point_camera=cam_pose.t_cam,
                                                                       height=height)
                        if result :
                            number_of_zone_value[i] = 1


            zone_ring.write_obstruction(number_of_zone_value,frame_id=msg.frame_id,timestamp_ns=msg.timestamp_ns)