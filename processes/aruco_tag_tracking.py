import multiprocessing
from p_aruco.calibration import CameraPose
from p_aruco.running_file import RobotTagTracking
from p_aruco.tag_extractor import TagExtractor
from config import RuntimeConfig
import numpy as np
from config.loader import CalibrationConfig
from ring.frame_ring import FrameRing
from bus_zmq.zmq_bus import Publisher
import time
import os

def process_aruco_running(pub : Publisher , camera_matrice : np.ndarray,runtime_config: RuntimeConfig, frame_ring : FrameRing, start_event : multiprocessing.Event, stop_event : multiprocessing.Event, param_calib : CalibrationConfig) :
    dico_height = {"BLUE" : runtime_config.blue_rob_height, "YELLOW" : runtime_config.yellow_rob_height}


    while not(start_event.is_set()):
        time.sleep(0.1)

    # /!\ camera_pose est "créé" par initialisation, donc DOIT être appelé après le changement de camera_pose (comme initialisation set le start event la temporalité est respectée).
    chemin_complet = os.path.join(param_calib.output_calib_file, "pose_cam.npz")
    cam_pose = CameraPose.load(chemin_complet)  # appel direct

    robot_tracking = RobotTagTracking(  tag_extractor=TagExtractor(),
                                        camera_pose=cam_pose,
                                        cam_mtx=camera_matrice,
                                        heights_mm=dico_height)


    while not (stop_event.is_set()) :
        F_mes = frame_ring.read_last_image()
        frame = F_mes.frame
        liste_robot_pos = robot_tracking.track(frame, is_printing=True)

        # Pour le test / débug only
#        for id in liste_robot_pos:
#            print(f"Process aruco tracking | id : {id} | pose : {liste_robot_pos[id]}")

        pub.send(
            frame_id=F_mes.frame_id,
            timestamp_ns=F_mes.frame_timestamp_ns,
            data=liste_robot_pos,
        )

