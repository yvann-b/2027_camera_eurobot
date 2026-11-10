import time

from ring.frame_ring_redresse import FrameRingRedresse
from config.loader import Subframe_Configuration, Homographie_config, YOLO_config, CalibrationConfig
from p_frame_redresse.extraction_subframe import extract_subframe_mm
from p_frame_redresse.YOLO_detector import YoloDetector
from bus_zmq.zmq_bus import Publisher
from p_frame_redresse.correction_paralaxe import parallax_list
from p_aruco.calibration import CameraPose

import numpy as np
import multiprocessing as mp
import time
import cv2
import os

def process_analyse_redresse(
        RingRedresse : FrameRingRedresse,
        start_event : mp.Event,
        stop_event : mp.Event,
        config_subframe : dict[str, Subframe_Configuration],
        homo_config : Homographie_config,
        yolo_config : YOLO_config,
        pub_a_to_k : Publisher,
        param_calib : CalibrationConfig,
        is_plotting_subframe : bool


                                ):


    yolo_detector = YoloDetector(yolo_config)


    while not(start_event.is_set()):
        time.sleep(0.01)


    # /!\ camera_pose est "créé" par initialisation, donc DOIT être appelé après le changement de camera_pose (comme initialisation set le start event la temporalité est respectée).
    chemin_complet = os.path.join(param_calib.output_calib_file, "pose_cam.npz")
    cam_pose = CameraPose.load(chemin_complet)  # appel direct

    while not(stop_event.is_set()):
        msg = RingRedresse.read_last_image()
        if msg is None :
            continue
        frame = msg.frame                       # Frame redressée
        dico_image = {}                         # Subframe_configuration : np.ndarray



        for subframe_name in config_subframe.keys():
            dico_image[config_subframe[subframe_name]] = extract_subframe_mm(
                frame = frame,
                corners_mm = config_subframe[subframe_name].corners,
                security_margin= homo_config.security_margin,
                ratio_mm_per_pixel= homo_config.ratio_mm_per_pixel,
            )
            # Création du dictionnaire Subframe_configuration : np.ndarray (pour pouvoir donner ça au YOLO detector,
            # Et qu'il puisse directement en sortir une


        liste_detection = yolo_detector.analyze_batch(dico_image)
        liste_detection = parallax_list(liste_detection,cam_pose.t_cam,25)
        pub_a_to_k.send(data=liste_detection, frame_id=msg.frame_id,timestamp_ns=time.time_ns())


        if is_plotting_subframe :
            for subframe_config in dico_image.keys():
                cv2.imshow(subframe_config.name, dico_image[subframe_config])
                cv2.waitKey(1)

    cv2.destroyAllWindows()