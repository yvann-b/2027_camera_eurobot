import multiprocessing

from bus_zmq.zmq_bus import Publisher, Subscriber
from config.loader import load_config,CalibrationData


# ============================  Import des config ZMQ   ============================
from bus_zmq.bus_config import ARUCO_TO_RAYCASTING , INIT_TO_APP_HOMO, YOLO_TO_KALMAN
# ================================================================================

# ============================  Import des Ring   ============================
from ring.estimation_ring import DetectionRingBuffer
from ring.frame_ring import FrameRing
from ring.frame_ring_redresse import FrameRingRedresse
from ring.zone_obstruction import ZoneObstructionRing
# ================================================================================


# ============================  Import des process   ============================
from processes.image_acquisition import process_image_acquisition
from processes.raw_image_plotting import process_frame_debug_display
from processes.initialisation import process_initialisation
from processes.aruco_tag_tracking import process_aruco_running
from processes.raycasting import process_raycasting
from processes.redressement_image import process_redressement_image
from processes.frame_redresse_analyse import process_analyse_redresse
from processes.timer import timer
from processes.kalman_filter import process_kalman_filter
from processes.debug_kf import process_plotting_kf
from processes.post_traitement import process_post_traitement
# ================================================================================

from multiprocessing import Process
import time
import cv2
import numpy as np


if __name__ == '__main__':

    print("1")
    cfg = load_config()  # Le chemin d'accès est défini dans le default dans loader


    def matrice_correction_distorsion():
        with np.load(cfg.calibration.npz_path) as data:
            camera_matrix = data["mtx"]
            dist_coeffs = data["dist"]

        img_w = cfg.camera.width  # Normalement défini à l'acquisition d'une frame, à changer si la caméra change
        img_h = cfg.camera.height  # Normalement défini à l'acquisition d'une frame, à changer si la caméra change

        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (img_w, img_h), 0, (img_w,
                                                                                                img_h))  # calcule une nouvelle matrice intrinsèque optimisée pour réduire la distorsion et définir une ROI utile

        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_mtx,
            (img_w, img_h),
            cv2.CV_16SC2
        )   # Avec ça, on corrige directement la distorsion sur la frame et on a plus à s'occuper de la distorsion de l'image dans le futur

        calib_data = CalibrationData(new_camera_mtx,np.zeros((5, 1)),roi)

        return calib_data, map1, map2

    calib_frame_data, map1, map2 = matrice_correction_distorsion()


    # ========================================
    # Objets partagés entre différents process :

    # ========================================
    # Circular Buffer :
    # ========================================
    FrameRingBuffer = FrameRing(
            cam = cfg.camera,
            header = cfg.ring_buffer_header,
            data_slot= cfg.ring_buffers["camera"],
            is_creating=True)
    ZoneObstructionBuffer = ZoneObstructionRing(
            n_zones= 18,
            header=cfg.ring_buffer_header,
            data_slot=cfg.ring_buffers["zone_state"],
            is_creating=True
    )
    FrameRedresseRingBuffer = FrameRingRedresse(
            table = cfg.homographie_config,
            cam= cfg.camera,
            header = cfg.ring_buffer_header,
            data_slot=cfg.ring_buffers["frame_redresse"],
            is_creating=True
    )

    EstimationRingBuffer = DetectionRingBuffer(
            data_slot=cfg.ring_buffers["kf_output"],
            is_creating = True
    )

    liste_buffer = [FrameRingBuffer, ZoneObstructionBuffer, FrameRedresseRingBuffer, EstimationRingBuffer]

    # ======================================== Multiprocessing Event ========================================
    stop_event_flag = multiprocessing.Event()
    start_running_configuration = multiprocessing.Event()
    stop_running_configuration = multiprocessing.Event()

    # -------------- Timer --------------
    timer_100hz = multiprocessing.Event()


    # ========================================
    # Bus ZMQ
    # ========================================
    pub_a_to_r = Publisher(ARUCO_TO_RAYCASTING)
    sub_a_to_r = Subscriber(ARUCO_TO_RAYCASTING)

    pub_init_to_app_homo = Publisher(INIT_TO_APP_HOMO)
    sub_init_to_app_homo = Subscriber(INIT_TO_APP_HOMO)

    pub_anal_to_k = Publisher(YOLO_TO_KALMAN)
    sub_anal_to_k = Subscriber(YOLO_TO_KALMAN)

    # ========================================
    # Processes
    # ========================================

    process_image = multiprocessing.Process(
        target=process_image_acquisition,
        kwargs=dict(
            camera_config=cfg.camera,
            frame_ring=FrameRingBuffer,
            stop_event=stop_event_flag,
            map1 = map1,
            map2 = map2,
        ),
    )
    process_display_raw = multiprocessing.Process(
        target=process_frame_debug_display,
        kwargs=dict(
            frame_ring=FrameRingBuffer,
            stop_event=stop_event_flag,
        ),
    )
    process_display_homo = multiprocessing.Process(
        target=process_frame_debug_display,
        kwargs=dict(
            frame_ring=FrameRedresseRingBuffer,
            stop_event=stop_event_flag,
        ),
    )

    process_initialisation_calib = multiprocessing.Process(
        target=process_initialisation,
        kwargs=dict(
            frame_ring = FrameRingBuffer,
            param_calib = cfg.calibration,
            calib_frame = calib_frame_data,
            camera_config = cfg.camera,
            start_event = start_running_configuration,
            zones_header = cfg.zones_header,
            pub_i_to_app= pub_init_to_app_homo,
            homographie_config = cfg.homographie_config,
        ),
    )

    process_aruco = multiprocessing.Process(
        target = process_aruco_running,
        kwargs=dict(
            pub=pub_a_to_r,
            camera_matrice=calib_frame_data.camera_matrix,
            runtime_config=cfg.runtime,
            frame_ring = FrameRingBuffer,
            start_event = start_running_configuration,
            stop_event =stop_running_configuration,
            param_calib = cfg.calibration,
        ),

    )

    process_raycast = multiprocessing.Process(
        target = process_raycasting,
        kwargs=dict(
            sub = sub_a_to_r,
            start_event = start_running_configuration,
            stop_event = stop_running_configuration,
            param_calib = cfg.calibration,
            zones_header = cfg.zones_header,
            runtime = cfg.runtime,
            zone_ring = ZoneObstructionBuffer
        )
    )

    process_redressement = multiprocessing.Process(
        target=process_redressement_image,
        kwargs=dict(
            sub = sub_init_to_app_homo,
            frame_ring = FrameRingBuffer,
            frame_redresse_ring = FrameRedresseRingBuffer,
            start_event = start_running_configuration,
        )
    )

    process_analyse_red = multiprocessing.Process(
        target = process_analyse_redresse,
        kwargs=dict(
            RingRedresse = FrameRedresseRingBuffer,
            start_event = start_running_configuration,
            stop_event = stop_running_configuration,
            homo_config = cfg.homographie_config,
            is_plotting_subframe = True,                        # Pour du débug rapide
            config_subframe = cfg.subframes_dico,
            pub_a_to_k = pub_anal_to_k,
            yolo_config = cfg.yolo_config,
            param_calib = cfg.calibration
        )
    )

    process_timer100hz = multiprocessing.Process(
        target = timer,
        kwargs=dict(
            frequency = 100,
            event = timer_100hz
        )
    )

    process_kf = multiprocessing.Process(
        target = process_kalman_filter,
        kwargs=dict(
            sub = sub_anal_to_k,
            kalman_config = cfg.kalman_config,
            timer_interupt = timer_100hz,
            start_event = start_running_configuration,
            stop_event = stop_running_configuration,
            EstimationRing = EstimationRingBuffer,
        )
    )

    process_debug_kf = multiprocessing.Process(
        target = process_plotting_kf,
        kwargs=dict(
            ring = EstimationRingBuffer,
            start_event = start_running_configuration,
            stop_event = stop_running_configuration,
        )

    )

    process_pt = multiprocessing.Process(
        target=process_post_traitement,
        kwargs=dict(
            estimation_ring = EstimationRingBuffer,
            obstruction_ring =ZoneObstructionBuffer,
            start_event = start_running_configuration,
            stop_event = stop_running_configuration,
            zones_header = cfg.zones_header,
            param_calib = cfg.calibration,
            jenga=cfg.jenga,
            external_bus = cfg.external_bus,

        )

    )



    liste_process = [process_image, process_display_raw,process_display_homo,process_initialisation_calib,process_aruco,process_raycast,process_redressement,process_analyse_red,process_timer100hz,process_kf,process_debug_kf,process_pt]

    for p in liste_process:
        p.start()

    time.sleep(5)           # Attention à bien laisser le temps à tout les procédés de démarrer
    for p in liste_process:
        print(f"{p.name} alive={p.is_alive()} exitcode={p.exitcode}")

    while not(stop_event_flag.is_set()):
        print("Waiting for q to quit")
        if str(input()) == "q":
            stop_event_flag.set()

    for p in liste_process:
        p.join()

    for buffer in liste_buffer:
        buffer.close()
        buffer.unlink()