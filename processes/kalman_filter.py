from p_kalman.MultipleObjectKalmanFilter import MultipleObjectKalmanFilter
from bus_zmq.zmq_bus import Subscriber
from config.loader import Kalman_config
from ring.estimation_ring import DetectionRingBuffer
import multiprocessing as mp

import cv2
import numpy as np
import math
import time


def process_kalman_filter(
        sub : Subscriber,
        kalman_config : Kalman_config,
        timer_interupt : mp.Event,
        start_event : mp.Event,
        stop_event : mp.Event,
        EstimationRing : DetectionRingBuffer

):


    MOKF = MultipleObjectKalmanFilter(kalman_config)

    while not(start_event.is_set()):
        time.sleep(0.001)


    while not(stop_event.is_set()):
        if timer_interupt.is_set():
            MOKF.predict_kf()
            timer_interupt.clear()

        msg = sub.receive(0)

        if msg is not None and msg.data is not None:
            MOKF.update_kf(msg.data)
            liste_estimate_state = MOKF.get_augmented_estimate_state()
            EstimationRing.write_data(detections=liste_estimate_state,frame_id=msg.frame_id,timestamp_ns=msg.timestamp_ns)




        else:
            # on libère le CPU pour une milliseconde.
            time.sleep(0.001)