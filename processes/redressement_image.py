
from bus_zmq.zmq_bus import Subscriber
from ring.frame_ring import FrameRing
from ring.frame_ring_redresse import FrameRingRedresse
import multiprocessing
import cv2
import numpy as np

def process_redressement_image(     sub : Subscriber, frame_ring : FrameRing,
                                    frame_redresse_ring : FrameRingRedresse, start_event : multiprocessing.Event,
                                    ):
    H = None
    while H is None or not(start_event.is_set()) :
        msg = sub.receive(timeout=2.0)
        if msg is None:
            print("Message not received")
            continue
        print(f"[Redressement] frame_id={msg.frame_id} "
              f"timestamp_ns={msg.timestamp_ns} data={msg.data}")
        H = np.array(msg.data["Matrice"])          # Parce qu'on a du le transformer en liste python pour le sérialiser
        W_map = msg.data["Width"]
        H_map = msg.data["Height"]

    while start_event.is_set():
        F_mes = frame_ring.read_last_image()
        frame = F_mes.frame
        frame_map = cv2.warpPerspective(frame, H, (W_map, H_map))
        frame_map = cv2.flip(frame_map, 0)
        frame_redresse_ring.write_image(
            frame= frame_map,
            frame_id= F_mes.frame_id,
            timestamp_ns= F_mes.frame_timestamp_ns,
        )
