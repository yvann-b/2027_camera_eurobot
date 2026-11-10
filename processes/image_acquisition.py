from config import CameraConfig
from video.simulation import VideoFileSimulation
from video.camera import DahengCamera
from ring.frame_ring import FrameRing
import multiprocessing
import traceback
import cv2

def process_image_acquisition(camera_config: CameraConfig, frame_ring : FrameRing,stop_event: multiprocessing.Event, map1, map2):
    source = None
    try :
        if camera_config.is_camera_used:
            source = DahengCamera(camera_config)
            source.start()
        else:
            source = VideoFileSimulation(camera_config.video_name_simulation)

        while (not stop_event.is_set()):
            result = source.get_frame()
            if result is None:
                continue

            frame, frame_id, timestamp = result
            frame_corrige = cv2.remap(frame, map1, map2, cv2.INTER_NEAREST)
            frame_ring.write_image(frame_corrige, frame_id, timestamp)

    except StopIteration:
        print("Fin de la simulation vidéo (loop=False).")

    except Exception:
        print("CRASH dans process_image_acquisition :")
        traceback.print_exc()
    finally:
        if source is not None:
            source.stop()