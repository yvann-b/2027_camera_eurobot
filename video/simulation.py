import time
import cv2
import numpy as np


class VideoFileSimulation:
    def __init__(self, video_path: str, loop: bool = True):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")

        self.loop = loop
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 20.0
        self.delay_ns = int(1e9 / self.fps)

        self._frame_id = 0
        self._last_read_ns = None  # None => pas encore de frame précédente

    def start(self) -> None:
        pass  # rien à démarrer : get_frame() est autoportant, pas de thread

    def get_frame(self) -> tuple[np.ndarray, int, int]:
        ret, frame = self.cap.read()

        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError("Vidéo illisible même après rembobinage")
            else:
                raise StopIteration("Fin de la vidéo (loop=False)")

        now_ns = time.time_ns()
        if self._last_read_ns is not None:
            elapsed_ns = now_ns - self._last_read_ns
            wait_ns = self.delay_ns - elapsed_ns
            if wait_ns > 0:
                time.sleep(wait_ns / 1e9)

        timestamp_ns = time.time_ns()
        self._last_read_ns = timestamp_ns
        self._frame_id += 1

        return frame, self._frame_id, timestamp_ns

    def stop(self) -> None:
        self.cap.release()