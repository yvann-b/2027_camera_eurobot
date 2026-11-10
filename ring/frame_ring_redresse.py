import struct
import numpy as np

from ring.circular_buffer import CircularBuffer
from config.loader import RingBufferHeaderSpec, RingBufferStreamSpec, FrameMessage, Homographie_config, CameraConfig


class FrameRingRedresse(CircularBuffer):
    def __init__(self, table : Homographie_config, cam : CameraConfig,header: RingBufferHeaderSpec, data_slot: RingBufferStreamSpec,
                 is_creating: bool = False, write_count=None):

        size_width = table.width + 2 * table.security_margin            # mm
        size_height = table.height + 2 * table.security_margin          # mm

        size_width_pixel = int(size_width / table.ratio_mm_per_pixel)              # Devrait donner 1600 px
        size_height_pixel = int(size_height / table.ratio_mm_per_pixel)             # Devrait donner 1100 px

        self.image_shape = (size_height_pixel,size_width_pixel, cam.channels)            # Ouais numpy inverse largeur et hauteur ici :'(
        self._header_format = header.struct_format
        self._header_size = header.header_bytes

        super().__init__(
            name=data_slot.name,
            capacity=data_slot.n_slots,
            item_size=data_slot.slot_bytes,   # inclut déjà le header, une seule fois
            is_creating=is_creating,
            write_count=write_count,
        )

    def write_image(self, frame: np.ndarray, frame_id: int, timestamp_ns: int):
        header_bytes = np.frombuffer(
            struct.pack(self._header_format, frame_id, timestamp_ns), dtype=np.uint8
        )
        body = frame.reshape(-1).view(np.uint8)
        raw = np.concatenate([header_bytes, body])
        self.write(raw)

    def read_last_image(self):
        result = self.read_latest()
        if result is None:
            return None
        raw, _ = result
        frame_id, timestamp_ns = struct.unpack(self._header_format, raw[:self._header_size].tobytes())
        frame = raw[self._header_size:].reshape(self.image_shape)
        return FrameMessage(frame_id, timestamp_ns, frame)