import struct
import numpy as np
from ring.circular_buffer import CircularBuffer
from config.loader import RingBufferHeaderSpec, RingBufferStreamSpec, ZoneObstructionMessage


class ZoneObstructionRing(CircularBuffer):
    """CB du chemin 1 (aruco -> raycasting) : obstruction dynamique des zones
    par les robots. À NE PAS CONFONDRE avec ZoneHeader.is_visible, qui est le
    FoV caméra figé, calculé une seule fois à l'initialisation."""

    def __init__(self, n_zones: int, header: RingBufferHeaderSpec, data_slot: RingBufferStreamSpec,
                 is_creating: bool = False, write_count=None):
        self.n_zones = n_zones
        self._header_format = header.struct_format
        self._header_size = header.header_bytes

        super().__init__(
            name=data_slot.name,
            capacity=data_slot.n_slots,
            item_size=data_slot.slot_bytes,   # inclut déjà le header, une seule fois
            is_creating=is_creating,
            write_count=write_count,
        )

    def write_obstruction(self, is_blocked: np.ndarray, frame_id: int, timestamp_ns: int):
        assert is_blocked.shape == (self.n_zones,), \
            f"is_blocked: attendu ({self.n_zones},), reçu {is_blocked.shape}"

        header_bytes = np.frombuffer(
            struct.pack(self._header_format, frame_id, timestamp_ns), dtype=np.uint8
        )
        body = is_blocked.astype(np.int8).reshape(-1).view(np.uint8)
        raw = np.concatenate([header_bytes, body])
        self.write(raw)

    def read_last_obstruction(self):
        result = self.read_latest()
        if result is None:
            return None
        raw, _ = result
        frame_id, timestamp_ns = struct.unpack(self._header_format, raw[:self._header_size].tobytes())
        is_blocked = raw[self._header_size:].view(np.int8).astype(bool)
        return ZoneObstructionMessage(frame_id, timestamp_ns, is_blocked)


    def read_last_n_obstruction(self, n):
        raw, overrun = super().read_last_n(n)
        messages = []
        for row in raw:
            frame_id, timestamp_ns = struct.unpack(
                self._header_format, row[:self._header_size].tobytes()
            )
            is_blocked = row[self._header_size:].view(np.int8).astype(bool)
            messages.append(ZoneObstructionMessage(frame_id, timestamp_ns, is_blocked))
        return messages, overrun