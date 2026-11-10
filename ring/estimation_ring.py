import numpy as np
from ring.circular_buffer import CircularBuffer
from config.loader import RingBufferStreamSpec

MAX_DETECTIONS = 60

detection_dtype = np.dtype([
    ('x_mm', np.float64), ('y_mm', np.float64),
    ('theta_r', np.float64),
    ('color', 'U10'),
])

slot_dtype = np.dtype([
    ('frame_id', np.int64),
    ('timestamp_ns', np.int64),
    ('count', np.int32),
    ('detections', detection_dtype, (MAX_DETECTIONS,)),
])


class DetectionRingBuffer(CircularBuffer):
    def __init__(self, data_slot: RingBufferStreamSpec, is_creating, write_count=None):
        # item_size dérivé du dtype structuré, pas besoin que l'appelant le connaisse
        super().__init__(
            name=data_slot.name,
            capacity = data_slot.n_slots,
            item_size = slot_dtype.itemsize,
            is_creating=is_creating,
            write_count=write_count)

    def write_data(self, frame_id, timestamp_ns, detections):
        # detections liste de dico :         dico = {"x_mm" : self.estimated_state[0], "y_mm" : self.estimated_state[1], "theta_r" : self.estimated_state[2], "color" : self.couleur}
        n = len(detections)
        if n > MAX_DETECTIONS:
            detections = detections[:MAX_DETECTIONS]
            n = MAX_DETECTIONS
            # dépassement jamais attendu en match (40 max), pas grave si ça arrive en debug

        record = np.zeros(1, dtype=slot_dtype)
        record['frame_id'] = frame_id
        record['timestamp_ns'] = timestamp_ns
        record['count'] = n
        for i, det in enumerate(detections):
            record['detections'][0, i] = (
                det["x_mm"], det["y_mm"], det["theta_r"], det["color"]
            )

        super().write(record)

    @staticmethod
    def _decode_record(record):
        n = int(record['count'])
        return {
            'frame_id': int(record['frame_id']),
            'timestamp_ns': int(record['timestamp_ns']),
            'detections': record['detections'][:n].copy(),
        }

    def read_latest(self):
        result = super().read_latest()
        if result is None:
            return None
        raw, current = result
        record = raw.view(slot_dtype)[0]
        n = int(record['count'])
        return {
            'frame_id': int(record['frame_id']),
            'timestamp_ns': int(record['timestamp_ns']),
            'detections': record['detections'][:n].copy(),  # copy pour détacher de la vue transitoire
        }

    def read_next_fifo(self, cursor_id):
        result = super().read_next_fifo(cursor_id)
        if result is None:
            return None
        raw, last_seen, gap = result
        record = raw.view(slot_dtype)[0]
        n = int(record['count'])
        return {
            'frame_id': int(record['frame_id']),
            'timestamp_ns': int(record['timestamp_ns']),
            'detections': record['detections'][:n].copy(),
        }, last_seen, gap

    def read_last_n(self, n):
        """Renvoie le tableau structuré NON décodé des n derniers slots.
        Utiliser items['timestamp_ns'] pour chercher le plus proche voisin,
        puis decode_one(items[idx]) seulement sur le slot retenu."""
        raw, overrun = super().read_last_n(n)
        if raw.shape[0] == 0:
            return np.empty(0, dtype=slot_dtype), overrun
        items = raw.reshape(-1).view(slot_dtype)
        return items, overrun

    def decode_one(self, item):
        return self._decode_record(item)