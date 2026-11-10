from ring.estimation_ring import DetectionRingBuffer
from ring.zone_obstruction import ZoneObstructionRing
import numpy as np


def find_closest(ts_array, target):
    idx = np.searchsorted(ts_array, target)
    if idx == 0:
        return 0
    if idx == len(ts_array):
        return len(ts_array) - 1
    before, after = ts_array[idx - 1], ts_array[idx]
    return idx - 1 if (target - before) <= (after - target) else idx


def merge_branches(detection_cb: DetectionRingBuffer, zone_cb: ZoneObstructionRing,
                    staleness_threshold_ns: int):

    """
    :param detection_cb: ring des estimations kalman
    :param zone_cb: ring des zones état
    :param staleness_threshold_ns: seuil de tolérance
    :return:
    """
    det_items, _ = detection_cb.read_last_n(3)
    zone_msgs, _ = zone_cb.read_last_n_obstruction(3)


    if det_items.shape[0] == 0 or not zone_msgs or len(zone_msgs) == 0 or len(det_items) == 0:
        return None,None

    det_ts = det_items['timestamp_ns']
    zone_ts = np.array([m.timestamp_ns for m in zone_msgs])

    if det_ts[-1] <= zone_ts[-1]:
        # branche détection (KF) est la plus lente : elle fixe la référence
        target_ts = int(det_ts[-1])
        det_record = detection_cb.decode_one(det_items[-1])

        idx = find_closest(zone_ts, target_ts)
        zone_record, matched_ts = zone_msgs[idx], int(zone_ts[idx])
    else:
        # branche obstruction est la plus lente
        target_ts = int(zone_ts[-1])
        zone_record = zone_msgs[-1]
        idx = find_closest(det_ts, target_ts)
        det_record, matched_ts = detection_cb.decode_one(det_items[idx]), int(det_ts[idx])

    is_stale = abs(matched_ts - target_ts) > staleness_threshold_ns
    return zone_record, det_record

