from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detection.model import Detection
from src.tracking.kalman import KalmanBoxTracker


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)), dtype=np.float32)

    a = bboxes_a[:, None, :]  # [N,1,4]
    b = bboxes_b[None, :, :]  # [1,M,4]

    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter + 1e-6

    return (inter / union).astype(np.float32)


def associate(
    detections: np.ndarray, trackers: np.ndarray, iou_threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:

    if len(trackers) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(trackers)))

    iou_matrix = iou_batch(detections, trackers)
    cost = 1.0 - iou_matrix

    row_ind, col_ind = linear_sum_assignment(cost)

    matches, unmatched_dets, unmatched_tracks = [], [], []
    matched_det = set()
    matched_trk = set()

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] < iou_threshold:
            continue
        matches.append((r, c))
        matched_det.add(r)
        matched_trk.add(c)

    for d in range(len(detections)):
        if d not in matched_det:
            unmatched_dets.append(d)
    for t in range(len(trackers)):
        if t not in matched_trk:
            unmatched_tracks.append(t)

    return matches, unmatched_dets, unmatched_tracks


class ByteTracker:

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh

        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: List[Detection]) -> List[KalmanBoxTracker]:
        self.frame_count += 1

        high_dets = [d for d in detections if d.score >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]

        for trk in self.trackers:
            trk.predict()

        trk_bboxes = np.array([t.get_state() for t in self.trackers], dtype=np.float32)
        high_bboxes = np.array([d.bbox for d in high_dets], dtype=np.float32) \
            if high_dets else np.zeros((0, 4), dtype=np.float32)

        matches, unmatched_high, unmatched_trk = associate(
            high_bboxes, trk_bboxes, self.iou_threshold
        )

        for d_idx, t_idx in matches:
            det = high_dets[d_idx]
            self.trackers[t_idx].update(det.bbox, det.score, det.class_id)

        if len(unmatched_trk) > 0 and len(low_dets) > 0:
            remaining_trk_bboxes = np.array(
                [self.trackers[t].get_state() for t in unmatched_trk], dtype=np.float32
            )
            low_bboxes = np.array([d.bbox for d in low_dets], dtype=np.float32)
            matches2, _, unmatched_trk2 = associate(
                low_bboxes, remaining_trk_bboxes, 0.5  
            )
            for d_idx, t_idx in matches2:
                det = low_dets[d_idx]
                real_t_idx = unmatched_trk[t_idx]
                self.trackers[real_t_idx].update(det.bbox, det.score, det.class_id)

            unmatched_trk = [unmatched_trk[i] for i in unmatched_trk2]

        for d_idx in unmatched_high:
            det = high_dets[d_idx]
            trk = KalmanBoxTracker(det.bbox)
            trk.class_id = det.class_id
            trk.score = det.score
            self.trackers.append(trk)


        alive = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                alive.append(trk)
        self.trackers = alive

        return [t for t in self.trackers if t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits]

    def reset(self) -> None:
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker._count = 0