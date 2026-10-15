from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    _count = 0  

    def __init__(self, bbox: np.ndarray) -> None:
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        self.history: list[np.ndarray] = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_id = -1
        self.score = 0.0

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """[x1,y1,x2,y2] -> [cx, cy, s, r]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / max(h, 1e-6)
        return np.array([cx, cy, s, r]).reshape((4, 1))

    @staticmethod
    def _z_to_bbox(x: np.ndarray) -> np.ndarray:
        """[cx, cy, s, r] -> [x1, y1, x2, y2]."""
        w = np.sqrt(max(x[2] * x[3], 0.0))
        h = x[2] / max(w, 1e-6)
        cx, cy = x[0], x[1]
        return np.array(
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
        ).reshape((4,))

    def predict(self) -> np.ndarray:
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        bbox = self._z_to_bbox(self.kf.x)
        self.history.append(bbox)
        return bbox

    def update(self, bbox: np.ndarray, score: float = 1.0, class_id: int = -1) -> None:
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.score = score
        self.class_id = class_id
        self.kf.update(self._bbox_to_z(bbox))

    def get_state(self) -> np.ndarray:
        return self._z_to_bbox(self.kf.x)