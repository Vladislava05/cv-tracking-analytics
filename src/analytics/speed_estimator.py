from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Tuple

import cv2
import numpy as np


class SpeedEstimator:
    def __init__(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        history_len: int = 15,
    ) -> None:
        self.H, _ = cv2.findHomography(
            np.array(src_points, dtype=np.float32),
            np.array(dst_points, dtype=np.float32),
        )
        if self.H is None:
            raise ValueError("Homography computation failed")

        self.history_len = history_len
        self._history: Dict[int, Deque[Tuple[float, np.ndarray]]] = {}

    def _to_world(self, point_px: np.ndarray) -> np.ndarray:
        pts = np.array([[point_px]], dtype=np.float32)
        world = cv2.perspectiveTransform(pts, self.H)[0, 0]
        return world

    def update(self, track_id: int, bbox: np.ndarray) -> float:
        cx = (bbox[0] + bbox[2]) / 2.0
        by = bbox[3]
        world_pt = self._to_world(np.array([cx, by]))
        now = time.time()

        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.history_len)

        hist = self._history[track_id]
        hist.append((now, world_pt))

        if len(hist) < 2:
            return 0.0

        t0, p0 = hist[0]
        t1, p1 = hist[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return 0.0

        dist = float(np.linalg.norm(p1 - p0))
        if len(hist) >= 5:
            ts = np.array([h[0] for h in hist])
            pts = np.array([h[1] for h in hist])
            ts = ts - ts[0]
            vx = np.polyfit(ts, pts[:, 0], 1)[0]
            vy = np.polyfit(ts, pts[:, 1], 1)[0]
            speed = float(np.hypot(vx, vy))
        else:
            speed = dist / dt

        return speed

    def cleanup(self, active_ids: set) -> None:
        dead = [tid for tid in self._history if tid not in active_ids]
        for tid in dead:
            del self._history[tid]