from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np


def _side_of_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))


def _segments_intersect(p1, p2, p3, p4) -> bool:
    d1 = _side_of_line(p3, p1, p2)
    d2 = _side_of_line(p4, p1, p2)
    d3 = _side_of_line(p1, p3, p4)
    d4 = _side_of_line(p2, p3, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


class LineCounter:

    def __init__(self, start: Tuple[int, int], end: Tuple[int, int], name: str = "line"):
        self.a = np.array(start, dtype=float)
        self.b = np.array(end, dtype=float)
        self.name = name

        self.in_count = 0
        self.out_count = 0
        self._last_pos: Dict[int, np.ndarray] = {}
        self._crossed: Dict[int, bool] = defaultdict(bool)

    def update(self, track_id: int, bbox: np.ndarray) -> bool:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        curr = np.array([cx, cy])

        if track_id not in self._last_pos:
            self._last_pos[track_id] = curr
            return False

        prev = self._last_pos[track_id]
        crossed = _segments_intersect(prev, curr, self.a, self.b)
        self._last_pos[track_id] = curr

        if not crossed:
            return False
        s_prev = _side_of_line(prev, self.a, self.b)
        s_curr = _side_of_line(curr, self.a, self.b)
        if s_prev < 0 <= s_curr:
            self.in_count += 1
        elif s_prev > 0 >= s_curr:
            self.out_count += 1
        return True

    def cleanup(self, active_ids: set) -> None:
        dead = [tid for tid in self._last_pos if tid not in active_ids]
        for tid in dead:
            del self._last_pos[tid]

    def stats(self) -> dict:
        return {
            "name": self.name,
            "in": self.in_count,
            "out": self.out_count,
            "total": self.in_count + self.out_count,
        }