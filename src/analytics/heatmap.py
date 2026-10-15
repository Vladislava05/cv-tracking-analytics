from __future__ import annotations

import cv2
import numpy as np


class Heatmap:
    def __init__(self, frame_size: tuple[int, int], resolution=(64, 36), decay: float = 0.99):
        self.frame_w, self.frame_h = frame_size
        self.res_w, self.res_h = resolution
        self.decay = decay
        self.accum = np.zeros((self.res_h, self.res_w), dtype=np.float32)

    def update(self, bboxes: list[np.ndarray]) -> None:
        self.accum *= self.decay
        for bbox in bboxes:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            gx = int(np.clip(cx / self.frame_w * self.res_w, 0, self.res_w - 1))
            gy = int(np.clip(cy / self.frame_h * self.res_h, 0, self.res_h - 1))
            self.accum[gy, gx] += 1.0

    def render(self, alpha: float = 0.5) -> np.ndarray:
        norm = self.accum / (self.accum.max() + 1e-6)
        norm = (norm * 255).astype(np.uint8)
        big = cv2.resize(norm, (self.frame_w, self.frame_h), interpolation=cv2.INTER_CUBIC)
        colored = cv2.applyColorMap(big, cv2.COLORMAP_JET)
        return colored

    def blend(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        overlay = self.render()
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)