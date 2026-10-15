from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional

import cv2
import numpy as np
from loguru import logger

from src.analytics.heatmap import Heatmap
from src.analytics.line_counter import LineCounter
from src.analytics.speed_estimator import SpeedEstimator
from src.detection.model import Detector
from src.tracking.byte_tracker import ByteTracker



CLASS_COLORS = {
    0: (0, 255, 0),     # person
    1: (255, 128, 0),   # bicycle
    2: (0, 128, 255),   # car
    3: (255, 0, 255),   # motorcycle
    5: (255, 255, 0),   # bus
    7: (0, 255, 255),   # truck
}


@dataclass
class FrameResult:
    frame: np.ndarray
    frame_idx: int
    timestamp: float
    tracks: List[dict] = field(default_factory=list)
    line_stats: List[dict] = field(default_factory=list)
    fps: float = 0.0


class TrackingPipeline:

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.detector = Detector(**config["detector"])
        self.tracker = ByteTracker(**config["tracker"])

        self.line_counters: List[LineCounter] = []
        for line_cfg in config.get("analytics", {}).get("line", []):
            self.line_counters.append(
                LineCounter(
                    start=tuple(line_cfg["start"]),
                    end=tuple(line_cfg["end"]),
                    name=line_cfg.get("name", "line"),
                )
            )

        speed_cfg = config.get("analytics", {}).get("speed", None)
        self.speed_estimator: Optional[SpeedEstimator] = None
        if speed_cfg:
            self.speed_estimator = SpeedEstimator(
                src_points=speed_cfg["src_points"],
                dst_points=speed_cfg["dst_points"],
            )

        self.heatmap: Optional[Heatmap] = None
        self._heatmap_cfg = config.get("analytics", {}).get("heatmap", None)

        self._frame_idx = 0
        self._fps_ema = 0.0

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()

        detections = self.detector(frame)

        alive_tracks = self.tracker.update(detections)

        if self._heatmap_cfg is not None and self.heatmap is None:
            h, w = frame.shape[:2]
            self.heatmap = Heatmap(
                frame_size=(w, h),
                resolution=tuple(self._heatmap_cfg.get("resolution", [64, 36])),
                decay=self._heatmap_cfg.get("decay", 0.99),
            )

        track_infos = []
        bboxes_for_heat = []
        active_ids = set()

        for trk in alive_tracks:
            bbox = trk.get_state()
            active_ids.add(trk.id)
            bboxes_for_heat.append(bbox)

            speed = 0.0
            if self.speed_estimator is not None:
                speed = self.speed_estimator.update(trk.id, bbox)

            for lc in self.line_counters:
                lc.update(trk.id, bbox)

            track_infos.append(
                {
                    "id": trk.id,
                    "bbox": bbox.tolist(),
                    "class_id": trk.class_id,
                    "score": trk.score,
                    "speed_ms": speed,
                    "age": trk.age,
                }
            )

        if self.heatmap is not None and bboxes_for_heat:
            self.heatmap.update(bboxes_for_heat)

        for lc in self.line_counters:
            lc.cleanup(active_ids)
        if self.speed_estimator is not None:
            self.speed_estimator.cleanup(active_ids)

        dt = time.perf_counter() - t0
        fps = 1.0 / max(dt, 1e-6)
        self._fps_ema = 0.9 * self._fps_ema + 0.1 * fps if self._fps_ema else fps

        self._frame_idx += 1
        return FrameResult(
            frame=frame,
            frame_idx=self._frame_idx,
            timestamp=time.time(),
            tracks=track_infos,
            line_stats=[lc.stats() for lc in self.line_counters],
            fps=self._fps_ema,
        )

    def render(self, result: FrameResult, show_heatmap: bool = False) -> np.ndarray:
        frame = result.frame.copy()

        if show_heatmap and self.heatmap is not None:
            frame = self.heatmap.blend(frame, alpha=0.4)

        # Bboxes
        for t in result.tracks:
            x1, y1, x2, y2 = map(int, t["bbox"])
            color = CLASS_COLORS.get(t["class_id"], (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID {t['id']}"
            if t["speed_ms"] > 0.1:
                label += f" | {t['speed_ms']:.1f} m/s"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
            )

        for lc in self.line_counters:
            a = tuple(map(int, lc.a))
            b = tuple(map(int, lc.b))
            cv2.line(frame, a, b, (0, 255, 255), 3)
            cv2.putText(
                frame,
                f"{lc.name}: in={lc.in_count} out={lc.out_count}",
                (a[0], a[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )

        # FPS
        cv2.putText(
            frame, f"FPS: {result.fps:.1f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )
        return frame

    def run_video(
        self,
        source: str | int,
        output_path: Optional[str] = None,
        show: bool = False,
        show_heatmap: bool = False,
        max_frames: Optional[int] = None,
    ) -> Generator[FrameResult, None, None]:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        writer = None
        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        logger.info(f"Starting pipeline on {source}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.process_frame(frame)
                rendered = self.render(result, show_heatmap=show_heatmap)

                if writer is not None:
                    writer.write(rendered)
                if show:
                    cv2.imshow("Tracking", rendered)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                yield result

                if max_frames and result.frame_idx >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()
            logger.info("Pipeline finished.")