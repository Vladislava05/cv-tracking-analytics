from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from ultralytics import YOLO

from loguru import logger


@dataclass
class Detection:
    bbox: np.ndarray      
    score: float
    class_id: int
    class_name: str


class Detector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cuda",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        classes: Optional[Sequence[int]] = None,
        imgsz: int = 640,
        half: bool = True,
    ) -> None:
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = list(classes) if classes else None
        self.imgsz = imgsz
        self.half = half and self.device == "cuda"

        logger.info(f"Loading detector: {model_path} on {self.device}")
        self.model = YOLO(model_path)

        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model.predict(
            dummy, imgsz=imgsz, device=self.device, half=self.half, verbose=False
        )
        logger.info("Detector warmed up.")

    @torch.no_grad()
    def __call__(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        names = results.names

        for bbox, score, cls_id in zip(boxes, scores, cls_ids):
            detections.append(
                Detection(
                    bbox=bbox.astype(np.float32),
                    score=float(score),
                    class_id=int(cls_id),
                    class_name=names[int(cls_id)],
                )
            )
        return detections

    def export_onnx(self, output_path: str = "yolov8n.onnx") -> str:
        logger.info(f"Exporting to ONNX: {output_path}")
        self.model.export(format="onnx", imgsz=self.imgsz, half=self.half, dynamic=False)
        return output_path

    def export_tensorrt(self, output_path: str = "yolov8n.engine") -> str:
        if self.device != "cuda":
            raise RuntimeError("TensorRT export requires CUDA device")
        logger.info(f"Exporting to TensorRT: {output_path}")
        self.model.export(format="engine", imgsz=self.imgsz, half=self.half)
        return output_path