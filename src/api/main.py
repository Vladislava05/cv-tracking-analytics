from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from src.pipeline import TrackingPipeline


class TrackInfo(BaseModel):
    id: int
    bbox: List[float]
    class_id: int
    score: float
    speed_ms: float


class MetricsResponse(BaseModel):
    frame_idx: int
    fps: float
    active_tracks: int
    line_stats: List[dict]
    tracks: List[TrackInfo]
    running: bool

class AppState:
    pipeline: Optional[TrackingPipeline] = None
    latest_metrics: Dict[str, Any] = {}
    subscribers: List[WebSocket] = []
    worker_thread: Optional[threading.Thread] = None
    stop_flag: threading.Event = threading.Event()
    started: bool = False


state = AppState()


def _pipeline_worker(video_source: str, config: dict) -> None:
    logger.info(f"Pipeline worker started on source: {video_source}")
    try:
        pipeline = TrackingPipeline(config)
        state.pipeline = pipeline

        while not state.stop_flag.is_set():
            for result in pipeline.run_video(source=video_source):
                if state.stop_flag.is_set():
                    break

              
                state.latest_metrics = {
                    "frame_idx": result.frame_idx,
                    "fps": result.fps,
                    "active_tracks": len(result.tracks),
                    "line_stats": result.line_stats,
                    "tracks": result.tracks,
                    "running": True,
                }
              
                time.sleep(0.001)

            logger.info("Video finished, looping...")
    
            pipeline = TrackingPipeline(config)
            state.pipeline = pipeline

    except Exception as e:
        logger.exception(f"Pipeline worker crashed: {e}")
        state.latest_metrics = {**state.latest_metrics, "running": False}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up API...")

    cfg_path = Path("configs/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    import os
    video_source = os.getenv("VIDEO_SOURCE", "data/sample.mp4")

    if not Path(video_source).exists():
        logger.warning(f"Video source not found: {video_source}. Pipeline not started.")
    else:
        state.stop_flag.clear()
        state.worker_thread = threading.Thread(
            target=_pipeline_worker,
            args=(video_source, config),
            daemon=True,
        )
        state.worker_thread.start()
        state.started = True
        logger.info(f"Pipeline worker launched for {video_source}")

    yield

    logger.info("Shutting down API...")
    state.stop_flag.set()
    if state.worker_thread is not None:
        state.worker_thread.join(timeout=3)


app = FastAPI(title="CV Tracking API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_running": state.started}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Последние метрики пайплайна."""
    if not state.latest_metrics:
        return MetricsResponse(
            frame_idx=0,
            fps=0.0,
            active_tracks=0,
            line_stats=[],
            tracks=[],
            running=False,
        )
    return MetricsResponse(**state.latest_metrics)


@app.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket):
    """WebSocket: пушит метрики каждые 200мс."""
    await ws.accept()
    logger.info("WS subscriber connected.")
    try:
        while True:
            if state.latest_metrics:
                await ws.send_json(state.latest_metrics)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        logger.info("WS subscriber disconnected.")
    except Exception as e:
        logger.warning(f"WS error: {e}")