from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Dict, List

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


class AppState:
    pipeline: TrackingPipeline | None = None
    latest_metrics: dict = {}
    subscribers: List[WebSocket] = []


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    yield
    logger.info("Shutting down...")


app = FastAPI(title="CV Tracking API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    return state.latest_metrics or MetricsResponse(
        frame_idx=0, fps=0, active_tracks=0, line_stats=[], tracks=[]
    )


@app.post("/push_metrics")
async def push_metrics(metrics: MetricsResponse):
    state.latest_metrics = metrics.model_dump()
    dead = []
    for ws in state.subscribers:
        try:
            await ws.send_json(state.latest_metrics)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.subscribers.remove(ws)
    return {"ok": True}


@app.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket):
    await ws.accept()
    state.subscribers.append(ws)
    logger.info(f"WS subscriber connected. Total: {len(state.subscribers)}")
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        state.subscribers.remove(ws)
        logger.info("WS subscriber disconnected.")