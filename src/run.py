from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from loguru import logger

from src.pipeline import TrackingPipeline


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--source", required=True, help="video path, RTSP URL, or camera index")
    parser.add_argument("--output", default=None, help="output video path")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.remove()
    logger.add(lambda m: print(m, end=""), level=cfg["logging"]["level"])

    source = int(args.source) if args.source.isdigit() else args.source

    pipeline = TrackingPipeline(cfg)

    total = 0
    for result in pipeline.run_video(
        source=source,
        output_path=args.output,
        show=args.show,
        show_heatmap=args.heatmap,
        max_frames=args.max_frames,
    ):
        total = result.frame_idx
        if result.frame_idx % 30 == 0:
            logger.info(
                f"Frame {result.frame_idx} | FPS {result.fps:.1f} | "
                f"tracks={len(result.tracks)} | lines={result.line_stats}"
            )

    logger.info(f"Done. Processed {total} frames.")


if __name__ == "__main__":
    main()