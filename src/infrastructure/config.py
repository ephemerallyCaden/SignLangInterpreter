from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class CameraConfig(BaseModel):
    index: int = 0
    video_file: str | None = None


class PipelineConfig(BaseModel):
    window_size_ms: int = 1000
    stride_ms: int = 500
    max_queue_size: int = 3


class AppConfig(BaseModel):
    camera: CameraConfig = CameraConfig()
    pipeline: PipelineConfig = PipelineConfig()
    show_overlay: bool = True


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return AppConfig(**data)
