from __future__ import annotations

import logging

import cv2
import numpy as np

from src.ports.frame_source import FrameSource

log = logging.getLogger(__name__)


class CameraOpenCVAdapter(FrameSource):
    def __init__(self, camera_index: int = 0):
        self._camera_index = camera_index
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._camera_index}")
        log.info("Camera %d opened", self._camera_index)

    def acquire_frame(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def get_fps(self) -> float:
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Camera released")
