from __future__ import annotations

import cv2
import numpy as np

from src.core.models import HandLandmarks, LandmarkFrame, TranslationResult
from src.ports.output_sink import OutputSink

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


class OverlayOutputAdapter(OutputSink):
    def __init__(self) -> None:
        self._last_landmarks: LandmarkFrame | None = None
        self._last_text: str | None = None

    def set_landmarks(self, landmarks: LandmarkFrame) -> None:
        self._last_landmarks = landmarks

    def emit(self, result: TranslationResult, frame: np.ndarray | None = None) -> None:
        self._last_text = f"{result.text} ({result.confidence:.0%})"

    def draw_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw landmarks and translation text on a copy of the frame."""
        vis = frame.copy()
        if self._last_landmarks is not None:
            # Use raw (un-normalized) hand coords for display
            raw = self._last_landmarks.raw_hands
            if raw is not None:
                self._draw_hands(vis, raw)
        if self._last_text:
            self._draw_text(vis, self._last_text)
        return vis

    def _draw_hands(self, frame: np.ndarray, hands: HandLandmarks) -> None:
        h, w = frame.shape[:2]
        for hand_pts in [hands.left, hands.right]:
            if hand_pts is None:
                continue
            if hand_pts.shape[0] < 21 or hand_pts.shape[1] < 2:
                continue
            pts = hand_pts[:, :2]
            pixel_pts = (pts * [w, h]).astype(int)
            for i, j in HAND_CONNECTIONS:
                cv2.line(frame, tuple(pixel_pts[i]), tuple(pixel_pts[j]), (0, 255, 0), 2)
            for pt in pixel_pts:
                cv2.circle(frame, tuple(pt), 3, (0, 0, 255), -1)

    def _draw_text(self, frame: np.ndarray, text: str) -> None:
        h = frame.shape[0]
        cv2.putText(
            frame, text, (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
        )
