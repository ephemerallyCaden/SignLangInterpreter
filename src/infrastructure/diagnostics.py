from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.core.models import LandmarkFrame, TranslationResult

log = logging.getLogger(__name__)


class SessionRecorder:
    """Records pipeline diagnostics to a session directory.

    Produces:
    - summary.json: aggregate stats for the whole session
    - metrics.jsonl: per-frame metrics (one JSON line per frame)
    - frames/: annotated JPEG snapshots at configurable intervals
    - events.log: human-readable state transitions
    """

    def __init__(self, output_dir: str = "sessions", snapshot_interval: int = 30):
        self._snapshot_interval = snapshot_interval
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._session_dir = Path(output_dir) / ts
        self._frames_dir = self._session_dir / "frames"
        self._start_time = time.monotonic()
        self._frame_count = 0
        self._frames_with_hands = 0
        self._frames_with_pose = 0
        self._frames_with_face = 0
        self._translations_emitted = 0
        self._total_extract_ms = 0.0
        self._total_confidence = 0.0
        self._prev_hands: tuple[bool, bool] = (False, False)
        self._metrics_file = None
        self._events_file = None

    def open(self, config_dict: dict | None = None) -> None:
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_file = open(self._session_dir / "metrics.jsonl", "w")
        self._events_file = open(self._session_dir / "events.log", "w")
        self._start_time = time.monotonic()

        cfg_str = json.dumps(config_dict) if config_dict else "{}"
        self._write_event(f"Session started | config: {cfg_str}")
        log.info("Recording session to %s", self._session_dir)

    def record_frame(
        self,
        frame: np.ndarray,
        landmarks: LandmarkFrame | None,
        result: TranslationResult | None,
        extract_ms: float,
        buffer_size: int,
        shoulder_width: float | None,
        annotated_frame: np.ndarray | None = None,
    ) -> None:
        idx = self._frame_count
        self._frame_count += 1

        hands = (False, False)
        pose_ok = False
        face_ok = False
        ts_ms = 0

        if landmarks is not None:
            hands = landmarks.hand_presence
            pose_ok = bool(np.any(landmarks.pose.points != 0))
            face_ok = bool(np.any(landmarks.face.points != 0))
            ts_ms = landmarks.timestamp_ms

            if any(hands):
                self._frames_with_hands += 1
            if pose_ok:
                self._frames_with_pose += 1
            if face_ok:
                self._frames_with_face += 1

        self._total_extract_ms += extract_ms

        # Detect hand state transitions
        if hands != self._prev_hands:
            if any(hands) and not any(self._prev_hands):
                self._write_event(f"Hands detected | left={hands[0]}, right={hands[1]}")
            elif not any(hands) and any(self._prev_hands):
                self._write_event("Hands lost")
            self._prev_hands = hands

        # Record translation
        translation_data = None
        if result is not None:
            self._translations_emitted += 1
            self._total_confidence += result.confidence
            glosses = [t.sign_id for t in result.gloss_sequence.tokens]
            translation_data = {
                "text": result.text,
                "confidence": round(result.confidence, 3),
                "glosses": glosses,
                "processing_ms": result.processing_time_ms,
            }
            self._write_event(
                f"Translation: \"{result.text}\" | "
                f"confidence={result.confidence:.2f} | glosses={glosses}"
            )

        # Write per-frame metrics
        metric = {
            "frame": idx,
            "ts_ms": ts_ms,
            "hands": list(hands),
            "pose": pose_ok,
            "face": face_ok,
            "extract_ms": round(extract_ms, 1),
            "shoulder_width": round(shoulder_width, 4) if shoulder_width else None,
            "buffer_size": buffer_size,
            "translation": translation_data,
        }
        if self._metrics_file:
            self._metrics_file.write(json.dumps(metric) + "\n")
            if idx % 30 == 0:
                self._metrics_file.flush()

        # Save annotated snapshot
        if idx % self._snapshot_interval == 0:
            save_frame = annotated_frame if annotated_frame is not None else frame
            self._save_snapshot(save_frame, idx, hands, extract_ms, result)

    def close(self) -> None:
        elapsed = time.monotonic() - self._start_time
        self._write_event(
            f"Session ended | {self._frame_count} frames, "
            f"{self._translations_emitted} translations, {elapsed:.1f}s"
        )

        # Write summary
        summary = {
            "start_time": datetime.now().isoformat(),
            "duration_s": round(elapsed, 1),
            "total_frames": self._frame_count,
            "frames_with_hands": self._frames_with_hands,
            "frames_with_pose": self._frames_with_pose,
            "frames_with_face": self._frames_with_face,
            "translations_emitted": self._translations_emitted,
            "avg_extraction_latency_ms": (
                round(self._total_extract_ms / self._frame_count, 1)
                if self._frame_count > 0
                else 0
            ),
            "avg_classification_confidence": (
                round(self._total_confidence / self._translations_emitted, 3)
                if self._translations_emitted > 0
                else 0
            ),
            "hand_detection_rate": (
                round(self._frames_with_hands / self._frame_count, 3)
                if self._frame_count > 0
                else 0
            ),
        }
        with open(self._session_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if self._metrics_file:
            self._metrics_file.close()
        if self._events_file:
            self._events_file.close()

        log.info("Session saved to %s", self._session_dir)

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    def _write_event(self, msg: str) -> None:
        elapsed = time.monotonic() - self._start_time
        line = f"[{elapsed:.1f}s] {msg}\n"
        if self._events_file:
            self._events_file.write(line)
            self._events_file.flush()

    def _save_snapshot(
        self,
        frame: np.ndarray,
        idx: int,
        hands: tuple[bool, bool],
        extract_ms: float,
        result: TranslationResult | None,
    ) -> None:
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Status bar at top
        status_parts = [
            f"Frame {idx}",
            f"Hands: L={'Y' if hands[0] else 'N'} R={'Y' if hands[1] else 'N'}",
            f"Extract: {extract_ms:.0f}ms",
        ]
        if result:
            status_parts.append(f"=> {result.text} ({result.confidence:.0%})")
        status = " | ".join(status_parts)

        cv2.rectangle(vis, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(vis, status, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        path = self._frames_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(path), vis)
