from __future__ import annotations

import json

import numpy as np

from src.core.models import (
    DominantHand,
    FaceLandmarks,
    GlossSequence,
    GlossToken,
    HandLandmarks,
    LandmarkFrame,
    PoseLandmarks,
    TranslationResult,
)
from src.infrastructure.diagnostics import SessionRecorder


def _make_frame(ts: int = 0, hands: tuple[bool, bool] = (True, True)) -> LandmarkFrame:
    return LandmarkFrame(
        timestamp_ms=ts,
        hands=HandLandmarks(
            left=np.zeros((21, 3)) if hands[0] else None,
            right=np.zeros((21, 3)) if hands[1] else None,
        ),
        face=FaceLandmarks(points=np.ones((478, 3))),
        pose=PoseLandmarks(points=np.ones((33, 3))),
        hand_presence=hands,
        raw_hands=HandLandmarks(
            left=np.random.rand(21, 3).astype(np.float32) if hands[0] else None,
            right=np.random.rand(21, 3).astype(np.float32) if hands[1] else None,
        ),
    )


def _make_result(text: str = "Hello.", confidence: float = 0.3) -> TranslationResult:
    token = GlossToken("HELLO", confidence, (0, 500), DominantHand.RIGHT)
    seq = GlossSequence([token], [], {}, 500)
    return TranslationResult(text, confidence, seq, 1)


def test_session_creates_directory(tmp_path):
    rec = SessionRecorder(output_dir=str(tmp_path / "sess"), snapshot_interval=1)
    rec.open()
    assert rec.session_dir.exists()
    assert (rec.session_dir / "frames").is_dir()
    rec.close()


def test_summary_written_on_close(tmp_path):
    rec = SessionRecorder(output_dir=str(tmp_path / "sess"), snapshot_interval=100)
    rec.open()

    frame_img = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = _make_frame(ts=0, hands=(True, False))
    rec.record_frame(frame_img, landmarks, None, 15.0, 1, 0.25)
    rec.record_frame(frame_img, landmarks, None, 16.0, 2, 0.25)
    rec.close()

    summary = json.loads((rec.session_dir / "summary.json").read_text())
    assert summary["total_frames"] == 2
    assert summary["frames_with_hands"] == 2
    assert summary["avg_extraction_latency_ms"] == 15.5


def test_metrics_jsonl_written(tmp_path):
    rec = SessionRecorder(output_dir=str(tmp_path / "sess"), snapshot_interval=100)
    rec.open()

    frame_img = np.zeros((480, 640, 3), dtype=np.uint8)
    rec.record_frame(frame_img, _make_frame(0), None, 10.0, 1, 0.2)
    rec.record_frame(frame_img, _make_frame(33), _make_result(), 12.0, 2, 0.2)
    rec.close()

    lines = (rec.session_dir / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2

    m0 = json.loads(lines[0])
    assert m0["frame"] == 0
    assert m0["translation"] is None

    m1 = json.loads(lines[1])
    assert m1["frame"] == 1
    assert m1["translation"]["text"] == "Hello."


def test_events_log_records_transitions(tmp_path):
    rec = SessionRecorder(output_dir=str(tmp_path / "sess"), snapshot_interval=100)
    rec.open()

    frame_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # No hands -> hands detected
    rec.record_frame(frame_img, _make_frame(0, (True, True)), None, 10.0, 1, 0.2)
    # Hands lost
    rec.record_frame(frame_img, _make_frame(33, (False, False)), None, 10.0, 1, 0.2)
    rec.close()

    events = (rec.session_dir / "events.log").read_text()
    assert "Hands detected" in events
    assert "Hands lost" in events
    assert "Session ended" in events


def test_snapshots_saved_at_interval(tmp_path):
    rec = SessionRecorder(output_dir=str(tmp_path / "sess"), snapshot_interval=2)
    rec.open()

    frame_img = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(5):
        rec.record_frame(frame_img, _make_frame(i * 33), None, 10.0, 1, 0.2)
    rec.close()

    frames_dir = rec.session_dir / "frames"
    jpgs = list(frames_dir.glob("*.jpg"))
    # Frames 0, 2, 4 should be saved (every 2nd frame)
    assert len(jpgs) == 3


def test_translation_counted_in_summary(tmp_path):
    rec = SessionRecorder(output_dir=str(tmp_path / "sess"), snapshot_interval=100)
    rec.open()

    frame_img = np.zeros((480, 640, 3), dtype=np.uint8)
    rec.record_frame(frame_img, _make_frame(0), _make_result("Hi.", 0.8), 10.0, 1, 0.2)
    rec.record_frame(frame_img, _make_frame(33), _make_result("Go.", 0.6), 10.0, 2, 0.2)
    rec.close()

    summary = json.loads((rec.session_dir / "summary.json").read_text())
    assert summary["translations_emitted"] == 2
    assert summary["avg_classification_confidence"] == 0.7


def test_no_landmarks_frame(tmp_path):
    rec = SessionRecorder(output_dir=str(tmp_path / "sess"), snapshot_interval=100)
    rec.open()

    frame_img = np.zeros((480, 640, 3), dtype=np.uint8)
    rec.record_frame(frame_img, None, None, 0.0, 0, None)
    rec.close()

    summary = json.loads((rec.session_dir / "summary.json").read_text())
    assert summary["total_frames"] == 1
    assert summary["frames_with_hands"] == 0
    assert summary["hand_detection_rate"] == 0.0
