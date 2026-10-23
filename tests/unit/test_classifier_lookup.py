from __future__ import annotations

import numpy as np

from src.adapters.classifier_lookup import LookupTableAdapter
from src.core.models import (
    FaceLandmarks,
    HandLandmarks,
    LandmarkFrame,
    LandmarkWindow,
    PoseLandmarks,
)


def _make_window(hands_present: tuple[bool, bool] = (True, True), n_frames: int = 5) -> LandmarkWindow:
    frames = [
        LandmarkFrame(
            timestamp_ms=i * 33,
            hands=HandLandmarks(
                left=np.zeros((21, 3)) if hands_present[0] else None,
                right=np.zeros((21, 3)) if hands_present[1] else None,
            ),
            face=FaceLandmarks(points=np.zeros((478, 3))),
            pose=PoseLandmarks(points=np.zeros((33, 3))),
            hand_presence=hands_present,
        )
        for i in range(n_frames)
    ]
    return LandmarkWindow(frames=frames, window_size_ms=500, stride_ms=250)


def test_both_hands_returns_hello():
    adapter = LookupTableAdapter()
    result = adapter.classify(_make_window((True, True)))
    assert len(result) == 1
    assert result[0].sign_id == "HELLO"
    assert result[0].confidence == 0.3


def test_single_hand_returns_unknown():
    adapter = LookupTableAdapter()
    result = adapter.classify(_make_window((False, True)))
    assert len(result) == 1
    assert result[0].sign_id == "UNKNOWN"
    assert result[0].confidence == 0.1


def test_no_hands_returns_empty():
    adapter = LookupTableAdapter()
    result = adapter.classify(_make_window((False, False)))
    assert result == []


def test_empty_window_returns_empty():
    adapter = LookupTableAdapter()
    window = LandmarkWindow(frames=[], window_size_ms=500, stride_ms=250)
    assert adapter.classify(window) == []


def test_vocabulary_size():
    assert LookupTableAdapter().get_vocabulary_size() == 5


def test_does_not_support_continuous():
    assert LookupTableAdapter().supports_continuous() is False


def test_temporal_span_matches_window():
    adapter = LookupTableAdapter()
    window = _make_window((True, True), n_frames=10)
    result = adapter.classify(window)
    assert result[0].temporal_span == (0, 9 * 33)
