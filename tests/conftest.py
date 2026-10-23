from __future__ import annotations

import numpy as np
import pytest

from src.core.models import (
    DominantHand,
    FaceLandmarks,
    HandLandmarks,
    LandmarkFrame,
    PoseLandmarks,
)


@pytest.fixture
def make_frame():
    """Factory for creating LandmarkFrame with sensible defaults."""

    def _make(
        ts: int = 0,
        hands_present: tuple[bool, bool] = (True, True),
    ) -> LandmarkFrame:
        return LandmarkFrame(
            timestamp_ms=ts,
            hands=HandLandmarks(
                left=np.zeros((21, 3)) if hands_present[0] else None,
                right=np.zeros((21, 3)) if hands_present[1] else None,
            ),
            face=FaceLandmarks(points=np.zeros((478, 3))),
            pose=PoseLandmarks(points=np.zeros((33, 3))),
            hand_presence=hands_present,
        )

    return _make
