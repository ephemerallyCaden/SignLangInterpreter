from __future__ import annotations

import numpy as np

from src.core.models import FaceLandmarks, HandLandmarks, LandmarkFrame, PoseLandmarks
from src.infrastructure.smoothing import HandPresenceSmoother


def _frame(ts: int, hands: tuple[bool, bool]) -> LandmarkFrame:
    return LandmarkFrame(
        timestamp_ms=ts,
        hands=HandLandmarks(
            left=np.zeros((21, 3)) if hands[0] else None,
            right=np.zeros((21, 3)) if hands[1] else None,
        ),
        face=FaceLandmarks(points=np.zeros((478, 3))),
        pose=PoseLandmarks(points=np.zeros((33, 3))),
        hand_presence=hands,
    )


def test_hand_stays_present_during_brief_dropout():
    smoother = HandPresenceSmoother(grace_frames=3)

    # Hand detected
    f = smoother.smooth(_frame(0, (True, False)))
    assert f.hand_presence == (True, False)

    # Hand drops for 2 frames (below grace=3)
    f = smoother.smooth(_frame(33, (False, False)))
    assert f.hand_presence == (True, False)  # still smoothed as present

    f = smoother.smooth(_frame(66, (False, False)))
    assert f.hand_presence == (True, False)  # still smoothed

    # Hand comes back
    f = smoother.smooth(_frame(100, (True, False)))
    assert f.hand_presence == (True, False)


def test_hand_lost_after_grace_period():
    smoother = HandPresenceSmoother(grace_frames=3)

    # Hand detected
    smoother.smooth(_frame(0, (True, False)))

    # Hand drops for grace=3 frames
    smoother.smooth(_frame(33, (False, False)))
    smoother.smooth(_frame(66, (False, False)))
    f = smoother.smooth(_frame(100, (False, False)))
    assert f.hand_presence == (False, False)  # now truly lost


def test_both_hands_tracked_independently():
    smoother = HandPresenceSmoother(grace_frames=2)

    smoother.smooth(_frame(0, (True, True)))

    # Left drops, right stays
    f = smoother.smooth(_frame(33, (False, True)))
    assert f.hand_presence == (True, True)  # left still in grace

    f = smoother.smooth(_frame(66, (False, True)))
    assert f.hand_presence == (False, True)  # left grace expired


def test_no_smoothing_needed_when_detection_stable():
    smoother = HandPresenceSmoother(grace_frames=5)

    for i in range(10):
        f = smoother.smooth(_frame(i * 33, (True, True)))
        assert f.hand_presence == (True, True)


def test_initial_state_is_absent():
    smoother = HandPresenceSmoother(grace_frames=3)

    # First frame with no hands — should stay absent (no prior detection to smooth)
    f = smoother.smooth(_frame(0, (False, False)))
    assert f.hand_presence == (False, False)
