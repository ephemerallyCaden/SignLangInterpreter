from __future__ import annotations

from src.core.models import LandmarkFrame

# Number of consecutive "absent" frames before a hand is considered truly lost.
# At 30fps, 5 frames = ~167ms of grace period.
HAND_LOSS_GRACE_FRAMES = 5


class HandPresenceSmoother:
    """Smooths hand detection flicker by requiring N consecutive absent frames
    before declaring a hand lost. Prevents rapid on/off oscillation."""

    def __init__(self, grace_frames: int = HAND_LOSS_GRACE_FRAMES):
        self._grace = grace_frames
        self._left_absent_count = 0
        self._right_absent_count = 0
        self._smoothed_left = False
        self._smoothed_right = False

    def smooth(self, frame: LandmarkFrame) -> LandmarkFrame:
        raw_left, raw_right = frame.hand_presence

        # Left hand
        if raw_left:
            self._left_absent_count = 0
            self._smoothed_left = True
        else:
            self._left_absent_count += 1
            if self._left_absent_count >= self._grace:
                self._smoothed_left = False

        # Right hand
        if raw_right:
            self._right_absent_count = 0
            self._smoothed_right = True
        else:
            self._right_absent_count += 1
            if self._right_absent_count >= self._grace:
                self._smoothed_right = False

        smoothed_presence = (self._smoothed_left, self._smoothed_right)

        if smoothed_presence == frame.hand_presence:
            return frame

        # Return a new frame with smoothed hand presence
        return LandmarkFrame(
            timestamp_ms=frame.timestamp_ms,
            hands=frame.hands,
            face=frame.face,
            pose=frame.pose,
            hand_presence=smoothed_presence,
            raw_hands=frame.raw_hands,
        )
