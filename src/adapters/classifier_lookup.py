from __future__ import annotations

import numpy as np

from src.core.models import GlossCandidate, LandmarkWindow
from src.ports.sign_classifier import SignClassifier

# Stub vocabulary: hand-position heuristics for a handful of signs.
# This will be replaced by KNN/DTW on BSL-1K templates once dataset access is secured.
STUB_SIGNS = {
    "HELLO": "open hand, palm forward, near head",
    "THANK_YOU": "flat hand moving away from chin",
    "YES": "fist nodding",
    "NO": "index finger wagging",
    "POINT": "index finger extended",
}


class LookupTableAdapter(SignClassifier):
    """Stub classifier returning placeholder glosses based on hand presence.

    Real implementation will use KNN/DTW on BSL-1K landmark templates.
    For now, this returns a low-confidence UNKNOWN gloss so the rest of
    the pipeline can be tested end-to-end.
    """

    def classify(self, window: LandmarkWindow) -> list[GlossCandidate]:
        if not window.frames:
            return []

        last_frame = window.frames[-1]

        # Basic hand-presence detection to return _something_
        if not any(last_frame.hand_presence):
            return []

        # Check if both hands are up (crude "HELLO" heuristic)
        if all(last_frame.hand_presence):
            return [
                GlossCandidate(
                    sign_id="HELLO",
                    confidence=0.3,
                    temporal_span=(
                        window.frames[0].timestamp_ms,
                        window.frames[-1].timestamp_ms,
                    ),
                )
            ]

        # Single hand detected
        return [
            GlossCandidate(
                sign_id="UNKNOWN",
                confidence=0.1,
                temporal_span=(
                    window.frames[0].timestamp_ms,
                    window.frames[-1].timestamp_ms,
                ),
            )
        ]

    def get_vocabulary_size(self) -> int:
        return len(STUB_SIGNS)

    def supports_continuous(self) -> bool:
        return False
