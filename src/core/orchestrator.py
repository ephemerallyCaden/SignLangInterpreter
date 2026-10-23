from __future__ import annotations

import logging

from src.core.models import (
    DominantHand,
    GlossSequence,
    GlossToken,
    LandmarkFrame,
    LandmarkWindow,
)
from src.ports.grammar_transformer import GrammarTransformer
from src.ports.sign_classifier import SignClassifier

log = logging.getLogger(__name__)

MIN_FRAMES_FOR_CLASSIFICATION = 5
MIN_CONFIDENCE = 0.05


class TranslationOrchestrator:
    """Simplified Phase 1 orchestrator: classifier -> grammar transformer.

    No NMS, no spatial registry, no confidence routing yet.
    Those are Phase 1.5 additions.
    """

    def __init__(
        self,
        sign_classifier: SignClassifier,
        grammar_transformer: GrammarTransformer,
        window_size_ms: int = 1000,
        stride_ms: int = 500,
    ):
        self._classifier = sign_classifier
        self._transformer = grammar_transformer
        self._window_size_ms = window_size_ms
        self._stride_ms = stride_ms
        self._frame_buffer: list[LandmarkFrame] = []
        self._last_translate_ms: int | None = None

    def process_frame(self, frame: LandmarkFrame):
        self._frame_buffer.append(frame)
        self._trim_buffer(frame.timestamp_ms)

        if len(self._frame_buffer) < MIN_FRAMES_FOR_CLASSIFICATION:
            return None

        # First classification fires immediately; subsequent ones throttled by stride
        if (
            self._last_translate_ms is not None
            and frame.timestamp_ms - self._last_translate_ms < self._stride_ms
        ):
            return None

        window = LandmarkWindow(
            frames=list(self._frame_buffer),
            window_size_ms=self._window_size_ms,
            stride_ms=self._stride_ms,
        )

        candidates = self._classifier.classify(window)
        if not candidates:
            return None

        tokens = [
            GlossToken(
                sign_id=c.sign_id,
                confidence=c.confidence,
                temporal_span=c.temporal_span,
                dominant_hand=self._infer_dominant_hand(frame),
                spatial_slot=c.spatial_slot,
            )
            for c in candidates
            if not c.is_transition and c.confidence > MIN_CONFIDENCE
        ]

        if not tokens:
            return None

        sequence = GlossSequence(
            tokens=tokens,
            nms_signals=[],
            spatial_context={},
            sequence_duration_ms=self._window_size_ms,
        )

        self._last_translate_ms = frame.timestamp_ms
        return self._transformer.transform(sequence)

    def _trim_buffer(self, current_ms: int) -> None:
        cutoff = current_ms - self._window_size_ms
        self._frame_buffer = [f for f in self._frame_buffer if f.timestamp_ms >= cutoff]

    def _infer_dominant_hand(self, frame: LandmarkFrame) -> DominantHand:
        left, right = frame.hand_presence
        if left and right:
            return DominantHand.BOTH
        if left:
            return DominantHand.LEFT
        return DominantHand.RIGHT
