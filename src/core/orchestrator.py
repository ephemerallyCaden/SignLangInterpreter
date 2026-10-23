from __future__ import annotations

import logging
import time

from src.core.models import (
    GlossSequence,
    GlossToken,
    LandmarkFrame,
    LandmarkWindow,
    TranslationResult,
)
from src.ports.grammar_transformer import GrammarTransformer
from src.ports.sign_classifier import SignClassifier

log = logging.getLogger(__name__)


class TranslationOrchestrator:
    """Simplified Phase 1 orchestrator: classifier → grammar transformer.

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
        self._last_translate_ms: int = 0

    def process_frame(self, frame: LandmarkFrame) -> TranslationResult | None:
        self._frame_buffer.append(frame)
        self._trim_buffer(frame.timestamp_ms)

        # Only attempt classification when we have enough frames
        if len(self._frame_buffer) < 5:
            return None

        # Throttle: only translate every stride_ms
        if frame.timestamp_ms - self._last_translate_ms < self._stride_ms:
            return None

        window = LandmarkWindow(
            frames=list(self._frame_buffer),
            window_size_ms=self._window_size_ms,
            stride_ms=self._stride_ms,
        )

        candidates = self._classifier.classify(window)
        if not candidates:
            return None

        # Convert candidates to tokens (Phase 1: take top candidate only)
        tokens = [
            GlossToken(
                sign_id=c.sign_id,
                confidence=c.confidence,
                temporal_span=c.temporal_span,
                dominant_hand="right",
                spatial_slot=c.spatial_slot,
            )
            for c in candidates
            if not c.is_transition and c.confidence > 0.05
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
