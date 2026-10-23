from __future__ import annotations

from typing import Any

import numpy as np

from src.core.models import (
    DominantHand,
    GlossCandidate,
    GlossSequence,
    LandmarkWindow,
    TranslationResult,
)
from src.core.orchestrator import MIN_CONFIDENCE, MIN_FRAMES_FOR_CLASSIFICATION, TranslationOrchestrator
from src.ports.grammar_transformer import GrammarTransformer
from src.ports.sign_classifier import SignClassifier


class StubClassifier(SignClassifier):
    def __init__(self, results: list[list[GlossCandidate]] | None = None):
        self._results = results or []
        self._call_count = 0
        self.received_windows: list[LandmarkWindow] = []

    def classify(self, window: LandmarkWindow) -> list[GlossCandidate]:
        self.received_windows.append(window)
        if self._call_count < len(self._results):
            result = self._results[self._call_count]
            self._call_count += 1
            return result
        return []

    def get_vocabulary_size(self) -> int:
        return 10

    def supports_continuous(self) -> bool:
        return False


class StubTransformer(GrammarTransformer):
    def __init__(self):
        self.call_count = 0
        self.last_sequence: GlossSequence | None = None

    def transform(self, sequence: GlossSequence, context: dict[str, Any] | None = None):
        self.call_count += 1
        self.last_sequence = sequence
        return TranslationResult(
            text="stub", confidence=0.9, gloss_sequence=sequence, processing_time_ms=1
        )


def test_needs_enough_frames(make_frame):
    classifier = StubClassifier()
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=500, stride_ms=100)

    for i in range(MIN_FRAMES_FOR_CLASSIFICATION - 1):
        result = orch.process_frame(make_frame(ts=i * 33))
        assert result is None
    assert transformer.call_count == 0


def test_produces_result_and_returns_it(make_frame):
    candidates = [GlossCandidate("HELLO", 0.9, (0, 200))]
    classifier = StubClassifier([candidates])
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=500, stride_ms=0)

    result = None
    for i in range(MIN_FRAMES_FOR_CLASSIFICATION + 1):
        r = orch.process_frame(make_frame(ts=i * 33))
        if r is not None:
            result = r

    assert result is not None
    assert result.text == "stub"
    assert transformer.call_count >= 1
    assert transformer.last_sequence is not None
    assert transformer.last_sequence.tokens[0].sign_id == "HELLO"


def test_skips_empty_classification(make_frame):
    classifier = StubClassifier([[]])
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=500, stride_ms=0)

    for i in range(10):
        orch.process_frame(make_frame(ts=i * 33))
    assert transformer.call_count == 0


def test_filters_transitions(make_frame):
    candidates = [GlossCandidate("HELLO", 0.9, (0, 200), is_transition=True)]
    classifier = StubClassifier([candidates])
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=500, stride_ms=0)

    for i in range(10):
        orch.process_frame(make_frame(ts=i * 33))
    assert transformer.call_count == 0


def test_filters_low_confidence(make_frame):
    candidates = [GlossCandidate("HELLO", MIN_CONFIDENCE, (0, 200))]
    classifier = StubClassifier([candidates])
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=500, stride_ms=0)

    for i in range(10):
        orch.process_frame(make_frame(ts=i * 33))
    assert transformer.call_count == 0  # exactly at threshold, not above


def test_confidence_just_above_threshold(make_frame):
    candidates = [GlossCandidate("HELLO", MIN_CONFIDENCE + 0.01, (0, 200))]
    classifier = StubClassifier([candidates])
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=500, stride_ms=0)

    for i in range(10):
        orch.process_frame(make_frame(ts=i * 33))
    assert transformer.call_count >= 1


def test_first_classification_fires_immediately(make_frame):
    candidates = [GlossCandidate("HELLO", 0.9, (0, 200))]
    classifier = StubClassifier([candidates])
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=2000, stride_ms=500)

    # Feed exactly enough frames — first classification should fire without waiting for stride
    for i in range(MIN_FRAMES_FOR_CLASSIFICATION + 1):
        orch.process_frame(make_frame(ts=i * 33))

    assert transformer.call_count == 1


def test_stride_throttles_subsequent_calls(make_frame):
    candidates = [GlossCandidate("HELLO", 0.9, (0, 200))]
    # Provide enough results for multiple calls
    classifier = StubClassifier([candidates, candidates, candidates, candidates])
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=5000, stride_ms=500)

    # Feed 30 frames over 1000ms — stride=500 means at most 3 classifications
    for i in range(30):
        orch.process_frame(make_frame(ts=i * 33))

    assert 1 <= transformer.call_count <= 3


def test_trim_buffer_discards_old_frames(make_frame):
    classifier = StubClassifier()
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=100, stride_ms=0)

    # Feed frames spanning 1000ms with 100ms window
    for i in range(30):
        orch.process_frame(make_frame(ts=i * 33))

    # Buffer should only contain frames within last 100ms
    assert len(orch._frame_buffer) <= 4  # ~3 frames at 30fps in 100ms


def test_dominant_hand_inferred_from_presence(make_frame):
    candidates = [GlossCandidate("HELLO", 0.9, (0, 200))]
    classifier = StubClassifier([candidates, candidates, candidates])
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=500, stride_ms=0)

    # Both hands
    for i in range(6):
        orch.process_frame(make_frame(ts=i * 33, hands_present=(True, True)))
    assert transformer.last_sequence is not None
    assert transformer.last_sequence.tokens[0].dominant_hand == DominantHand.BOTH

    # Left only
    transformer.last_sequence = None
    orch2 = TranslationOrchestrator(
        StubClassifier([candidates]), StubTransformer(), window_size_ms=500, stride_ms=0
    )
    for i in range(6):
        orch2.process_frame(make_frame(ts=i * 33, hands_present=(True, False)))
