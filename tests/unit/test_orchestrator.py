from __future__ import annotations

import numpy as np

from src.core.models import (
    FaceLandmarks,
    GlossCandidate,
    HandLandmarks,
    LandmarkFrame,
    LandmarkWindow,
    PoseLandmarks,
)
from src.core.orchestrator import TranslationOrchestrator
from src.ports.grammar_transformer import GrammarTransformer
from src.ports.sign_classifier import SignClassifier


def _make_frame(ts: int, hands_present: tuple[bool, bool] = (True, True)) -> LandmarkFrame:
    return LandmarkFrame(
        timestamp_ms=ts,
        hands=HandLandmarks(
            left=np.zeros((21, 3)) if hands_present[0] else None,
            right=np.zeros((21, 3)) if hands_present[1] else None,
        ),
        face=FaceLandmarks(points=np.zeros((468, 3))),
        pose=PoseLandmarks(points=np.zeros((33, 3))),
        hand_presence=hands_present,
    )


class StubClassifier(SignClassifier):
    def __init__(self, results: list[list[GlossCandidate]]):
        self._results = results
        self._call_count = 0

    def classify(self, window: LandmarkWindow) -> list[GlossCandidate]:
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

    def transform(self, sequence):
        self.call_count += 1
        from src.core.models import TranslationResult

        return TranslationResult(
            text="stub",
            confidence=0.9,
            gloss_sequence=sequence,
            processing_time_ms=1,
        )


def test_orchestrator_needs_enough_frames():
    classifier = StubClassifier([])
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=500, stride_ms=100)

    # Fewer than 5 frames: should return None
    for i in range(4):
        result = orch.process_frame(_make_frame(i * 33))
    assert transformer.call_count == 0


def test_orchestrator_produces_result():
    candidates = [GlossCandidate("HELLO", 0.9, (0, 200))]
    classifier = StubClassifier([candidates])
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=500, stride_ms=0)

    # Feed enough frames
    for i in range(6):
        result = orch.process_frame(_make_frame(i * 33))

    assert transformer.call_count >= 1


def test_orchestrator_skips_empty_classification():
    classifier = StubClassifier([[]])
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=500, stride_ms=0)

    for i in range(10):
        orch.process_frame(_make_frame(i * 33))

    assert transformer.call_count == 0


def test_orchestrator_filters_transitions():
    candidates = [GlossCandidate("HELLO", 0.9, (0, 200), is_transition=True)]
    classifier = StubClassifier([candidates])
    transformer = StubTransformer()
    orch = TranslationOrchestrator(classifier, transformer, window_size_ms=500, stride_ms=0)

    for i in range(10):
        orch.process_frame(_make_frame(i * 33))

    assert transformer.call_count == 0
