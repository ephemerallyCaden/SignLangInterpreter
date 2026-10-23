from __future__ import annotations

import pytest
import numpy as np

from src.core.models import (
    DominantHand,
    FaceLandmarks,
    GlossCandidate,
    GlossSequence,
    GlossToken,
    HandLandmarks,
    LandmarkFrame,
    LandmarkWindow,
    NMSSignal,
    NMSType,
    PoseLandmarks,
    SpatialSlot,
    TranslationResult,
)


def test_landmark_frame_creation():
    hands = HandLandmarks(left=np.zeros((21, 3)), right=np.zeros((21, 3)))
    face = FaceLandmarks(points=np.zeros((478, 3)))
    pose = PoseLandmarks(points=np.zeros((33, 3)))
    frame = LandmarkFrame(0, hands, face, pose, (True, True))
    assert frame.timestamp_ms == 0
    assert frame.hand_presence == (True, True)
    assert frame.raw_hands is None  # optional, not provided


def test_landmark_frame_is_frozen():
    hands = HandLandmarks(left=None, right=None)
    face = FaceLandmarks(points=np.zeros((478, 3)))
    pose = PoseLandmarks(points=np.zeros((33, 3)))
    frame = LandmarkFrame(0, hands, face, pose, (False, False))
    with pytest.raises(AttributeError):
        frame.timestamp_ms = 1  # type: ignore[misc]


def test_gloss_token_defaults():
    token = GlossToken("HELLO", 0.95, (0, 500), DominantHand.RIGHT)
    assert token.spatial_slot is None
    assert token.is_establishing is False
    assert token.is_referencing is False
    assert token.movement_vector is None


def test_gloss_candidate_defaults():
    candidate = GlossCandidate("STORE", 0.8, (100, 600))
    assert candidate.spatial_slot is None
    assert candidate.is_transition is False


def test_nms_signal():
    signal = NMSSignal(NMSType.QUESTION_YES_NO, 0.9, (0, 1000), 0.7)
    assert signal.signal_type == NMSType.QUESTION_YES_NO


def test_gloss_sequence_is_frozen():
    token = GlossToken("GO", 0.9, (0, 500), DominantHand.RIGHT)
    seq = GlossSequence(tokens=[token], nms_signals=[], spatial_context={}, sequence_duration_ms=500)
    with pytest.raises(AttributeError):
        seq.sequence_duration_ms = 999  # type: ignore[misc]


def test_translation_result():
    token = GlossToken("HELLO", 0.95, (0, 300), DominantHand.RIGHT)
    seq = GlossSequence([token], [], {}, 300)
    result = TranslationResult("Hello", 0.95, seq, 45)
    assert result.text == "Hello"
    assert result.processing_time_ms == 45


def test_spatial_slot_values():
    assert SpatialSlot.CENTER.value == "center"
    assert len(SpatialSlot) == 6


def test_nms_type_values():
    assert NMSType.NEGATION.value == "neg"
    assert len(NMSType) == 6


def test_dominant_hand_enum():
    assert DominantHand.LEFT.value == "left"
    assert DominantHand.RIGHT.value == "right"
    assert DominantHand.BOTH.value == "both"
    assert len(DominantHand) == 3
