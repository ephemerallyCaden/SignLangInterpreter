from __future__ import annotations

import numpy as np

from src.core.models import (
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
    hands = HandLandmarks(
        left=np.zeros((21, 3)),
        right=np.zeros((21, 3)),
    )
    face = FaceLandmarks(points=np.zeros((468, 3)))
    pose = PoseLandmarks(points=np.zeros((33, 3)))

    frame = LandmarkFrame(
        timestamp_ms=0,
        hands=hands,
        face=face,
        pose=pose,
        hand_presence=(True, True),
    )
    assert frame.timestamp_ms == 0
    assert frame.hand_presence == (True, True)


def test_landmark_frame_is_frozen():
    hands = HandLandmarks(left=None, right=None)
    face = FaceLandmarks(points=np.zeros((468, 3)))
    pose = PoseLandmarks(points=np.zeros((33, 3)))
    frame = LandmarkFrame(0, hands, face, pose, (False, False))

    try:
        frame.timestamp_ms = 1  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_gloss_token_defaults():
    token = GlossToken(
        sign_id="HELLO",
        confidence=0.95,
        temporal_span=(0, 500),
        dominant_hand="right",
    )
    assert token.spatial_slot is None
    assert token.is_establishing is False
    assert token.is_referencing is False
    assert token.movement_vector is None


def test_gloss_candidate_defaults():
    candidate = GlossCandidate(
        sign_id="STORE",
        confidence=0.8,
        temporal_span=(100, 600),
    )
    assert candidate.spatial_slot is None
    assert candidate.is_transition is False


def test_nms_signal():
    signal = NMSSignal(
        signal_type=NMSType.QUESTION_YES_NO,
        confidence=0.9,
        temporal_span=(0, 1000),
        intensity=0.7,
    )
    assert signal.signal_type == NMSType.QUESTION_YES_NO


def test_gloss_sequence():
    token = GlossToken("GO", 0.9, (0, 500), "right")
    seq = GlossSequence(
        tokens=[token],
        nms_signals=[],
        spatial_context={},
        sequence_duration_ms=500,
    )
    assert len(seq.tokens) == 1
    assert seq.sequence_duration_ms == 500


def test_translation_result():
    token = GlossToken("HELLO", 0.95, (0, 300), "right")
    seq = GlossSequence([token], [], {}, 300)
    result = TranslationResult(
        text="Hello",
        confidence=0.95,
        gloss_sequence=seq,
        processing_time_ms=45,
    )
    assert result.text == "Hello"
    assert result.processing_time_ms == 45


def test_spatial_slot_values():
    assert SpatialSlot.CENTER.value == "center"
    assert len(SpatialSlot) == 6


def test_nms_type_values():
    assert NMSType.NEGATION.value == "neg"
    assert len(NMSType) == 6
