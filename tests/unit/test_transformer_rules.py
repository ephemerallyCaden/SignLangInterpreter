from __future__ import annotations

from src.adapters.transformer_rules import RuleBasedAdapter
from src.core.models import (
    DominantHand,
    GlossSequence,
    GlossToken,
    NMSSignal,
    NMSType,
    SpatialSlot,
)


def _make_sequence(
    sign_ids: list[str],
    nms: list[NMSSignal] | None = None,
    spatial: dict[SpatialSlot, str] | None = None,
) -> GlossSequence:
    tokens = [
        GlossToken(sign_id=s, confidence=0.9, temporal_span=(0, 500), dominant_hand=DominantHand.RIGHT)
        for s in sign_ids
    ]
    return GlossSequence(
        tokens=tokens, nms_signals=nms or [], spatial_context=spatial or {},
        sequence_duration_ms=500,
    )


def test_simple_word_mapping():
    result = RuleBasedAdapter().transform(_make_sequence(["HELLO"]))
    assert result.text == "Hello."


def test_multiple_glosses():
    result = RuleBasedAdapter().transform(_make_sequence(["MOTHER", "GO", "STORE"]))
    assert result.text == "Mother go store."


def test_yes_no_question():
    nms = [NMSSignal(NMSType.QUESTION_YES_NO, 0.9, (0, 500), 0.8)]
    result = RuleBasedAdapter().transform(_make_sequence(["GO", "STORE"], nms=nms))
    assert result.text.endswith("?")
    assert "?" in result.text


def test_wh_question():
    nms = [NMSSignal(NMSType.QUESTION_WH, 0.9, (0, 500), 0.8)]
    result = RuleBasedAdapter().transform(_make_sequence(["GO", "STORE"], nms=nms))
    assert result.text.endswith("?")


def test_negation_multi_word():
    nms = [NMSSignal(NMSType.NEGATION, 0.9, (0, 500), 0.7)]
    result = RuleBasedAdapter().transform(_make_sequence(["GO", "STORE"], nms=nms))
    words = result.text.rstrip(".").lower().split()
    assert words[1] == "not"  # "not" inserted after first word


def test_negation_single_word():
    nms = [NMSSignal(NMSType.NEGATION, 0.9, (0, 500), 0.7)]
    result = RuleBasedAdapter().transform(_make_sequence(["GO"], nms=nms))
    assert "not" in result.text.lower()


def test_negation_plus_question():
    nms = [
        NMSSignal(NMSType.NEGATION, 0.9, (0, 500), 0.7),
        NMSSignal(NMSType.QUESTION_YES_NO, 0.9, (0, 500), 0.8),
    ]
    result = RuleBasedAdapter().transform(_make_sequence(["GO", "STORE"], nms=nms))
    assert "not" in result.text.lower()
    assert result.text.endswith("?")


def test_empty_sequence():
    result = RuleBasedAdapter().transform(_make_sequence([]))
    assert result.text == "..."
    assert result.confidence == 0.0


def test_spatial_reference_resolution():
    tokens = [
        GlossToken("MOTHER", 0.9, (0, 200), DominantHand.RIGHT, spatial_slot=SpatialSlot.LEFT_NEAR),
        GlossToken(
            "POINT", 0.9, (200, 400), DominantHand.RIGHT,
            spatial_slot=SpatialSlot.LEFT_NEAR, is_referencing=True,
        ),
        GlossToken("GO", 0.9, (400, 600), DominantHand.RIGHT),
    ]
    seq = GlossSequence(
        tokens=tokens, nms_signals=[],
        spatial_context={SpatialSlot.LEFT_NEAR: "MOTHER"},
        sequence_duration_ms=600,
    )
    result = RuleBasedAdapter().transform(seq)
    assert result.text.lower().count("mother") == 2


def test_unresolved_spatial_reference_defaults_to_they():
    tokens = [
        GlossToken(
            "POINT", 0.9, (0, 200), DominantHand.RIGHT,
            spatial_slot=SpatialSlot.RIGHT_FAR, is_referencing=True,
        ),
        GlossToken("GO", 0.9, (200, 400), DominantHand.RIGHT),
    ]
    seq = GlossSequence(tokens=tokens, nms_signals=[], spatial_context={}, sequence_duration_ms=400)
    result = RuleBasedAdapter().transform(seq)
    assert "they" in result.text.lower()


def test_tense_markers_suppressed():
    result = RuleBasedAdapter().transform(_make_sequence(["FUTURE", "GO", "STORE"]))
    assert "future" not in result.text.lower()
    assert "go" in result.text.lower()


def test_unknown_sign_falls_back_to_lowercase():
    result = RuleBasedAdapter().transform(_make_sequence(["DRINK"]))
    assert "drink" in result.text.lower()


def test_confidence_aggregation():
    tokens = [
        GlossToken("HELLO", 0.8, (0, 200), DominantHand.RIGHT),
        GlossToken("GO", 0.6, (200, 400), DominantHand.RIGHT),
    ]
    seq = GlossSequence(tokens=tokens, nms_signals=[], spatial_context={}, sequence_duration_ms=400)
    result = RuleBasedAdapter().transform(seq)
    assert abs(result.confidence - 0.7) < 0.01
