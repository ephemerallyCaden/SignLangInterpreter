from __future__ import annotations

from src.adapters.transformer_rules import RuleBasedAdapter
from src.core.models import (
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
        GlossToken(sign_id=s, confidence=0.9, temporal_span=(0, 500), dominant_hand="right")
        for s in sign_ids
    ]
    return GlossSequence(
        tokens=tokens,
        nms_signals=nms or [],
        spatial_context=spatial or {},
        sequence_duration_ms=500,
    )


def test_simple_word_mapping():
    adapter = RuleBasedAdapter()
    seq = _make_sequence(["HELLO"])
    result = adapter.transform(seq)
    assert "hello" in result.text.lower()


def test_multiple_glosses():
    adapter = RuleBasedAdapter()
    seq = _make_sequence(["MOTHER", "GO", "STORE"])
    result = adapter.transform(seq)
    assert "mother" in result.text.lower()
    assert "go" in result.text.lower()
    assert "store" in result.text.lower()


def test_yes_no_question_nms():
    adapter = RuleBasedAdapter()
    nms = [NMSSignal(NMSType.QUESTION_YES_NO, 0.9, (0, 500), 0.8)]
    seq = _make_sequence(["GO", "STORE"], nms=nms)
    result = adapter.transform(seq)
    assert result.text.endswith("?")


def test_negation_nms():
    adapter = RuleBasedAdapter()
    nms = [NMSSignal(NMSType.NEGATION, 0.9, (0, 500), 0.7)]
    seq = _make_sequence(["GO", "STORE"], nms=nms)
    result = adapter.transform(seq)
    assert "not" in result.text.lower()


def test_empty_sequence():
    adapter = RuleBasedAdapter()
    seq = _make_sequence([])
    result = adapter.transform(seq)
    assert result.text == "..."
    assert result.confidence == 0.0


def test_spatial_reference_resolution():
    adapter = RuleBasedAdapter()
    tokens = [
        GlossToken("MOTHER", 0.9, (0, 200), "right", spatial_slot=SpatialSlot.LEFT_NEAR),
        GlossToken(
            "POINT", 0.9, (200, 400), "right",
            spatial_slot=SpatialSlot.LEFT_NEAR,
            is_referencing=True,
        ),
        GlossToken("GO", 0.9, (400, 600), "right"),
    ]
    seq = GlossSequence(
        tokens=tokens,
        nms_signals=[],
        spatial_context={SpatialSlot.LEFT_NEAR: "MOTHER"},
        sequence_duration_ms=600,
    )
    result = adapter.transform(seq)
    # POINT:LEFT_NEAR should resolve to "mother" from spatial context
    text = result.text.lower()
    assert text.count("mother") == 2  # once for the sign, once for the reference


def test_confidence_aggregation():
    adapter = RuleBasedAdapter()
    tokens = [
        GlossToken("HELLO", 0.8, (0, 200), "right"),
        GlossToken("GO", 0.6, (200, 400), "right"),
    ]
    seq = GlossSequence(tokens=tokens, nms_signals=[], spatial_context={}, sequence_duration_ms=400)
    result = adapter.transform(seq)
    assert abs(result.confidence - 0.7) < 0.01
