from __future__ import annotations

import time
from typing import Any

from src.core.models import GlossSequence, NMSType, TranslationResult
from src.ports.grammar_transformer import GrammarTransformer

WORD_MAP: dict[str, str] = {
    "HELLO": "hello",
    "THANK_YOU": "thank you",
    "YES": "yes",
    "NO": "no",
    "GO": "go",
    "STORE": "store",
    "MOTHER": "mother",
    "FATHER": "father",
    "FUTURE": "",
    "PAST": "",
    "POINT": "",
    "UNKNOWN": "?",
}


class RuleBasedAdapter(GrammarTransformer):
    """Phase 1 BSL -> English rules.

    Currently implements:
    - Word-for-word gloss mapping (no reordering yet)
    - NMS question marking (yes/no and wh -> '?')
    - NMS negation ('not' inserted before verb or as standalone)
    - Spatial reference resolution (POINT:slot -> referent from context)
    - Tense markers (FUTURE/PAST) suppressed from output

    Not yet implemented (Phase 2+):
    - Topic-comment -> SVO reordering
    - Actual tense transformation on verbs
    """

    def transform(
        self, sequence: GlossSequence, context: dict[str, Any] | None = None
    ) -> TranslationResult:
        start = time.perf_counter()

        words = self._glosses_to_words(sequence)
        text = " ".join(w for w in words if w)

        text = self._apply_nms(text, sequence)

        if not text.strip():
            text = "..."

        text = text[0].upper() + text[1:] if text else text
        if text and text[-1] not in ".!?":
            text += "."

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        confidence = self._aggregate_confidence(sequence)

        return TranslationResult(
            text=text,
            confidence=confidence,
            gloss_sequence=sequence,
            processing_time_ms=elapsed_ms,
        )

    def _glosses_to_words(self, sequence: GlossSequence) -> list[str]:
        words: list[str] = []
        for token in sequence.tokens:
            if token.is_referencing and token.spatial_slot:
                referent = sequence.spatial_context.get(token.spatial_slot)
                if referent:
                    words.append(WORD_MAP.get(referent, referent.lower()))
                else:
                    words.append("they")
            else:
                word = WORD_MAP.get(token.sign_id, token.sign_id.lower())
                if word:
                    words.append(word)
        return words

    def _apply_nms(self, text: str, sequence: GlossSequence) -> str:
        is_question = False
        is_negated = False

        for nms in sequence.nms_signals:
            if nms.signal_type in (NMSType.QUESTION_YES_NO, NMSType.QUESTION_WH):
                is_question = True
            elif nms.signal_type == NMSType.NEGATION:
                is_negated = True

        if is_negated:
            words = text.split()
            if len(words) >= 2:
                words.insert(1, "not")
            elif len(words) == 1:
                words = ["not", words[0]]
            text = " ".join(words)

        if is_question:
            text = text.rstrip(".")
            text += "?"

        return text

    def _aggregate_confidence(self, sequence: GlossSequence) -> float:
        if not sequence.tokens:
            return 0.0
        return sum(t.confidence for t in sequence.tokens) / len(sequence.tokens)
