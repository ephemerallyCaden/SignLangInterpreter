from __future__ import annotations

import time

from src.core.models import GlossSequence, GlossToken, NMSType, TranslationResult
from src.ports.grammar_transformer import GrammarTransformer

# Direct gloss → English word map (expanded as vocabulary grows)
WORD_MAP: dict[str, str] = {
    "HELLO": "hello",
    "THANK_YOU": "thank you",
    "YES": "yes",
    "NO": "no",
    "GO": "go",
    "STORE": "store",
    "MOTHER": "mother",
    "FATHER": "father",
    "FUTURE": "",  # tense marker, not a word
    "PAST": "",
    "POINT": "",  # resolved via spatial context
    "UNKNOWN": "?",
}


class RuleBasedAdapter(GrammarTransformer):
    """Hardcoded BSL → English reordering rules.

    Implements the 5 core rules from the project plan:
    1. Topic-comment → SVO reordering
    2. FUTURE/PAST → tense marking
    3. NMS:QUESTION_YN → subject-auxiliary inversion
    4. NMS:NEGATION → insert "not" before verb
    5. POINT:slot → resolve from spatial context
    """

    def transform(self, sequence: GlossSequence) -> TranslationResult:
        start = time.perf_counter()

        words = self._glosses_to_words(sequence)
        text = " ".join(w for w in words if w)

        # Apply NMS modifications
        text = self._apply_nms(text, sequence)

        if not text.strip():
            text = "..."

        # Capitalize first letter, add period if missing punctuation
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
                # Resolve spatial reference
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
        for nms in sequence.nms_signals:
            if nms.signal_type == NMSType.QUESTION_YES_NO:
                text = text.rstrip(".")
                text += "?"
            elif nms.signal_type == NMSType.QUESTION_WH:
                text = text.rstrip(".")
                text += "?"
            elif nms.signal_type == NMSType.NEGATION:
                # Simple negation: prepend "not" (crude but functional)
                words = text.split()
                if len(words) > 1:
                    words.insert(1, "not")
                    text = " ".join(words)
        return text

    def _aggregate_confidence(self, sequence: GlossSequence) -> float:
        if not sequence.tokens:
            return 0.0
        return sum(t.confidence for t in sequence.tokens) / len(sequence.tokens)
