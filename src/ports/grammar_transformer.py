from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.models import GlossSequence, TranslationResult


class GrammarTransformer(ABC):
    @abstractmethod
    def transform(
        self, sequence: GlossSequence, context: dict[str, Any] | None = None
    ) -> TranslationResult: ...
