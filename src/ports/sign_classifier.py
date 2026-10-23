from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.models import GlossCandidate, LandmarkWindow


class SignClassifier(ABC):
    @abstractmethod
    def classify(self, window: LandmarkWindow) -> list[GlossCandidate]: ...

    @abstractmethod
    def get_vocabulary_size(self) -> int: ...

    @abstractmethod
    def supports_continuous(self) -> bool: ...
