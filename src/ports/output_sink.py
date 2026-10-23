from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.core.models import TranslationResult


class OutputSink(ABC):
    @abstractmethod
    def emit(self, result: TranslationResult, frame: np.ndarray | None = None) -> None: ...
