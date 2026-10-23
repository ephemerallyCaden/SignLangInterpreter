from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.core.models import LandmarkFrame


class LandmarkExtractor(ABC):
    @abstractmethod
    def extract(self, frame: np.ndarray) -> LandmarkFrame | None: ...

    @abstractmethod
    def get_latency_ms(self) -> float: ...
