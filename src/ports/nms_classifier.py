from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.models import FaceLandmarks, NMSSignal


class NMSClassifier(ABC):
    @abstractmethod
    def classify(self, face_landmarks: FaceLandmarks, window_ms: int) -> list[NMSSignal]: ...
