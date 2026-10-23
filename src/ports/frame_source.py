from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def acquire_frame(self) -> np.ndarray | None: ...

    @abstractmethod
    def get_fps(self) -> float: ...

    @abstractmethod
    def close(self) -> None: ...
