from __future__ import annotations

import numpy as np

from src.core.models import TranslationResult
from src.ports.output_sink import OutputSink


class ConsoleOutputAdapter(OutputSink):
    def emit(self, result: TranslationResult, frame: np.ndarray | None = None) -> None:
        glosses = " ".join(f"[{t.sign_id}]" for t in result.gloss_sequence.tokens)
        print(f"[{result.confidence:.0%}] {result.text}  ({glosses})")
