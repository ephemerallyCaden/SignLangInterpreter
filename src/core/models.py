from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SpatialSlot(Enum):
    LEFT_FAR = "left_far"
    LEFT_NEAR = "left_near"
    CENTER = "center"
    RIGHT_NEAR = "right_near"
    RIGHT_FAR = "right_far"
    TOP = "top"


class DominantHand(Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class NMSType(Enum):
    QUESTION_YES_NO = "q_yn"
    QUESTION_WH = "q_wh"
    NEGATION = "neg"
    TOPIC_MARKER = "topic"
    INTENSIFIER = "intens"
    CONDITIONAL = "cond"


@dataclass(frozen=True)
class HandLandmarks:
    left: np.ndarray | None  # (21, 3) or None
    right: np.ndarray | None  # (21, 3) or None


@dataclass(frozen=True)
class FaceLandmarks:
    points: np.ndarray  # (478, 3) — 468 mesh + 10 iris landmarks


@dataclass(frozen=True)
class PoseLandmarks:
    points: np.ndarray  # (33, 3)


@dataclass(frozen=True)
class LandmarkFrame:
    timestamp_ms: int
    hands: HandLandmarks          # normalized to signing space
    face: FaceLandmarks           # normalized to signing space
    pose: PoseLandmarks           # normalized to signing space
    hand_presence: tuple[bool, bool]
    raw_hands: HandLandmarks | None = None  # raw [0,1] coords for display


@dataclass(frozen=True)
class LandmarkWindow:
    frames: list[LandmarkFrame]
    window_size_ms: int
    stride_ms: int


@dataclass(frozen=True)
class GlossToken:
    sign_id: str
    confidence: float
    temporal_span: tuple[int, int]
    dominant_hand: DominantHand
    spatial_slot: SpatialSlot | None = None
    is_establishing: bool = False
    is_referencing: bool = False
    movement_vector: np.ndarray | None = None


@dataclass(frozen=True)
class GlossCandidate:
    sign_id: str
    confidence: float
    temporal_span: tuple[int, int]
    spatial_slot: SpatialSlot | None = None
    is_transition: bool = False


@dataclass(frozen=True)
class NMSSignal:
    signal_type: NMSType
    confidence: float
    temporal_span: tuple[int, int]
    intensity: float


@dataclass(frozen=True)
class GlossSequence:
    tokens: list[GlossToken]
    nms_signals: list[NMSSignal]
    spatial_context: dict[SpatialSlot, str]
    sequence_duration_ms: int


@dataclass(frozen=True)
class TranslationResult:
    text: str
    confidence: float
    gloss_sequence: GlossSequence
    processing_time_ms: int
