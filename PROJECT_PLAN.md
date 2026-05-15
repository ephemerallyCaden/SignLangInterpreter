# SignLanguageInterpreter: Technical Project Plan

## Document Metadata

| Field | Value |
|-------|-------|
| Codename | SignLanguageInterpreter |
| Target Language | British Sign Language (BSL) |
| Architecture | Hexagonal (Ports & Adapters) Monolith |
| Deployment Target | Laptop (dev); Mobile NPU (production) |
| Privacy Model | Fully offline, zero network inference |
| Author | Solo Developer |
| Status | Pre-implementation Planning |

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Data Model & Intermediate Representations](#3-data-model--intermediate-representations)
4. [Module Specifications](#4-module-specifications)
5. [Critical Technical Challenges & Mitigations](#5-critical-technical-challenges--mitigations)
6. [Training & Data Strategy](#6-training--data-strategy)
7. [Phased Implementation Roadmap](#7-phased-implementation-roadmap)
8. [Performance Budgets & Constraints](#8-performance-budgets--constraints)
9. [Testing Strategy](#9-testing-strategy)
10. [Risk Register](#10-risk-register)

---

## 1. System Overview

### 1.1 Problem Statement

British Sign Language (BSL) is a complete, independent language with its own grammar, syntax, and spatial semantics. It is not a signed encoding of English. Current translation approaches fail because they either:

- Treat signing as a sequence of static image classifications (losing temporal and spatial semantics)
- Rely on cloud inference (introducing latency and privacy violations for sensitive communication)
- Map signs 1:1 to English words (ignoring BSL's topic-comment structure, spatial anaphora, and non-manual grammar)

### 1.2 Solution Architecture (High-Level)

SignLanguageInterpreter implements a three-stage pipeline:

```
Camera → [Vision Layer] → Landmarks → [Classifier Layer] → Enriched Glosses → [SLM Layer] → English Text
                                              ↕
                                    [NMS Classifier]
                                              ↕
                                    [Spatial Registry]
```

The system interprets BSL as a **Dynamic Biomechanical Stream** — a continuous, multi-channel signal comprising manual articulation, non-manual signals, and spatial reference — and translates it into natural English using a locally-executed Small Language Model as a neural grammar transformer.

### 1.3 Key Design Principles

| Principle | Implication |
|-----------|-------------|
| Privacy-first | All inference on-device. No telemetry. No cloud fallback. |
| Offline-capable | Full functionality with no network connection. |
| Linguistically informed | BSL grammar drives architecture, not English assumptions. |
| Phased viability | Each phase produces a usable (if limited) system. |
| Swap-without-rewire | Any component replaceable via adapter pattern. |
| Observable | Every intermediate representation inspectable for debugging and trust-building. |

---

## 2. Architecture Deep Dive

### 2.1 Pattern: Hexagonal Architecture (Ports & Adapters)

The application core contains domain logic that is agnostic to all external dependencies. External systems (camera hardware, ML frameworks, model weights, output targets) connect through well-defined ports implemented by swappable adapters.

```
                         ┌─────────────────────────────────────────────────────┐
                         │                 APPLICATION CORE                     │
                         │                                                     │
   ┌─────────────┐      │  ┌─────────────────────────────────────────────┐   │      ┌─────────────┐
   │   Camera     │◄────┼──┤  Port: FrameSource                          │   │      │   Screen    │
   │   Adapter    │     │  │    acquire_frame() → Frame                   │   │      │   Adapter   │
   └─────────────┘      │  └─────────────────────────────────────────────┘   │      └──────▲──────┘
                         │                                                     │             │
   ┌─────────────┐      │  ┌─────────────────────────────────────────────┐   │      ┌──────┴──────┐
   │  MediaPipe   │◄────┼──┤  Port: LandmarkExtractor                     │   │      │    Port:    │
   │   Adapter    │     │  │    extract(Frame) → LandmarkFrame            │   │      │   Output    │
   └─────────────┘      │  └─────────────────────────────────────────────┘   │      └─────────────┘
                         │                                                     │
   ┌─────────────┐      │  ┌─────────────────────────────────────────────┐   │
   │  Lookup      │◄────┼──┤  Port: SignClassifier                         │   │
   │  Table       │     │  │    classify(LandmarkWindow) → GlossCandidates│   │
   ├─────────────┤      │  └─────────────────────────────────────────────┘   │
   │  CTC/GRU    │      │                                                     │
   │  Adapter     │      │  ┌─────────────────────────────────────────────┐   │
   └─────────────┘      │  │  Port: NMSClassifier                         │   │
                         │  │    classify(FaceLandmarks) → NMSSignal       │   │
   ┌─────────────┐      │  └─────────────────────────────────────────────┘   │
   │  Face CNN    │◄────┼──┤                                                   │
   │  Adapter     │     │  │  ┌─────────────────────────────────────────────┐ │
   └─────────────┘      │  │  │  Port: GrammarTransformer                   │ │
                         │  │  │    transform(GlossSequence, Context) → Text │ │
   ┌─────────────┐      │  │  └─────────────────────────────────────────────┘ │
   │  Phi-3       │◄────┼──┤                                                   │
   │  Adapter     │     │  │                                                   │
   ├─────────────┤      │  │  ┌─────────────────────────────────────────────┐ │
   │  Rule-Based  │     │  │  │  Domain: TranslationOrchestrator            │ │
   │  Adapter     │     │  │  │    - Pipeline coordination                  │ │
   └─────────────┘      │  │  │    - Spatial context management             │ │
                         │  │  │    - Hybrid routing (lookup vs. neural)     │ │
                         │  │  │    - NMS ↔ Gloss temporal alignment         │ │
                         │  │  │    - Confidence thresholding                │ │
                         │  │  └─────────────────────────────────────────────┘ │
                         │  └─────────────────────────────────────────────────┘ │
                         └─────────────────────────────────────────────────────┘
```

### 2.2 Rationale for Hexagonal over Alternatives

| Criterion | Microservice Pipeline | Event-Driven Stream | Hexagonal Monolith |
|-----------|:---:|:---:|:---:|
| Solo-dev operational complexity | High | Medium | **Low** |
| IPC latency at 30-60 FPS | ~2-5ms overhead | ~1ms (async) | **~0ms (in-process)** |
| Component swappability | High | Medium | **High** |
| Testability (mock injection) | High | Medium | **High** |
| Phased delivery (additive) | Medium | Medium | **High** |
| Mobile portability | Low | Low | **High** |
| Shared state (spatial context) | Difficult | Native (event store) | **Native (domain object)** |
| Frame backpressure handling | Manual | Native | **Bounded queue** |

### 2.3 Concurrency Model

Despite being a single process, the pipeline requires concurrent execution to avoid blocking:

```
[Frame Producer Thread]
         │
         ▼
   BoundedQueue(maxsize=3)  ← drops oldest frame if full (backpressure)
         │
         ▼
[Processing Thread]
   ├── LandmarkExtractor.extract()          ~15ms
   ├── Parallel:
   │     ├── SignClassifier.classify()       ~20ms
   │     └── NMSClassifier.classify()       ~5ms
   ├── Orchestrator.merge_and_route()        ~1ms
   └── GrammarTransformer.transform()        ~80ms (SLM inference)
         │
         ▼
   OutputQueue → [Render Thread]
```

**Backpressure policy:** If the processing thread cannot keep up with frame rate, the bounded queue drops the oldest unprocessed frame. The system prioritises recency over completeness — in real-time interpretation, a current-but-incomplete translation is more useful than a delayed-but-complete one.

### 2.4 Directory Structure

```
SignLanguageInterpreter/
├── src/
│   ├── core/                          # Domain logic (zero external dependencies)
│   │   ├── __init__.py
│   │   ├── orchestrator.py            # TranslationOrchestrator
│   │   ├── models.py                  # Domain data classes (GlossToken, LandmarkFrame, etc.)
│   │   ├── spatial_registry.py        # Spatial anaphora tracking
│   │   ├── temporal_aligner.py        # NMS ↔ Gloss alignment
│   │   └── confidence_router.py       # Hybrid routing logic
│   │
│   ├── ports/                         # Abstract interfaces (Port definitions)
│   │   ├── __init__.py
│   │   ├── frame_source.py            # ABC: FrameSource
│   │   ├── landmark_extractor.py      # ABC: LandmarkExtractor
│   │   ├── sign_classifier.py         # ABC: SignClassifier
│   │   ├── nms_classifier.py          # ABC: NMSClassifier
│   │   ├── grammar_transformer.py     # ABC: GrammarTransformer
│   │   └── output_sink.py            # ABC: OutputSink
│   │
│   ├── adapters/                      # Concrete implementations
│   │   ├── __init__.py
│   │   ├── camera_opencv.py           # FrameSource → OpenCV webcam
│   │   ├── camera_video_file.py       # FrameSource → Video file (testing/training)
│   │   ├── landmark_mediapipe.py      # LandmarkExtractor → MediaPipe Holistic
│   │   ├── classifier_lookup.py       # SignClassifier → Static lookup table (Phase 1)
│   │   ├── classifier_ctc.py          # SignClassifier → CTC neural (Phase 2)
│   │   ├── nms_face_cnn.py            # NMSClassifier → Lightweight face CNN
│   │   ├── transformer_rules.py       # GrammarTransformer → Rule-based reordering (Phase 1)
│   │   ├── transformer_slm.py         # GrammarTransformer → Phi-3/Mistral (Phase 2)
│   │   ├── output_console.py          # OutputSink → Terminal text
│   │   └── output_overlay.py          # OutputSink → Visual overlay on video
│   │
│   ├── infrastructure/                # Wiring, config, lifecycle
│   │   ├── __init__.py
│   │   ├── container.py               # Dependency injection container
│   │   ├── config.py                  # Runtime configuration
│   │   └── pipeline.py               # Thread management, queue setup
│   │
│   └── main.py                        # Entry point
│
├── models/                            # Model weights & artifacts (gitignored)
│   ├── mediapipe/
│   ├── sign_classifier/
│   ├── nms_classifier/
│   └── slm/
│
├── data/                              # Training data & processed datasets
│   ├── raw/                           # Original BSL-1K / BOBSL clips
│   ├── processed/                     # Extracted landmark sequences
│   ├── gloss_lexicon/                 # Sign ID → label mappings
│   └── synthetic_gloss/              # SLM-generated gloss annotations
│
├── training/                          # Training scripts (separate from inference)
│   ├── train_classifier.py
│   ├── train_nms.py
│   ├── finetune_slm.py
│   └── generate_synthetic_gloss.py
│
├── tests/
│   ├── unit/                          # Core logic tests (mocked ports)
│   ├── integration/                   # Adapter tests (real dependencies)
│   └── fixtures/                     # Saved landmark sequences for replay
│
├── configs/
│   ├── phase1.yaml                    # Lookup + Rules configuration
│   ├── phase2.yaml                    # CTC + SLM configuration
│   └── benchmark.yaml                # Performance profiling config
│
├── PROJECT_PLAN.md                    # This document
├── pyproject.toml
├── .gitignore
└── Makefile
```

---

## 3. Data Model & Intermediate Representations

### 3.1 Core Domain Objects

```python
@dataclass(frozen=True)
class LandmarkFrame:
    """Single frame of extracted body landmarks, normalized to signing space."""
    timestamp_ms: int
    hands: HandLandmarks          # 21 landmarks × 2 hands × 3 coords = 126 floats
    face: FaceLandmarks           # 468 landmarks × 3 coords = 1404 floats
    pose: PoseLandmarks           # 33 landmarks × 3 coords = 99 floats
    hand_presence: tuple[bool, bool]  # (left_detected, right_detected)

@dataclass(frozen=True)
class HandLandmarks:
    """Normalized 3D hand coordinates. Origin at wrist, scale-invariant."""
    left: np.ndarray   # shape: (21, 3) or None
    right: np.ndarray  # shape: (21, 3) or None

@dataclass(frozen=True)
class LandmarkWindow:
    """Sliding temporal window of landmark frames for classification."""
    frames: list[LandmarkFrame]    # Ordered by timestamp
    window_size_ms: int            # Typically 500-2000ms
    stride_ms: int                 # Overlap between consecutive windows

@dataclass
class SpatialSlot(Enum):
    """Discretized signing space zones for spatial anaphora."""
    LEFT_FAR = "left_far"
    LEFT_NEAR = "left_near"
    CENTER = "center"
    RIGHT_NEAR = "right_near"
    RIGHT_FAR = "right_far"
    TOP = "top"

@dataclass(frozen=True)
class GlossToken:
    """
    Enriched gloss representation preserving spatial and temporal semantics.
    This is the critical intermediate representation between classifier and SLM.
    """
    sign_id: str                        # e.g., "STORE", "GO", "POINT"
    confidence: float                   # 0.0 - 1.0
    temporal_span: tuple[int, int]      # (start_ms, end_ms)
    dominant_hand: str                  # "left" | "right" | "both"
    spatial_slot: SpatialSlot | None    # Where in signing space this was articulated
    is_establishing: bool               # True if this sign is PLACING a referent
    is_referencing: bool                # True if this is a point/reference to established slot
    movement_vector: np.ndarray | None  # For directional verbs (e.g., GIVE-to-you vs GIVE-to-me)

@dataclass(frozen=True)
class NMSSignal:
    """Non-manual signal detected from facial/body landmarks."""
    signal_type: NMSType               # QUESTION_YN, QUESTION_WH, NEGATION, TOPIC, INTENSIFIER
    confidence: float
    temporal_span: tuple[int, int]     # Duration this NMS is active
    intensity: float                    # 0.0 - 1.0 (e.g., how raised the eyebrows are)

class NMSType(Enum):
    QUESTION_YES_NO = "q_yn"           # Brow raise + head forward
    QUESTION_WH = "q_wh"              # Brow furrow + head tilt
    NEGATION = "neg"                   # Head shake
    TOPIC_MARKER = "topic"             # Brow raise + head back + pause
    INTENSIFIER = "intens"             # Puffed cheeks / pursed lips
    CONDITIONAL = "cond"               # Brow raise + head tilt (if-clause)

@dataclass
class GlossSequence:
    """
    Complete enriched sequence ready for SLM consumption.
    Merges manual glosses with non-manual signals and spatial context.
    """
    tokens: list[GlossToken]
    nms_signals: list[NMSSignal]
    spatial_context: dict[SpatialSlot, str]  # Slot → established referent sign_id
    sequence_duration_ms: int

@dataclass(frozen=True)
class TranslationResult:
    """Final output of the pipeline."""
    text: str                           # Natural English translation
    confidence: float                   # Aggregate confidence
    gloss_sequence: GlossSequence       # For debugging / visual feedback
    processing_time_ms: int
```

### 3.2 Serialization Format for SLM Input

The GlossSequence must be serialized into a text format the SLM can consume. This format is the contract between the classifier layer and the grammar transformer:

```
<spatial_context>
  LEFT_NEAR=MOTHER | RIGHT_NEAR=STORE
</spatial_context>
<gloss_stream>
  [MOTHER]{slot:LEFT_NEAR, establishing:true}
  [STORE]{slot:RIGHT_NEAR, establishing:true}
  [POINT:LEFT_NEAR]{referencing:true}
  [GO]{direction:RIGHT_NEAR}
  [FUTURE]
</gloss_stream>
<nms>
  [Q:YES_NO]{span:all}
</nms>
<instruction>
  Translate the above BSL gloss sequence to natural English.
  Resolve spatial references using the spatial context.
  Apply the sentence type indicated by NMS.
</instruction>
```

**Expected SLM output:** `"Is your mother going to the store later?"`

---

## 4. Module Specifications

### 4.1 Port: FrameSource

```python
class FrameSource(ABC):
    """Acquires raw video frames from an input device or file."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def acquire_frame(self) -> Frame | None:
        """Returns next frame or None if source exhausted/unavailable."""
        ...

    @abstractmethod
    def get_fps(self) -> float: ...

    @abstractmethod
    def close(self) -> None: ...
```

**Adapters:**
- `CameraOpenCVAdapter`: Live webcam via OpenCV VideoCapture
- `VideoFileAdapter`: Pre-recorded .mp4/.avi for testing and training pipeline validation

### 4.2 Port: LandmarkExtractor

```python
class LandmarkExtractor(ABC):
    """Extracts normalized body landmarks from a video frame."""

    @abstractmethod
    def extract(self, frame: Frame) -> LandmarkFrame:
        """
        Extracts 542 landmarks (hands: 42, face: 468, pose: 33) and normalizes
        to signing-space coordinates (origin: sternum, scale: shoulder width).
        """
        ...

    @abstractmethod
    def get_latency_ms(self) -> float:
        """Returns average extraction latency for profiling."""
        ...
```

**Normalization requirements:**
- Origin: Midpoint between shoulders (translation invariance)
- Scale: Normalized by shoulder width (scale invariance)
- Rotation: Aligned to torso plane (rotation invariance around vertical axis)
- Handedness: BSL signer handedness must be detected or configured; dominant hand landmarks always mapped to consistent array position

**Adapters:**
- `MediaPipeHolisticAdapter`: Google MediaPipe Holistic solution (542 landmarks, ~15ms on CPU)

### 4.3 Port: SignClassifier

```python
class SignClassifier(ABC):
    """Classifies landmark sequences into discrete sign glosses."""

    @abstractmethod
    def classify(self, window: LandmarkWindow) -> list[GlossCandidate]:
        """
        Returns ranked list of gloss candidates with confidence scores.
        May return multiple candidates for a single window (coarticulation).
        """
        ...

    @abstractmethod
    def get_vocabulary_size(self) -> int: ...

    @abstractmethod
    def supports_continuous(self) -> bool:
        """Whether this adapter handles coarticulated/continuous signing."""
        ...
```

```python
@dataclass
class GlossCandidate:
    sign_id: str
    confidence: float
    temporal_span: tuple[int, int]
    spatial_slot: SpatialSlot | None
    is_transition: bool  # True if classified as movement epenthesis
```

**Adapters:**

| Adapter | Phase | Approach | Continuous? |
|---------|-------|----------|-------------|
| `LookupTableAdapter` | 1 | KNN/DTW on pre-computed landmark templates | No |
| `CTCClassifierAdapter` | 2 | GRU encoder + CTC decoder | Yes |

### 4.4 Port: NMSClassifier

```python
class NMSClassifier(ABC):
    """Classifies non-manual signals from facial landmarks."""

    @abstractmethod
    def classify(self, face_landmarks: FaceLandmarks, window_ms: int) -> list[NMSSignal]:
        """
        Detects active non-manual signals over a temporal window.
        May return multiple simultaneous signals (e.g., NEGATION + QUESTION).
        """
        ...
```

**Implementation notes:**
- Input: 468-point face mesh subset (brows: points 65-70, 295-300; mouth: 61-67, 291-297; head pose: 1, 4, 5, 6, 197, 195)
- Architecture: Lightweight 1D-CNN over temporal face landmark deltas
- BSL-specific feature engineering:
  - Brow raise magnitude: `mean(brow_y) - mean(eye_y)` normalized
  - Head forward lean: nose tip z-displacement relative to ears
  - Head shake: lateral oscillation frequency of nose tip x-coordinate

### 4.5 Port: GrammarTransformer

```python
class GrammarTransformer(ABC):
    """Transforms enriched gloss sequences into natural English text."""

    @abstractmethod
    def transform(self, sequence: GlossSequence, context: ConversationContext) -> TranslationResult:
        """
        Performs BSL gloss → English translation with:
        - Grammatical reordering (BSL topic-comment → English SVO)
        - Spatial reference resolution
        - NMS integration (sentence type, negation)
        - Directional verb interpretation
        """
        ...
```

**Adapters:**

| Adapter | Phase | Approach | Latency |
|---------|-------|----------|---------|
| `RuleBasedAdapter` | 1 | Hardcoded BSL→English reordering rules + template filling | <5ms |
| `SLMAdapter` | 2 | Phi-3-Mini (4-bit quantized) or Mistral-7B (4-bit) via llama.cpp | ~80-150ms |

**Rule-Based Adapter (Phase 1) — Core Rules:**
1. BSL topic-comment → English SVO reordering
2. `[FUTURE]` / `[PAST]` → English tense marking on verb
3. `[NMS:QUESTION_YN]` → invert subject-auxiliary
4. `[NMS:NEGATION]` → insert "not" / "don't" before verb
5. `[POINT:slot]` → resolve from SpatialRegistry → pronoun

### 4.6 Domain: TranslationOrchestrator

```python
class TranslationOrchestrator:
    """
    Central domain coordinator. Manages pipeline flow, state, and routing.
    Contains NO external dependencies — only ports and domain objects.
    """

    def __init__(
        self,
        sign_classifier: SignClassifier,
        nms_classifier: NMSClassifier,
        grammar_transformer: GrammarTransformer,
        spatial_registry: SpatialRegistry,
        confidence_router: ConfidenceRouter,
        temporal_aligner: TemporalAligner,
    ): ...

    def process_window(self, window: LandmarkWindow) -> TranslationResult | None:
        """
        Full pipeline execution for one landmark window.
        Returns None if insufficient confidence for translation.
        """
        # 1. Classify manual signs
        gloss_candidates = self.sign_classifier.classify(window)

        # 2. Classify non-manual signals (parallel in production)
        face_landmarks = self._extract_face(window)
        nms_signals = self.nms_classifier.classify(face_landmarks, window.window_size_ms)

        # 3. Filter transitions, apply confidence threshold
        glosses = self.confidence_router.filter_and_route(gloss_candidates)

        # 4. Update spatial registry
        self.spatial_registry.update(glosses)

        # 5. Align NMS with gloss timeline
        aligned_sequence = self.temporal_aligner.align(glosses, nms_signals)

        # 6. Build enriched sequence with spatial context
        sequence = GlossSequence(
            tokens=aligned_sequence.tokens,
            nms_signals=aligned_sequence.nms,
            spatial_context=self.spatial_registry.get_active_context(),
            sequence_duration_ms=window.window_size_ms,
        )

        # 7. Grammar transformation
        if self.confidence_router.should_invoke_slm(sequence):
            return self.grammar_transformer.transform(sequence, self._conversation_context)
        else:
            return None  # Accumulate more context before translating
```

### 4.7 Domain: SpatialRegistry

```python
class SpatialRegistry:
    """
    Tracks spatial anaphora — which referents have been 'placed' where in signing space.
    Implements a decaying memory: older placements lose confidence over time.
    """

    def __init__(self, decay_rate: float = 0.95, max_slots: int = 6):
        self._registry: dict[SpatialSlot, SpatialReferent] = {}
        self._decay_rate = decay_rate

    def update(self, glosses: list[GlossToken]) -> None:
        """Process new glosses: establish new referents, decay existing ones."""
        self._decay_all()
        for gloss in glosses:
            if gloss.is_establishing and gloss.spatial_slot:
                self._registry[gloss.spatial_slot] = SpatialReferent(
                    sign_id=gloss.sign_id,
                    established_at=gloss.temporal_span[0],
                    confidence=gloss.confidence,
                )

    def resolve(self, slot: SpatialSlot) -> str | None:
        """Resolve a spatial reference to its established sign_id."""
        referent = self._registry.get(slot)
        if referent and referent.confidence > 0.3:
            return referent.sign_id
        return None

    def get_active_context(self) -> dict[SpatialSlot, str]:
        """Returns all active spatial assignments for SLM context injection."""
        return {
            slot: ref.sign_id
            for slot, ref in self._registry.items()
            if ref.confidence > 0.3
        }

    def _decay_all(self) -> None:
        for ref in self._registry.values():
            ref.confidence *= self._decay_rate
```

### 4.8 Domain: ConfidenceRouter

```python
class ConfidenceRouter:
    """
    Implements hybrid routing logic:
    - High-confidence isolated signs → direct lookup translation
    - Accumulated gloss buffer → SLM invocation for grammar transformation
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.92,
        slm_trigger_threshold: int = 3,        # Min glosses before invoking SLM
        slm_silence_trigger_ms: int = 800,     # Invoke SLM after 800ms pause
    ): ...

    def filter_and_route(self, candidates: list[GlossCandidate]) -> list[GlossToken]:
        """Filter transition frames, apply thresholds, buffer low-confidence."""
        ...

    def should_invoke_slm(self, sequence: GlossSequence) -> bool:
        """
        Determines if enough context has accumulated to warrant SLM invocation.
        Triggers on:
          - Buffer reaches slm_trigger_threshold tokens
          - Silence detected (no new signs for slm_silence_trigger_ms)
          - NMS sentence boundary detected (e.g., topic marker followed by comment)
        """
        ...
```

---

## 5. Critical Technical Challenges & Mitigations

### 5.1 Coarticulation (The "Cursive" Problem)

**Challenge:** In fluent BSL, signs do not have discrete boundaries. The articulatory end of one sign is modified by the starting configuration of the next (anticipatory coarticulation). A classifier trained on isolated sign clips will fail on continuous streams because it never sees transition movements.

**Mitigation: Multi-Strategy Approach**

**Strategy A — CTC Loss Function (Phase 2)**

Connectionist Temporal Classification allows the model to learn alignment between an unsegmented input sequence and a target label sequence without requiring explicit segmentation boundaries.

```
Input:  [frame_1, frame_2, ..., frame_T]     (landmark sequences)
Output: [BLANK, STORE, STORE, BLANK, GO, GO, GO, BLANK, ...]   (CTC alignment)
                    ↓ CTC collapse
        [STORE, GO]
```

The CTC blank token naturally absorbs transition frames, eliminating the need for explicit segmentation.

**Strategy B — Movement Epenthesis Detection (Phase 1.5)**

Transition movements between signs exhibit measurable kinematic properties distinct from within-sign movements:

| Property | Within-Sign | Transition (Epenthesis) |
|----------|-------------|------------------------|
| Path shape | Curved, deliberate | Straight-line (ballistic) |
| Velocity profile | Bell-shaped (smooth) | Asymmetric (fast start, slow end) |
| Hand configuration | Stable or systematically changing | Rapidly restructuring |
| Wrist acceleration | Moderate | High peak |

A binary classifier on these features (input: velocity + acceleration vectors over 5-frame micro-window) can flag transition frames for exclusion before main classification.

**Strategy C — Training Curriculum**

| Phase | Training Data | Coarticulation Exposure |
|-------|--------------|------------------------|
| 1 | BSL-1K isolated clips | None — accepts limitation |
| 1.5 | Self-recorded sign pairs/triplets | Bounded transitions |
| 2 | BOBSL continuous segments | Full coarticulation |

### 5.2 Loss of Spatial Anaphora

**Challenge:** BSL signers establish referents at locations in 3D space. Subsequent pointing or eye-gaze to those locations constitutes pronominal reference. A flat gloss string `[POINT]` loses the critical information of *where* the point was directed.

**Mitigation: Enriched Gloss Tokens + SpatialRegistry**

The architecture addresses this at three levels:

1. **Detection (Classifier layer):** The SignClassifier detects spatial slot assignment using the wrist position at sign termination relative to the normalized signing space. Six discretized zones provide sufficient resolution for BSL spatial grammar.

2. **Tracking (Domain layer):** The `SpatialRegistry` maintains a decaying memory of what's been established where. Decay prevents stale references from persisting indefinitely (BSL discourse typically refreshes spatial assignments every 30-60 seconds).

3. **Resolution (SLM layer):** The SLM receives explicit spatial context as part of its input prompt. It resolves `[POINT:LEFT_NEAR]` to the appropriate English pronoun using the provided context map.

**Key design decision:** The classifier does NOT resolve references. It only reports spatial geometry. Resolution is the SLM's job — this is where its language understanding capability is most valuable.

**Spatial discretization scheme:**

```
                    [TOP]
                      │
    [LEFT_FAR] [LEFT_NEAR] [CENTER] [RIGHT_NEAR] [RIGHT_FAR]
         │          │          │          │            │
       -0.8       -0.4       0.0        0.4          0.8
                  (normalized x-coordinate)
```

Thresholds are calibrated relative to shoulder width. The signing space is approximately ±1.0 shoulder-widths horizontally and ±0.6 vertically from the sternum.

### 5.3 Non-Manual Signal Omission

**Challenge:** BSL encodes critical grammatical information through non-manual signals (facial expressions, head movements, body shifts). Without these, sentence type is ambiguous and negation may be invisible.

**Mitigation: Parallel NMS Classification Channel**

**Architecture:** A dedicated `NMSClassifier` port runs in parallel with the `SignClassifier`, consuming only the face landmark subset. This is a separate model, not an additional output head on the sign classifier, because:

1. NMS operate on different timescales (suprasegmental — spanning multiple signs)
2. NMS detection requires temporal analysis of facial landmarks only (smaller, faster model)
3. Independent failure modes — a missed NMS degrades output but doesn't corrupt sign detection

**BSL NMS Feature Engineering:**

| NMS Type | Primary Features | Detection Method |
|----------|-----------------|------------------|
| Yes/No Question | Inner brow raise magnitude, head forward displacement | Threshold on brow-eye distance + nose z-delta |
| WH-Question | Brow furrow (corrugator), head tilt angle | Brow convergence metric + head roll |
| Negation | Head shake frequency (2-4 Hz oscillation) | FFT on nose-tip x-coordinate, peak at 2-4 Hz |
| Topic Marker | Brow raise + head back + temporal pause | Combined brow metric + head pitch + silence detection |
| Conditional | Brow raise + slight head tilt + body lean | Multi-feature threshold |
| Intensifier | Puffed cheeks (mouth landmark expansion), pursed lips | Mouth landmark area ratio |

**Temporal Alignment:**

NMS signals are suprasegmental — they span clauses, not individual signs. The `TemporalAligner` assigns each NMS signal to the gloss tokens it temporally overlaps:

```python
class TemporalAligner:
    def align(self, glosses: list[GlossToken], nms_signals: list[NMSSignal]) -> AlignedSequence:
        """
        Assigns NMS signals to gloss tokens based on temporal overlap.
        An NMS signal applies to ALL gloss tokens within its temporal span.
        """
        for nms in nms_signals:
            overlapping_glosses = [
                g for g in glosses
                if self._overlaps(g.temporal_span, nms.temporal_span)
            ]
            # NMS applies to the entire overlapping clause
            ...
```

### 5.4 The "Gloss" Bottleneck

**Challenge:** There is no universal BSL gloss annotation standard. Creating gloss-annotated training data requires expert BSL linguists, making large-scale supervised training impractical for a solo developer.

**Mitigation: Multi-Strategy Data Pipeline**

**Strategy A — Phase 1 Sidestep**

For isolated sign recognition, there is no gloss annotation problem. BSL-1K provides isolated clips with single-word labels. The "gloss" is simply the sign name. This is sufficient for Phase 1's lookup-based system.

**Strategy B — Synthetic Gloss Generation (Phase 2 Bootstrap)**

Use the SLM itself to generate approximate BSL gloss orderings from English subtitles:

```
Pipeline:
1. Input:  BOBSL subtitle "Is your mother going to the store?"
2. Prompt: "Convert to BSL gloss order. BSL uses topic-comment structure,
            time markers before verbs, questions indicated by NMS not word order."
3. Output: [MOTHER] [STORE] [GO] [FUTURE] [NMS:Q_YN]
4. Validation: Cross-reference temporal alignment with video landmarks
```

**Accuracy estimate:** ~60-70% correct ordering. Sufficient as weak supervision for training, not for evaluation.

**Strategy C — Two-Tower Contrastive Learning (Phase 3)**

Bypass explicit gloss entirely for the continuous signing case:

```
Tower 1 (Visual):   LandmarkSequence → SignEmbedding (via trained encoder)
Tower 2 (Text):     EnglishText → TextEmbedding (via SLM encoder)

Training objective: Contrastive loss — pull (video, subtitle) pairs together,
                    push non-matching pairs apart.

Inference: SignEmbedding → nearest TextEmbedding → decode to English
```

This learns an implicit mapping without requiring gloss as an intermediate representation. The explicit gloss pipeline remains available for interpretability and debugging.

**Strategy D — Community Bootstrapping (Long-term)**

If the project gains traction, the visual feedback overlay (showing detected glosses) can be verified by BSL users. Correct/incorrect feedback creates a supervised signal for fine-tuning.

---

## 6. Training & Data Strategy

### 6.1 Dataset Pipeline

| Dataset | Size | Use | Access |
|---------|------|-----|--------|
| BSL-1K | ~1,000 sign classes | Phase 1 isolated sign classifier | Academic application (Oxford VGG) |
| BOBSL | ~1,400 hours | Phase 2 continuous signing, SLM fine-tuning | Academic application (Oxford VGG) |
| BSL Corpus (UCL DCAL) | ~250 hours | Gold-standard gloss validation | Academic application (UCL) |
| SignBank BSL | Dictionary-scale | Linguistic validation, gloss lexicon | Open access |
| Self-recorded | ~50 hours (target) | Domain-specific signs, transition pairs | Self-generated |

### 6.2 Data Processing Pipeline

```
Raw Video → MediaPipe Extraction → Normalized Landmarks → Windowed Sequences → Training Format

Specifics:
- Frame rate: Standardized to 30 FPS (downsample 60fps sources)
- Window size: 64 frames (2.13s) with 50% overlap for training
- Normalization: Shoulder-width scaling, sternum-origin, torso-aligned
- Augmentation:
    - Temporal jitter: ±3 frames random offset
    - Spatial noise: Gaussian σ=0.01 on landmark positions
    - Speed variation: 0.8x - 1.2x playback speed (resample landmarks)
    - Mirror: Horizontal flip (with hand label swap) — but CAUTION: some BSL signs
      are hand-specific and should NOT be mirrored
```

### 6.3 Model Training Specifications

**Sign Classifier (Phase 2 — CTC):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Architecture | 2-layer GRU (256 hidden) + CTC head | Temporal modelling with tractable size |
| Input | 126-dim hand landmarks per frame (flattened 21×3×2) | Hands carry primary manual information |
| Output | vocab_size + 1 (CTC blank) | Per-frame probability distribution |
| Loss | CTC loss | Handles unsegmented sequences |
| Training data | BSL-1K (isolated → padded sequences) → BOBSL (continuous) |
| Batch size | 32 |
| Learning rate | 1e-3 (Adam) with cosine annealing |
| Expected training time | ~8 hours on laptop GPU (BSL-1K), ~48 hours (BOBSL subset) |

**NMS Classifier:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Architecture | 1D-CNN (3 layers, 64-128-64 filters) + sigmoid heads | Multi-label: multiple NMS can co-occur |
| Input | Engineered features: brow metrics, head pose angles, mouth ratios (8-dim) per frame | Reduces dimensionality from 1404 raw face landmarks |
| Output | 6 NMS types × confidence | Multi-label binary classification |
| Window | 30 frames (1 second) | NMS are sustained signals, need temporal context |
| Training data | Semi-supervised: BSL Corpus (ELAN annotations) + rule-based pseudo-labels from BOBSL |

**SLM Fine-tuning (Phase 2 — LoRA):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Phi-3-Mini (3.8B) or Mistral-7B | Best size/quality tradeoff for on-device |
| Fine-tuning method | LoRA (rank 16, alpha 32) | Trainable params: ~10M vs 3.8B total |
| Quantization | 4-bit (GPTQ or AWQ) post-training | Enables laptop/mobile inference |
| Training data | Synthetic gloss → English pairs (10K-50K examples) | Generated from BOBSL subtitles |
| Task format | Structured prompt (§3.2) → natural English completion |
| Evaluation | BLEU-4 + human eval on held-out BSL Corpus translations |

---

## 7. Phased Implementation Roadmap

### Phase 1: Foundation & Isolated Sign Recognition

**Goal:** Working end-to-end pipeline with isolated sign lookup. Validates architecture.

**Deliverables:**
- [ ] Project scaffolding (Hexagonal structure, DI container, config system)
- [ ] `FrameSource` port + `CameraOpenCVAdapter` + `VideoFileAdapter`
- [ ] `LandmarkExtractor` port + `MediaPipeHolisticAdapter` with normalization
- [ ] `SignClassifier` port + `LookupTableAdapter` (KNN/DTW on 100 BSL signs)
- [ ] `GrammarTransformer` port + `RuleBasedAdapter` (basic BSL→English reordering)
- [ ] `OutputSink` port + `ConsoleAdapter` + `OverlayAdapter` (skeleton visualization)
- [ ] `TranslationOrchestrator` (simplified — no NMS, no spatial)
- [ ] Unit tests with fixture landmark sequences
- [ ] Data pipeline: BSL-1K → extracted landmarks → training format

**Acceptance criteria:**
- Recognizes 100 isolated BSL signs at >85% accuracy (controlled conditions)
- End-to-end latency (frame → text) < 200ms
- Skeleton overlay renders at 30 FPS

**Estimated scope:** 4-6 weeks

---

### Phase 1.5: NMS & Spatial Awareness

**Goal:** Add non-manual signal detection and spatial tracking. Still isolated signs but grammatically aware output.

**Deliverables:**
- [ ] `NMSClassifier` port + `FaceCNNAdapter`
- [ ] `SpatialRegistry` implementation
- [ ] `TemporalAligner` implementation
- [ ] `ConfidenceRouter` (hybrid routing logic)
- [ ] NMS training: semi-supervised on BSL Corpus annotations
- [ ] Extend sign vocabulary to 250 signs
- [ ] Integration tests: NMS correctly modifies translation output

**Acceptance criteria:**
- Correctly identifies question type (YN vs. WH) at >80% accuracy
- Spatial references resolve correctly for 2-referent scenarios
- "Are you going?" vs. "You are going" distinguished by brow detection

**Estimated scope:** 3-4 weeks

---

### Phase 2: Continuous Signing & SLM Integration

**Goal:** Handle fluent, coarticulated BSL via neural classifier and SLM grammar transformer.

**Deliverables:**
- [ ] `CTCClassifierAdapter` — GRU + CTC trained on continuous BSL sequences
- [ ] `SLMAdapter` — Phi-3-Mini (4-bit) via llama.cpp Python bindings
- [ ] LoRA fine-tuning pipeline for gloss→English task
- [ ] Synthetic gloss generation pipeline (BOBSL subtitles → approximate gloss)
- [ ] Coarticulation handling (CTC blank absorption + epenthesis detection)
- [ ] Sign vocabulary expansion to 500+ signs
- [ ] Performance optimization: pipeline profiling, bottleneck elimination

**Acceptance criteria:**
- Handles continuous 3-5 sign sentences at >70% semantic accuracy
- SLM inference (time to first token) < 150ms on laptop
- Graceful degradation: drops frames rather than accumulating latency

**Estimated scope:** 6-8 weeks

---

### Phase 3: Optimization & Robustness

**Goal:** Production-quality system with robust performance across conditions.

**Deliverables:**
- [ ] Model distillation: compress CTC classifier for mobile-viable size
- [ ] Two-tower contrastive training (bypass explicit gloss for complex sentences)
- [ ] Signer adaptation: few-shot calibration for new users
- [ ] Robustness: varied lighting, backgrounds, signing speeds
- [ ] Domain expansion: multiple vocabulary domains (medical, legal, everyday)
- [ ] Mobile deployment investigation: CoreML / ONNX Runtime / TFLite conversion

**Acceptance criteria:**
- >75% semantic accuracy on unseen signers
- Functional in varied indoor lighting conditions
- Total pipeline latency < 300ms on target mobile hardware

**Estimated scope:** 8-12 weeks

---

## 8. Performance Budgets & Constraints

### 8.1 Latency Budget (Per Translation Cycle)

| Stage | Target (Laptop) | Target (Mobile, Phase 3) | Notes |
|-------|:---:|:---:|-------|
| Frame acquisition | <5ms | <5ms | Camera driver dependent |
| Landmark extraction | <20ms | <30ms | MediaPipe CPU/GPU |
| Sign classification | <25ms | <40ms | GRU inference |
| NMS classification | <5ms | <10ms | Small CNN, parallel with above |
| Temporal alignment + routing | <2ms | <2ms | Pure computation |
| SLM inference (TTFT) | <150ms | <200ms | 4-bit quantized, NPU |
| **Total (worst case)** | **<207ms** | **<287ms** | |
| **Total (typical, hybrid lookup)** | **<55ms** | **<90ms** | When SLM not invoked |

### 8.2 Memory Budget

| Component | Laptop | Mobile Target |
|-----------|--------|---------------|
| MediaPipe Holistic | ~50 MB | ~50 MB |
| Sign Classifier (GRU) | ~15 MB | ~8 MB (distilled) |
| NMS Classifier | ~5 MB | ~3 MB |
| SLM (4-bit quantized) | ~2.2 GB (Phi-3) | ~2.2 GB |
| Runtime overhead | ~200 MB | ~100 MB |
| **Total** | **~2.5 GB** | **~2.4 GB** |

### 8.3 Throughput Target

- Minimum viable: Process every 3rd frame (10 FPS effective analysis)
- Target: Process every frame at 30 FPS for landmark extraction; classifier operates on sliding windows (not per-frame)
- SLM invocation: Triggered by clause boundary detection, not per-frame (~1-3 invocations per sentence)

---

## 9. Testing Strategy

### 9.1 Test Pyramid

```
              ┌──────────────┐
              │  End-to-End  │   Video → English text (5-10 scenarios)
              │   (Manual)   │
              └──────┬───────┘
                     │
           ┌─────────┴─────────┐
           │    Integration     │   Adapter + real model inference (per adapter)
           └─────────┬─────────┘
                     │
     ┌───────────────┴───────────────┐
     │          Unit Tests            │   Core domain logic (mocked ports)
     └───────────────────────────────┘
```

### 9.2 Unit Tests (Core Domain)

All domain logic tested with mocked ports. No ML models, no cameras, no external dependencies.

| Component | Test Cases |
|-----------|-----------|
| `TranslationOrchestrator` | Correct pipeline sequencing; handles empty classifier output; respects confidence threshold |
| `SpatialRegistry` | Establishes referents; resolves references; decays correctly; handles slot conflicts |
| `TemporalAligner` | Correct NMS↔gloss overlap detection; handles non-overlapping signals; multi-NMS alignment |
| `ConfidenceRouter` | Filters transitions; triggers SLM at threshold; triggers on silence; buffers correctly |

### 9.3 Integration Tests (Adapters)

Each adapter tested against real dependencies with fixture inputs.

| Adapter | Fixture | Assertion |
|---------|---------|-----------|
| `MediaPipeHolisticAdapter` | 10 saved video frames (.png) | Returns 542 landmarks, normalized correctly |
| `LookupTableAdapter` | 20 saved landmark sequences | Top-1 accuracy >80% on known signs |
| `CTCClassifierAdapter` | 5 continuous signing sequences | Produces valid gloss sequences (no crashes) |
| `SLMAdapter` | 10 gloss sequence strings | Produces grammatical English (manual verification) |

### 9.4 Regression Test Suite

Saved landmark sequences with expected translations. Run on every model retrain to detect regressions:

```
fixtures/
├── isolated_signs/
│   ├── HELLO_001.npz          → expected: "Hello"
│   ├── THANKYOU_001.npz       → expected: "Thank you"
│   └── ...
├── sentences/
│   ├── store_question_001.npz → expected: "Are you going to the store?"
│   ├── mother_reference_001.npz → expected: "My mother said..."
│   └── ...
└── edge_cases/
    ├── fast_signing_001.npz   → expected: graceful degradation, no crash
    ├── one_hand_only_001.npz  → expected: partial translation
    └── poor_lighting_001.npz  → expected: low confidence, no hallucination
```

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|:---:|:---:|------|
| BSL-1K dataset access denied | Medium | Critical | Fallback: self-record 100 signs using webcam; use SignBank as reference |
| Phi-3 4-bit too slow on laptop CPU | Low | High | Alternative: use smaller SLM (TinyLlama 1.1B), or offload to GPU via llama.cpp CUDA |
| Coarticulation unsolvable with available data | Medium | High | Accept Phase 1 limitation (isolated signs); focus on domain-specific controlled signing |
| BSL spatial grammar too complex for rule-based Phase 1 | Medium | Medium | Limit Phase 1 to non-spatial sentences; spatial features are Phase 1.5+ |
| MediaPipe landmark noise too high for classification | Low | High | Add Kalman filter smoothing to landmark stream; increase training augmentation noise |
| Solo dev burnout / scope creep | High | Critical | Strict phase gates; each phase produces usable system; no feature creep between phases |
| SLM hallucination on unfamiliar gloss sequences | Medium | Medium | Constrained decoding (limit output vocabulary); confidence-gated output (suppress low-confidence translations) |
| Two-handed BSL fingerspelling recognition failure | Medium | Medium | Defer fingerspelling to Phase 3; treat as out-of-vocabulary in earlier phases |

---

## Appendix A: Technology Stack

| Layer | Technology | Version | License |
|-------|-----------|---------|---------|
| Language | Python | 3.11+ | PSF |
| Vision | MediaPipe | 0.10+ | Apache 2.0 |
| ML Framework | PyTorch | 2.0+ | BSD |
| SLM Inference | llama-cpp-python | latest | MIT |
| SLM Base Model | Phi-3-Mini / Mistral-7B | — | MIT / Apache 2.0 |
| Fine-tuning | PEFT (LoRA) | latest | Apache 2.0 |
| Quantization | AutoGPTQ / llama.cpp | — | MIT |
| Video I/O | OpenCV | 4.8+ | Apache 2.0 |
| DI Container | dependency-injector | 4.41+ | BSD |
| Config | Pydantic + YAML | — | MIT |
| Testing | pytest | 7.0+ | MIT |

## Appendix B: BSL Linguistic Reference

**Key BSL grammatical features the system must handle:**

1. **Topic-Comment structure:** "STORE, I GO" = "I'm going to the store" (topic fronted)
2. **Time-first ordering:** "TOMORROW I WORK" = "I'm working tomorrow" (temporal marker precedes verb)
3. **Directional verbs:** "I-GIVE-you" vs. "you-GIVE-me" encoded in movement direction, not separate words
4. **Spatial anaphora:** Referents established in space, referenced by pointing/gaze
5. **Non-manual grammar:** Sentence type (question, negation) marked by face/head, not hands
6. **Classifiers:** Handshapes representing object categories used in spatial descriptions
7. **Two-handed fingerspelling:** Distinct from ASL's one-handed alphabet

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| Gloss | A written label representing a sign (typically an English word in CAPS, e.g., `[STORE]`) |
| Coarticulation | The modification of a sign's form due to the surrounding signs |
| Movement epenthesis | Non-meaningful transitional movement between signs |
| Non-manual signal (NMS) | Grammatical information conveyed by face, head, or body (not hands) |
| Spatial anaphora | Using physical space to establish and reference discourse entities |
| Topic-comment | BSL sentence structure where the topic is stated first, then commented upon |
| CTC | Connectionist Temporal Classification — loss function for unsegmented sequences |
| LoRA | Low-Rank Adaptation — parameter-efficient fine-tuning method |
| NPU | Neural Processing Unit — dedicated ML accelerator on modern mobile chips |
| TTFT | Time to First Token — latency from input to first generated output token |
| DTW | Dynamic Time Warping — elastic distance metric for temporal sequences |
