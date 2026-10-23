# SignLanguageInterpreter: Technical Project Plan

## Document Metadata

| Field | Value |
|-------|-------|
| Codename | SignLanguageInterpreter |
| Target Language | British Sign Language (BSL) |
| Architecture | Hexagonal (Ports & Adapters) Monolith |
| Deployment Target | Laptop GPU (dev); Mobile NPU (production) |
| Privacy Model | Fully offline, zero network inference |
| Author | Solo Developer |
| Status | Pre-implementation Planning |

---

## 1. System Overview

### 1.1 Problem Statement

BSL is a complete, independent language — not a signed encoding of English. Current approaches fail because they:

- Treat signing as static image classification (losing temporal/spatial semantics)
- Rely on cloud inference (latency + privacy violations)
- Map signs 1:1 to English words (ignoring BSL topic-comment structure, spatial anaphora, non-manual grammar)

### 1.2 Pipeline

```
Camera → [Vision Layer] → Landmarks → [Classifier Layer] → Enriched Glosses → [SLM Layer] → English Text
                                              ↕
                                    [NMS Classifier]
                                              ↕
                                    [Spatial Registry]
```

The system treats BSL as a **Dynamic Biomechanical Stream** — a continuous, multi-channel signal (manual articulation + non-manual signals + spatial reference) — and translates it to English via a local Small Language Model.

### 1.3 Design Principles

| Principle | Implication |
|-----------|-------------|
| Privacy-first | All inference on-device. No telemetry. No cloud fallback. |
| Offline-capable | Full functionality with no network connection. |
| Linguistically informed | BSL grammar drives architecture, not English assumptions. |
| Phased viability | Each phase produces a usable (if limited) system. |
| Swap-without-rewire | Any component replaceable via adapter pattern. |
| Observable | Every intermediate representation inspectable. |

---

## 2. Architecture in Detail

### 2.1 Hexagonal Architecture (Ports & Adapters)

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
   │  Composite   │     │  │    extract(Frame) → LandmarkFrame            │   │      │   Output    │
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
   │  SLM         │◄────┼──┤                                                   │
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

### 2.2 Why Hexagonal?

| Criterion | Microservice | Event-Driven | Hexagonal Monolith |
|-----------|:---:|:---:|:---:|
| Solo-dev complexity | High | Medium | **Low** |
| IPC latency at 30-60 FPS | ~2-5ms | ~1ms (async) | **~0ms (in-process)** |
| Component swappability | High | Medium | **High** |
| Testability (mock injection) | High | Medium | **High** |
| Phased delivery | Medium | Medium | **High** |
| Mobile portability | Low | Low | **High** |
| Shared state (spatial context) | Difficult | Native (event store) | **Native (domain object)** |

### 2.3 Concurrency Model

```
[Frame Producer Thread]
         │
         ▼
   BoundedQueue(maxsize=3)  ← drops oldest frame if full (backpressure)
         │
         ▼
[Processing Thread]
   ├── LandmarkExtractor.extract()          ~20ms (3 landmarkers in parallel)
   ├── Parallel:
   │     ├── SignClassifier.classify()       ~20ms
   │     └── NMSClassifier.classify()       ~5ms
   ├── Orchestrator.merge_and_route()        ~1ms
   └── GrammarTransformer.transform()        ~80ms (SLM inference)
         │
         ▼
   OutputQueue → [Render Thread]
```

**Backpressure:** Queue drops oldest unprocessed frame. Recency over completeness — a current-but-incomplete translation beats a delayed-but-complete one.

### 2.4 Directory Structure

```
SignLanguageInterpreter/
├── src/
│   ├── core/                          # Domain logic (zero external dependencies)
│   │   ├── orchestrator.py            # TranslationOrchestrator
│   │   ├── models.py                  # Domain data classes
│   │   ├── spatial_registry.py        # Spatial anaphora tracking
│   │   ├── temporal_aligner.py        # NMS ↔ Gloss alignment
│   │   └── confidence_router.py       # Hybrid routing logic
│   │
│   ├── ports/                         # Abstract interfaces
│   │   ├── frame_source.py
│   │   ├── landmark_extractor.py
│   │   ├── sign_classifier.py
│   │   ├── nms_classifier.py
│   │   ├── grammar_transformer.py
│   │   └── output_sink.py
│   │
│   ├── adapters/                      # Concrete implementations
│   │   ├── camera_opencv.py           # FrameSource → OpenCV webcam
│   │   ├── camera_video_file.py       # FrameSource → Video file (testing)
│   │   ├── landmark_mediapipe.py      # LandmarkExtractor → MediaPipe Composite
│   │   ├── classifier_lookup.py       # SignClassifier → KNN/DTW (Phase 1)
│   │   ├── classifier_ctc.py          # SignClassifier → CTC neural (Phase 2)
│   │   ├── nms_face_cnn.py            # NMSClassifier → Face CNN
│   │   ├── transformer_rules.py       # GrammarTransformer → Rules (Phase 1)
│   │   ├── transformer_slm.py         # GrammarTransformer → SLM (Phase 2)
│   │   ├── output_console.py          # OutputSink → Terminal
│   │   └── output_overlay.py          # OutputSink → Video overlay
│   │
│   ├── infrastructure/                # Wiring, config, lifecycle
│   │   ├── container.py               # DI container
│   │   ├── config.py
│   │   └── pipeline.py               # Thread management, queues
│   │
│   └── main.py
│
├── models/                            # Weights & artifacts (gitignored)
├── data/                              # Training data & processed datasets
│   ├── raw/                           # BSL-1K / BOBSL clips
│   ├── processed/                     # Extracted landmark sequences
│   ├── gloss_lexicon/                 # Sign ID → label mappings
│   └── synthetic_gloss/              # SLM-generated gloss annotations
├── training/
│   ├── train_classifier.py
│   ├── train_nms.py
│   ├── finetune_slm.py
│   └── generate_synthetic_gloss.py
├── tests/
│   ├── unit/                          # Core logic (mocked ports)
│   ├── integration/                   # Adapter tests (real dependencies)
│   └── fixtures/                     # Saved landmark sequences
├── configs/
│   ├── phase1.yaml
│   ├── phase2.yaml
│   └── benchmark.yaml
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
    timestamp_ms: int
    hands: HandLandmarks          # 21 landmarks × 2 hands × 3 coords = 126 floats
    face: FaceLandmarks           # 468 landmarks × 3 coords = 1404 floats
    pose: PoseLandmarks           # 33 landmarks × 3 coords = 99 floats
    hand_presence: tuple[bool, bool]  # (left_detected, right_detected)

@dataclass(frozen=True)
class HandLandmarks:
    """Origin at wrist, scale-invariant."""
    left: np.ndarray   # shape: (21, 3) or None
    right: np.ndarray  # shape: (21, 3) or None

@dataclass(frozen=True)
class LandmarkWindow:
    frames: list[LandmarkFrame]    # Ordered by timestamp
    window_size_ms: int            # Typically 500-2000ms
    stride_ms: int                 # Overlap between consecutive windows

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
    """Critical intermediate representation between classifier and SLM."""
    sign_id: str                        # e.g., "STORE", "GO", "POINT"
    confidence: float                   # 0.0 - 1.0
    temporal_span: tuple[int, int]      # (start_ms, end_ms)
    dominant_hand: str                  # "left" | "right" | "both"
    spatial_slot: SpatialSlot | None    # Where in signing space
    is_establishing: bool               # True if PLACING a referent
    is_referencing: bool                # True if pointing to established slot
    movement_vector: np.ndarray | None  # For directional verbs (GIVE-to-you vs GIVE-to-me)

@dataclass(frozen=True)
class NMSSignal:
    signal_type: NMSType
    confidence: float
    temporal_span: tuple[int, int]
    intensity: float                    # 0.0 - 1.0

class NMSType(Enum):
    QUESTION_YES_NO = "q_yn"           # Brow raise + head forward
    QUESTION_WH = "q_wh"              # Brow furrow + head tilt
    NEGATION = "neg"                   # Head shake
    TOPIC_MARKER = "topic"             # Brow raise + head back + pause
    INTENSIFIER = "intens"             # Puffed cheeks / pursed lips
    CONDITIONAL = "cond"               # Brow raise + head tilt (if-clause)

@dataclass
class GlossSequence:
    """Enriched sequence ready for SLM: glosses + NMS + spatial context."""
    tokens: list[GlossToken]
    nms_signals: list[NMSSignal]
    spatial_context: dict[SpatialSlot, str]  # Slot → established referent sign_id
    sequence_duration_ms: int

@dataclass(frozen=True)
class TranslationResult:
    text: str                           # Natural English translation
    confidence: float
    gloss_sequence: GlossSequence       # For debugging / visual feedback
    processing_time_ms: int
```

### 3.2 SLM Input Serialization

Contract between classifier layer and grammar transformer:

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

**Expected output:** `"Is your mother going to the store later?"`

---

## 4. Module Specifications

### 4.1 FrameSource

```python
class FrameSource(ABC):
    @abstractmethod
    def open(self) -> None: ...
    @abstractmethod
    def acquire_frame(self) -> Frame | None: ...
    @abstractmethod
    def get_fps(self) -> float: ...
    @abstractmethod
    def close(self) -> None: ...
```

**Adapters:** `CameraOpenCVAdapter` (live webcam), `VideoFileAdapter` (pre-recorded, for testing)

### 4.2 LandmarkExtractor

```python
class LandmarkExtractor(ABC):
    @abstractmethod
    def extract(self, frame: Frame) -> LandmarkFrame:
        """Extracts 542 landmarks (hands: 42, face: 468, pose: 33),
        normalized to signing-space coordinates (origin: sternum, scale: shoulder width)."""
        ...
    @abstractmethod
    def get_latency_ms(self) -> float: ...
```

**Normalization:**
- Origin: midpoint between shoulders (translation invariance)
- Scale: normalized by shoulder width (scale invariance)
- Rotation: aligned to torso plane (vertical axis rotation invariance)
- Handedness: dominant hand landmarks always mapped to consistent array position

**Note:** MediaPipe Holistic is deprecated. Uses three separate MediaPipe tasks (Hand, Pose, Face Landmarker) via a composite adapter that merges outputs.

**Adapter:** `MediaPipeCompositeAdapter` — runs 3 landmarkers in parallel (~20ms GPU, ~30ms CPU)

### 4.3 SignClassifier

```python
class SignClassifier(ABC):
    @abstractmethod
    def classify(self, window: LandmarkWindow) -> list[GlossCandidate]: ...
    @abstractmethod
    def get_vocabulary_size(self) -> int: ...
    @abstractmethod
    def supports_continuous(self) -> bool: ...

@dataclass
class GlossCandidate:
    sign_id: str
    confidence: float
    temporal_span: tuple[int, int]
    spatial_slot: SpatialSlot | None
    is_transition: bool  # True if movement epenthesis
```

| Adapter | Phase | Approach | Continuous? |
|---------|-------|----------|-------------|
| `LookupTableAdapter` | 1 | KNN/DTW on landmark templates | No |
| `CTCClassifierAdapter` | 2 | GRU encoder + CTC decoder | Yes |

### 4.4 NMSClassifier

```python
class NMSClassifier(ABC):
    @abstractmethod
    def classify(self, face_landmarks: FaceLandmarks, window_ms: int) -> list[NMSSignal]: ...
```

- Input: 468-point face mesh subset (brows: 65-70, 295-300; mouth: 61-67, 291-297; head pose: 1, 4, 5, 6, 197, 195)
- Architecture: 1D-CNN over temporal face landmark deltas
- Features:
  - Brow raise: `mean(brow_y) - mean(eye_y)` normalized
  - Head forward lean: nose tip z-displacement relative to ears
  - Head shake: lateral oscillation frequency of nose tip x-coordinate

### 4.5 GrammarTransformer

```python
class GrammarTransformer(ABC):
    @abstractmethod
    def transform(self, sequence: GlossSequence, context: ConversationContext) -> TranslationResult: ...
```

| Adapter | Phase | Approach | Latency |
|---------|-------|----------|---------|
| `RuleBasedAdapter` | 1 | Hardcoded BSL→English reordering + templates | <5ms |
| `SLMAdapter` | 2 | Candidate SLM (4-bit quantized) via llama.cpp — see §6.3 | ~80-150ms |

**Rule-Based Adapter (Phase 1) rules:**
1. BSL topic-comment → English SVO reordering
2. `[FUTURE]` / `[PAST]` → tense marking on verb
3. `[NMS:QUESTION_YN]` → invert subject-auxiliary
4. `[NMS:NEGATION]` → insert "not" / "don't" before verb
5. `[POINT:slot]` → resolve from SpatialRegistry → pronoun

### 4.6 TranslationOrchestrator

```python
class TranslationOrchestrator:
    """Central coordinator. No external dependencies — only ports and domain objects."""

    def __init__(self, sign_classifier, nms_classifier, grammar_transformer,
                 spatial_registry, confidence_router, temporal_aligner): ...

    def process_window(self, window: LandmarkWindow) -> TranslationResult | None:
        # 1. Classify manual signs
        gloss_candidates = self.sign_classifier.classify(window)
        # 2. Classify non-manual signals (parallel in production)
        nms_signals = self.nms_classifier.classify(self._extract_face(window), window.window_size_ms)
        # 3. Filter transitions, apply confidence threshold
        glosses = self.confidence_router.filter_and_route(gloss_candidates)
        # 4. Update spatial registry
        self.spatial_registry.update(glosses)
        # 5. Align NMS with gloss timeline
        aligned = self.temporal_aligner.align(glosses, nms_signals)
        # 6. Build enriched sequence
        sequence = GlossSequence(
            tokens=aligned.tokens, nms_signals=aligned.nms,
            spatial_context=self.spatial_registry.get_active_context(),
            sequence_duration_ms=window.window_size_ms,
        )
        # 7. Grammar transformation (if enough context accumulated)
        if self.confidence_router.should_invoke_slm(sequence):
            return self.grammar_transformer.transform(sequence, self._conversation_context)
        return None
```

### 4.7 SpatialRegistry

Tracks spatial anaphora — which referents have been "placed" where in signing space. Implements decaying memory (older placements lose confidence).

```python
class SpatialRegistry:
    def __init__(self, decay_rate: float = 0.95, max_slots: int = 6):
        self._registry: dict[SpatialSlot, SpatialReferent] = {}

    def update(self, glosses: list[GlossToken]) -> None:
        self._decay_all()
        for gloss in glosses:
            if gloss.is_establishing and gloss.spatial_slot:
                self._registry[gloss.spatial_slot] = SpatialReferent(
                    sign_id=gloss.sign_id, established_at=gloss.temporal_span[0],
                    confidence=gloss.confidence)

    def resolve(self, slot: SpatialSlot) -> str | None:
        ref = self._registry.get(slot)
        return ref.sign_id if ref and ref.confidence > 0.3 else None

    def get_active_context(self) -> dict[SpatialSlot, str]:
        return {slot: ref.sign_id for slot, ref in self._registry.items() if ref.confidence > 0.3}

    def _decay_all(self) -> None:
        for ref in self._registry.values():
            ref.confidence *= self._decay_rate
```

### 4.8 ConfidenceRouter

```python
class ConfidenceRouter:
    """High-confidence isolated signs → direct lookup. Accumulated buffer → SLM invocation."""

    def __init__(self,
        high_confidence_threshold: float = 0.92,
        slm_trigger_threshold: int = 3,        # Min glosses before invoking SLM
        slm_silence_trigger_ms: int = 800,     # Invoke SLM after 800ms pause
    ): ...

    def filter_and_route(self, candidates: list[GlossCandidate]) -> list[GlossToken]: ...

    def should_invoke_slm(self, sequence: GlossSequence) -> bool:
        """Triggers on: buffer >= threshold tokens, 800ms silence, or NMS sentence boundary."""
        ...
```

---

## 5. Critical Technical Challenges

### 5.1 Coarticulation ("Cursive" Problem)

**Problem:** In fluent BSL, signs blend together. A classifier trained on isolated clips fails on continuous streams because it never sees transition movements.

**Strategy A — CTC (Phase 2):** CTC blank token absorbs transition frames, eliminating explicit segmentation:

```
Input:  [frame_1, frame_2, ..., frame_T]
Output: [BLANK, STORE, STORE, BLANK, GO, GO, GO, BLANK, ...]
                    ↓ CTC collapse
        [STORE, GO]
```

**Strategy B — Epenthesis Detection (Phase 1.5):** Transitions have distinct kinematics:

| Property | Within-Sign | Transition |
|----------|-------------|------------|
| Path shape | Curved, deliberate | Straight-line (ballistic) |
| Velocity profile | Bell-shaped (smooth) | Asymmetric (fast start, slow end) |
| Hand configuration | Stable / systematically changing | Rapidly restructuring |
| Wrist acceleration | Moderate | High peak |

Binary classifier on velocity + acceleration (5-frame micro-window) flags transitions for exclusion.

**Strategy C — Training Curriculum:**

| Phase | Data | Coarticulation Exposure |
|-------|------|------------------------|
| 1 | BSL-1K isolated clips | None (accepts limitation) |
| 1.5 | BSL-1K sign pairs/triplets + self-recorded fingerspelling | Bounded transitions |
| 2 | BOBSL continuous segments | Full |

### 5.2 Spatial Anaphora Loss

**Problem:** BSL signers establish referents at 3D locations. Pointing/gaze to those locations = pronominal reference. A flat `[POINT]` gloss loses *where* the point was directed.

**Solution (three layers):**
1. **Classifier:** Detects spatial slot from wrist position at sign termination. Six discretized zones.
2. **SpatialRegistry:** Decaying memory of established referents (BSL refreshes spatial assignments every ~30-60s).
3. **SLM:** Receives spatial context in prompt, resolves `[POINT:LEFT_NEAR]` → pronoun.

**Key decision:** Classifier only reports geometry. Resolution is the SLM's job.

```
                    [TOP]
                      │
    [LEFT_FAR] [LEFT_NEAR] [CENTER] [RIGHT_NEAR] [RIGHT_FAR]
         │          │          │          │            │
       -0.8       -0.4       0.0        0.4          0.8
                  (normalized x-coordinate, relative to shoulder width)
```

Signing space: ~±1.0 shoulder-widths horizontal, ±0.6 vertical from sternum.

### 5.3 Non-Manual Signal Omission

**Problem:** BSL encodes sentence type and negation through face/head/body, not hands. Without NMS, questions and negation are invisible.

**Solution:** Dedicated `NMSClassifier` runs in parallel with `SignClassifier` on face landmarks only. Separate model because:
1. NMS are suprasegmental (span multiple signs)
2. Face-only input = smaller, faster model
3. Independent failure modes — missed NMS degrades but doesn't corrupt sign detection

| NMS Type | Features | Detection |
|----------|---------|-----------|
| Yes/No Question | Brow raise, head forward | brow-eye distance + nose z-delta |
| WH-Question | Brow furrow, head tilt | Brow convergence + head roll |
| Negation | Head shake (2-4 Hz) | FFT on nose-tip x, peak at 2-4 Hz |
| Topic Marker | Brow raise + head back + pause | Combined brow + head pitch + silence |
| Conditional | Brow raise + head tilt + body lean | Multi-feature threshold |
| Intensifier | Puffed cheeks, pursed lips | Mouth landmark area ratio |

**Temporal alignment:** NMS span clauses, not signs. `TemporalAligner` assigns each NMS to all temporally overlapping gloss tokens.

### 5.4 Gloss Bottleneck

**Problem:** No universal BSL gloss standard. Annotation requires expert linguists — impractical for a solo developer.

**Strategy A (Phase 1):** BSL-1K provides isolated clips with single-word labels. No annotation problem for isolated recognition.

**Strategy B (Phase 2):** Synthetic gloss generation from BOBSL subtitles:

```
Input:  "Is your mother going to the store?"
Prompt: "Convert to BSL gloss order (topic-comment, time-first, NMS for questions)"
Output: [MOTHER] [STORE] [GO] [FUTURE] [NMS:Q_YN]
Validate: Cross-reference temporal alignment with video landmarks
```

~60-70% correct ordering. Sufficient as weak supervision, not evaluation.

**Strategy C (Phase 3):** Two-tower contrastive learning bypasses gloss entirely:

```
Tower 1: LandmarkSequence → SignEmbedding (trained encoder)
Tower 2: EnglishText → TextEmbedding (SLM encoder)
Loss:    Contrastive — pull matching (video, subtitle) pairs together
```

Explicit gloss pipeline remains for interpretability.

**Strategy D (long-term):** Community bootstrapping — BSL users verify detected glosses via feedback overlay.

---

## 6. Training & Data Strategy

### 6.1 Datasets

| Dataset | Size | Use | Access |
|---------|------|-----|--------|
| BSL-1K | ~1,000 sign classes | Phase 1 isolated classifier | Academic application (Oxford VGG) |
| BOBSL | ~1,400 hours | Phase 2 continuous signing, SLM fine-tuning | Academic application (Oxford VGG) |
| BSL Corpus (UCL DCAL) | ~250 hours | Gold-standard gloss validation | Academic application (UCL) |
| SignBank BSL | Dictionary-scale | Linguistic validation, gloss lexicon | Open access |
| Self-recorded fingerspelling | ~5 hours (target) | Two-handed BSL alphabet (Phase 1.5) | Self-generated |

> **IMMEDIATE ACTION:** Apply for BSL-1K and BOBSL access now. Critical path for Phase 1 — application turnaround unknown.

### 6.2 Data Processing

```
Raw Video → MediaPipe Extraction → Normalized Landmarks → Windowed Sequences → Training Format

- Frame rate: 30 FPS (downsample 60fps)
- Window: 64 frames (2.13s), 50% overlap
- Normalization: shoulder-width scaling, sternum-origin, torso-aligned
- Augmentation:
    - Temporal jitter: ±3 frames
    - Spatial noise: Gaussian σ=0.01 on landmarks
    - Speed: 0.8x-1.2x (resample landmarks)
    - Mirror: horizontal flip (hand label swap) — CAUTION: some BSL signs are hand-specific
```

### 6.3 Training Specs

**Sign Classifier (Phase 2 — CTC):**

| Parameter | Value |
|-----------|-------|
| Architecture | 2-layer GRU (256 hidden) + CTC head |
| Input | 126-dim hand landmarks (21×3×2) per frame |
| Output | vocab_size + 1 (CTC blank) |
| Loss | CTC |
| Data | BSL-1K (isolated) → BOBSL (continuous) |
| Batch / LR | 32 / 1e-3 Adam + cosine annealing |
| Training time | ~8h GPU (BSL-1K), ~48h (BOBSL subset) |

**NMS Classifier:**

| Parameter | Value |
|-----------|-------|
| Architecture | 1D-CNN (3 layers, 64-128-64) + sigmoid heads |
| Input | 8-dim engineered features (brow, head pose, mouth) per frame |
| Output | 6 NMS types × confidence (multi-label) |
| Window | 30 frames (1 second) |
| Data | BSL Corpus (ELAN annotations) + rule-based pseudo-labels from BOBSL |

**SLM Fine-tuning (Phase 2 — LoRA):**

| Parameter | Value |
|-----------|-------|
| Base model | TBD — benchmark in Phase 2 |
| Fine-tuning | LoRA (rank 16, alpha 32), ~10M trainable params |
| Quantization | 4-bit (GPTQ or AWQ) |
| Data | 10K-50K synthetic gloss→English pairs from BOBSL subtitles |
| Format | Structured prompt (§3.2) → English completion |
| Evaluation | BERTScore + COMET + human eval |

**SLM Candidates:**

| Model | Size | 4-bit VRAM | Notes |
|-------|------|------------|-------|
| Phi-4-mini | 3.8B | ~2.2 GB | Strong structured-input parsing |
| Gemma-2 | 2B | ~1.5 GB | Smallest, fastest, may trade quality |
| Qwen-2.5 | 3B | ~2.0 GB | Strong multilingual + structured input |
| Mistral-7B | 7B | ~4.0 GB | Baseline quality reference |

Benchmark: 50 gloss→English examples from §3.2 format. Select on BERTScore, TTFT, and memory.

**Why BERTScore/COMET over BLEU:** BSL→English allows many valid translations. Semantic similarity handles paraphrase variation; n-gram overlap does not.

---

## 7. Phased Roadmap

### Phase 1: Foundation & Isolated Sign Recognition (4-6 weeks)

Working end-to-end pipeline with isolated sign lookup. Validates architecture.

- [ ] Project scaffolding (hexagonal structure, DI container, config)
- [ ] `FrameSource` + `CameraOpenCVAdapter` + `VideoFileAdapter`
- [ ] `LandmarkExtractor` + `MediaPipeCompositeAdapter` with normalization
- [ ] `SignClassifier` + `LookupTableAdapter` (KNN/DTW, 100 BSL signs)
- [ ] `GrammarTransformer` + `RuleBasedAdapter` (basic BSL→English reordering)
- [ ] `OutputSink` + `ConsoleAdapter` + `OverlayAdapter` (skeleton viz)
- [ ] `TranslationOrchestrator` (simplified — no NMS, no spatial)
- [ ] Unit tests with fixture landmark sequences
- [ ] Data pipeline: BSL-1K → landmarks → training format

**Criteria:** 100 signs >85% accuracy, <200ms latency, 30 FPS overlay.

> **Priority:** End-to-end first. Ship working camera-to-text with fewer signs over polished architecture with no integration.

---

### Phase 1.5: NMS, Spatial Awareness & Fingerspelling (4-6 weeks)

Isolated signs + grammatically aware output + fingerspelling.

- [ ] `NMSClassifier` + `FaceCNNAdapter`
- [ ] `SpatialRegistry`, `TemporalAligner`, `ConfidenceRouter`
- [ ] NMS training (semi-supervised on BSL Corpus)
- [ ] Extend vocabulary to 250 signs
- [ ] Two-handed BSL fingerspelling classifier (26 configurations A-Z)
- [ ] Self-recorded fingerspelling data (~5 hours)
- [ ] Fingerspelling → word assembly (letter sequence → word)
- [ ] Integration tests: NMS modifies translation output

**Criteria:** Question type >80%, spatial resolution for 2 referents, fingerspelling >90% per-letter, word assembly for 3-6 letter words.

---

### Phase 2: Continuous Signing & SLM Integration (6-8 weeks)

Fluent, coarticulated BSL via neural classifier + SLM.

- [ ] `CTCClassifierAdapter` (GRU + CTC)
- [ ] SLM benchmark (§6.3 candidates)
- [ ] `SLMAdapter` (selected model, 4-bit, llama.cpp)
- [ ] LoRA fine-tuning pipeline
- [ ] Synthetic gloss generation (BOBSL subtitles)
- [ ] Coarticulation handling (CTC blanks + epenthesis detection)
- [ ] Vocabulary to 500+ signs
- [ ] Performance profiling + optimization

**Criteria:** 3-5 sign sentences >70% BERTScore, TTFT <150ms on GPU, graceful frame dropping.

---

### Phase 3: Optimization & Robustness (8-12 weeks)

Reliable performance across conditions.

- [ ] Model distillation for mobile
- [ ] Two-tower contrastive training (bypass gloss)
- [ ] Signer adaptation (few-shot calibration)
- [ ] Robustness (lighting, backgrounds, speeds)
- [ ] Domain expansion (medical, legal, everyday)
- [ ] Mobile deployment (CoreML / ONNX Runtime / TFLite)
- [ ] Fingerspelling robustness (speed variation, signer adaptation)

**Criteria:** >75% accuracy on unseen signers, varied lighting, <300ms on mobile.

---

## 8. Performance Budgets

### 8.1 Hardware

- Laptop with discrete NVIDIA GPU
- SLM inference: llama.cpp + CUDA
- MediaPipe tasks on GPU
- SLM fine-tuning may offload to cloud

### 8.2 Latency (Per Translation Cycle)

| Stage | Laptop GPU | Mobile (Phase 3) | Notes |
|-------|:---:|:---:|-------|
| Frame acquisition | <5ms | <5ms | |
| Landmark extraction | <20ms | <30ms | 3 landmarkers parallel |
| Sign classification | <25ms | <40ms | |
| NMS classification | <5ms | <10ms | Parallel with above |
| Alignment + routing | <2ms | <2ms | |
| SLM (TTFT) | <150ms | <200ms | 4-bit, GPU/NPU |
| **Worst case** | **<207ms** | **<287ms** | |
| **Typical (no SLM)** | **<55ms** | **<90ms** | |

### 8.3 Memory

| Component | Laptop | Mobile |
|-----------|--------|--------|
| MediaPipe Landmarkers | ~80 MB | ~80 MB |
| Sign Classifier (GRU) | ~15 MB | ~8 MB (distilled) |
| NMS Classifier | ~5 MB | ~3 MB |
| SLM (4-bit) | ~1.5-2.5 GB | ~1.5-2.5 GB |
| Runtime | ~200 MB | ~100 MB |
| **Total** | **~1.8-2.8 GB** | **~1.7-2.7 GB** |

### 8.4 Throughput

- Minimum: 10 FPS (every 3rd frame)
- Target: 30 FPS landmark extraction; classifier on sliding windows
- SLM: ~1-3 invocations per sentence (clause boundary triggered)

---

## 9. Testing

### 9.1 Pyramid

```
              ┌──────────────┐
              │  End-to-End  │   Video → English (5-10 scenarios, manual)
              └──────┬───────┘
           ┌─────────┴─────────┐
           │    Integration     │   Adapter + real model (per adapter)
           └─────────┬─────────┘
     ┌───────────────┴───────────────┐
     │          Unit Tests            │   Core domain (mocked ports)
     └───────────────────────────────┘
```

### 9.2 Unit Tests

| Component | Cases |
|-----------|-------|
| `TranslationOrchestrator` | Pipeline sequencing; empty output; confidence threshold |
| `SpatialRegistry` | Establish; resolve; decay; slot conflicts |
| `TemporalAligner` | NMS↔gloss overlap; non-overlapping; multi-NMS |
| `ConfidenceRouter` | Filter transitions; SLM trigger; silence trigger; buffering |

### 9.3 Integration Tests

| Adapter | Fixture | Assertion |
|---------|---------|-----------|
| `MediaPipeCompositeAdapter` | 10 frames (.png) | 542 landmarks, normalized |
| `LookupTableAdapter` | 20 landmark sequences | Top-1 >80% |
| `CTCClassifierAdapter` | 5 continuous sequences | Valid glosses (no crash) |
| `SLMAdapter` | 10 gloss strings | Grammatical English (manual) |

### 9.4 Regression Suite

Run on every model retrain:

```
fixtures/
├── isolated_signs/
│   ├── HELLO_001.npz          → "Hello"
│   ├── THANKYOU_001.npz       → "Thank you"
├── sentences/
│   ├── store_question_001.npz → "Are you going to the store?"
│   ├── mother_reference_001.npz → "My mother said..."
└── edge_cases/
    ├── fast_signing_001.npz   → graceful degradation
    ├── one_hand_only_001.npz  → partial translation
    └── poor_lighting_001.npz  → low confidence, no hallucination
```

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|:---:|:---:|------|
| BSL-1K/BOBSL access denied/delayed | High | Critical | Apply immediately. Fallback: SignBank + synthetic data; fingerspelling can proceed with self-recorded data |
| SLM too slow on laptop | Low | High | Benchmark multiple candidates (§6.3); CUDA GPU available; fallback to Gemma-2 2B |
| Coarticulation unsolvable with available data | Medium | High | Accept Phase 1 limitation; focus on controlled signing |
| Spatial grammar too complex for rule-based Phase 1 | Medium | Medium | Limit Phase 1 to non-spatial; spatial is Phase 1.5+ |
| MediaPipe landmark noise | Low | High | Kalman filter smoothing; increase augmentation noise |
| Solo dev burnout / scope creep | High | Critical | Strict phase gates; each phase = usable system |
| SLM hallucination on unfamiliar glosses | Medium | Medium | Constrained decoding; confidence-gated output |
| Fingerspelling recognition failure | Low | Medium | Developer can fingerspell BSL two-handed; self-recorded data for Phase 1.5 |

---

## Appendix A: Technology Stack

| Layer | Technology | License |
|-------|-----------|---------|
| Language | Python 3.11+ | PSF |
| Vision | MediaPipe Tasks (Hand, Pose, Face Landmarkers) | Apache 2.0 |
| ML | PyTorch 2.0+ | BSD |
| SLM Inference | llama-cpp-python (CUDA) | MIT |
| SLM Model | TBD — Phi-4-mini, Gemma-2 2B, Qwen-2.5 3B, Mistral-7B | MIT / Apache 2.0 |
| Fine-tuning | PEFT (LoRA) | Apache 2.0 |
| Quantization | AutoGPTQ / llama.cpp | MIT |
| Video I/O | OpenCV 4.8+ | Apache 2.0 |
| DI Container | dependency-injector 4.41+ | BSD |
| Config | Pydantic + YAML | MIT |
| Testing | pytest 7.0+ | MIT |

## Appendix B: BSL Linguistic Reference

1. **Topic-Comment:** "STORE, I GO" = "I'm going to the store"
2. **Time-first:** "TOMORROW I WORK" = "I'm working tomorrow"
3. **Directional verbs:** "I-GIVE-you" vs "you-GIVE-me" = movement direction, not word order
4. **Spatial anaphora:** Referents placed in space, referenced by pointing/gaze
5. **Non-manual grammar:** Sentence type (question, negation) marked by face/head
6. **Classifiers:** Handshapes representing object categories in spatial descriptions
7. **Two-handed fingerspelling:** Distinct from ASL's one-handed alphabet

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| Gloss | Written label for a sign (English word in CAPS, e.g., `[STORE]`) |
| Coarticulation | Sign form modified by surrounding signs |
| Movement epenthesis | Non-meaningful transition between signs |
| NMS | Non-manual signal — grammar via face, head, or body |
| Spatial anaphora | Using physical space to establish/reference discourse entities |
| Topic-comment | BSL structure: topic stated first, then commented upon |
| CTC | Connectionist Temporal Classification — loss for unsegmented sequences |
| LoRA | Low-Rank Adaptation — parameter-efficient fine-tuning |
| NPU | Neural Processing Unit — dedicated ML accelerator on mobile chips |
| TTFT | Time to First Token |
| DTW | Dynamic Time Warping — elastic distance for temporal sequences |
