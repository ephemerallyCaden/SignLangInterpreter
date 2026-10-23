from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.core.models import FaceLandmarks, HandLandmarks, LandmarkFrame, PoseLandmarks
from src.ports.landmark_extractor import LandmarkExtractor

log = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
VisionRunningMode = mp.tasks.vision.RunningMode

FACE_LANDMARK_COUNT = 478  # 468 mesh + 10 iris
SHOULDER_WIDTH_SMOOTHING = 0.7  # EMA alpha: higher = more smoothing


def _landmarks_to_array(landmark_list: list) -> np.ndarray:
    """Convert a MediaPipe landmark list to a (N, 3) numpy array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32)


class MediaPipeCompositeAdapter(LandmarkExtractor):
    """Runs Hand, Pose, and Face Landmarkers and merges results into a LandmarkFrame."""

    def __init__(self, models_dir: str | Path = "models/mediapipe", num_hands: int = 2):
        self._models_dir = Path(models_dir)
        self._num_hands = num_hands
        self._hand_landmarker: HandLandmarker | None = None
        self._pose_landmarker: PoseLandmarker | None = None
        self._face_landmarker: FaceLandmarker | None = None
        self._last_latency_ms: float = 0.0
        self._frame_count = 0
        self._last_origin: np.ndarray | None = None
        self._last_shoulder_width: float | None = None
        self._smoothed_shoulder_width: float | None = None

    def open(self) -> None:
        hand_model = str(self._models_dir / "hand_landmarker.task")
        pose_model = str(self._models_dir / "pose_landmarker_lite.task")
        face_model = str(self._models_dir / "face_landmarker.task")

        self._hand_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_model),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=self._num_hands,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
        self._pose_landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_model),
                running_mode=VisionRunningMode.IMAGE,
            )
        )
        self._face_landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=face_model),
                running_mode=VisionRunningMode.IMAGE,
            )
        )
        log.info("MediaPipe landmarkers initialized from %s", self._models_dir)

    def close(self) -> None:
        for lm in [self._hand_landmarker, self._pose_landmarker, self._face_landmarker]:
            if lm is not None:
                lm.close()

    def extract(self, frame: np.ndarray) -> LandmarkFrame | None:
        if self._hand_landmarker is None:
            return None

        start = time.perf_counter()

        # OpenCV captures BGR; MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        hand_result = self._hand_landmarker.detect(mp_image)
        pose_result = self._pose_landmarker.detect(mp_image)
        face_result = self._face_landmarker.detect(mp_image)

        self._last_latency_ms = (time.perf_counter() - start) * 1000
        self._frame_count += 1

        # Parse hands (raw MediaPipe [0,1] coords)
        left_hand = None
        right_hand = None
        for i, handedness_list in enumerate(hand_result.handedness):
            label = handedness_list[0].category_name
            pts = _landmarks_to_array(hand_result.hand_landmarks[i])
            # MediaPipe mirrors: "Left" in result = signer's right hand
            if label == "Left":
                right_hand = pts
            else:
                left_hand = pts

        # Parse pose
        if pose_result.pose_landmarks:
            pose_pts = _landmarks_to_array(pose_result.pose_landmarks[0])
        else:
            pose_pts = np.zeros((33, 3), dtype=np.float32)

        # Parse face
        if face_result.face_landmarks:
            face_pts = _landmarks_to_array(face_result.face_landmarks[0])
        else:
            face_pts = np.zeros((FACE_LANDMARK_COUNT, 3), dtype=np.float32)

        # Keep raw hand coords for display before normalization
        raw_hands = HandLandmarks(left=left_hand, right=right_hand)

        # Normalize all landmarks to signing space
        norm_result = self._normalize(left_hand, right_hand, pose_pts, face_pts)
        if norm_result is None:
            return None

        norm_hands, norm_pose, norm_face = norm_result
        timestamp_ms = int(self._frame_count * (1000 / 30))

        return LandmarkFrame(
            timestamp_ms=timestamp_ms,
            hands=norm_hands,
            face=FaceLandmarks(points=norm_face),
            pose=PoseLandmarks(points=norm_pose),
            hand_presence=(left_hand is not None, right_hand is not None),
            raw_hands=raw_hands,
        )

    def _normalize(
        self,
        left_hand: np.ndarray | None,
        right_hand: np.ndarray | None,
        pose: np.ndarray,
        face: np.ndarray,
    ) -> tuple[HandLandmarks, np.ndarray, np.ndarray] | None:
        """Normalize to signing space: origin=sternum, scale=shoulder width.

        Shoulder width is smoothed with an EMA to prevent jittery normalization.
        """
        left_shoulder = pose[11]
        right_shoulder = pose[12]
        raw_width = float(np.linalg.norm(left_shoulder - right_shoulder))

        if raw_width > 1e-6:
            origin = (left_shoulder + right_shoulder) / 2
            self._last_origin = origin

            # EMA smoothing on shoulder width
            if self._smoothed_shoulder_width is None:
                self._smoothed_shoulder_width = raw_width
            else:
                alpha = SHOULDER_WIDTH_SMOOTHING
                self._smoothed_shoulder_width = (
                    alpha * self._smoothed_shoulder_width + (1 - alpha) * raw_width
                )
            self._last_shoulder_width = self._smoothed_shoulder_width
            shoulder_width = self._smoothed_shoulder_width
        elif self._last_origin is not None:
            origin = self._last_origin
            shoulder_width = self._last_shoulder_width  # type: ignore[assignment]
        else:
            return None

        def shift_and_scale(pts: np.ndarray) -> np.ndarray:
            return (pts - origin) / shoulder_width

        norm_pose = shift_and_scale(pose)
        norm_face = shift_and_scale(face)
        norm_left = shift_and_scale(left_hand) if left_hand is not None else None
        norm_right = shift_and_scale(right_hand) if right_hand is not None else None

        return HandLandmarks(left=norm_left, right=norm_right), norm_pose, norm_face

    def get_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def last_shoulder_width(self) -> float | None:
        return self._last_shoulder_width
