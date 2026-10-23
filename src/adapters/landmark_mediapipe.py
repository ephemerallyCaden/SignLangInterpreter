from __future__ import annotations

import logging
import time
from pathlib import Path

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

    def open(self) -> None:
        hand_model = str(self._models_dir / "hand_landmarker.task")
        pose_model = str(self._models_dir / "pose_landmarker_lite.task")
        face_model = str(self._models_dir / "face_landmarker.task")

        self._hand_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_model),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=self._num_hands,
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        hand_result = self._hand_landmarker.detect(mp_image)
        pose_result = self._pose_landmarker.detect(mp_image)
        face_result = self._face_landmarker.detect(mp_image)

        self._last_latency_ms = (time.perf_counter() - start) * 1000
        self._frame_count += 1

        # Parse hands
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
            face_pts = np.zeros((468, 3), dtype=np.float32)

        # Normalize to signing space
        hands, pose_pts = self._normalize(left_hand, right_hand, pose_pts)

        timestamp_ms = int(self._frame_count * (1000 / 30))  # approximate

        return LandmarkFrame(
            timestamp_ms=timestamp_ms,
            hands=hands,
            face=FaceLandmarks(points=face_pts),
            pose=PoseLandmarks(points=pose_pts),
            hand_presence=(left_hand is not None, right_hand is not None),
        )

    def _normalize(
        self,
        left_hand: np.ndarray | None,
        right_hand: np.ndarray | None,
        pose: np.ndarray,
    ) -> tuple[HandLandmarks, np.ndarray]:
        """Normalize to signing space: origin=sternum, scale=shoulder width."""
        left_shoulder = pose[11]
        right_shoulder = pose[12]
        origin = (left_shoulder + right_shoulder) / 2
        shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))

        if shoulder_width < 1e-6:
            shoulder_width = 1.0  # fallback if pose detection failed

        def shift_and_scale(pts: np.ndarray) -> np.ndarray:
            return (pts - origin) / shoulder_width

        norm_pose = shift_and_scale(pose)
        norm_left = shift_and_scale(left_hand) if left_hand is not None else None
        norm_right = shift_and_scale(right_hand) if right_hand is not None else None

        return HandLandmarks(left=norm_left, right=norm_right), norm_pose

    def get_latency_ms(self) -> float:
        return self._last_latency_ms
