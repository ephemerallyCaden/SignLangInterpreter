from __future__ import annotations

import argparse
import logging
import time

import cv2

from src.adapters.camera_opencv import CameraOpenCVAdapter
from src.adapters.camera_video_file import VideoFileAdapter
from src.adapters.classifier_lookup import LookupTableAdapter
from src.adapters.landmark_mediapipe import MediaPipeCompositeAdapter
from src.adapters.output_console import ConsoleOutputAdapter
from src.adapters.output_overlay import OverlayOutputAdapter
from src.adapters.transformer_rules import RuleBasedAdapter
from src.core.orchestrator import TranslationOrchestrator
from src.infrastructure.config import load_config
from src.infrastructure.diagnostics import SessionRecorder
from src.infrastructure.smoothing import HandPresenceSmoother

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="BSL Interpreter")
    parser.add_argument("--config", default="configs/phase1.yaml", help="Config file path")
    parser.add_argument("--video", default=None, help="Video file (instead of camera)")
    parser.add_argument("--record", action="store_true", help="Record diagnostics session")
    args = parser.parse_args()

    config = load_config(args.config)

    # Wire adapters
    if args.video:
        camera = VideoFileAdapter(args.video)
    else:
        camera = CameraOpenCVAdapter(config.camera.index)

    extractor = MediaPipeCompositeAdapter()
    smoother = HandPresenceSmoother()
    classifier = LookupTableAdapter()
    transformer = RuleBasedAdapter()
    console = ConsoleOutputAdapter()
    overlay = OverlayOutputAdapter()

    orchestrator = TranslationOrchestrator(
        sign_classifier=classifier,
        grammar_transformer=transformer,
        window_size_ms=config.pipeline.window_size_ms,
        stride_ms=config.pipeline.stride_ms,
    )

    # Diagnostics recorder
    recorder = None
    if args.record or config.diagnostics.enabled:
        recorder = SessionRecorder(
            output_dir=config.diagnostics.output_dir,
            snapshot_interval=config.diagnostics.snapshot_interval,
        )

    try:
        camera.open()
        extractor.open()
        if recorder:
            recorder.open(config_dict=config.model_dump())

        log.info("Pipeline running. Press 'q' to quit.")

        while True:
            frame = camera.acquire_frame()
            if frame is None:
                break

            landmarks = extractor.extract(frame)
            result = None

            if landmarks is not None:
                landmarks = smoother.smooth(landmarks)
                overlay.set_landmarks(landmarks)
                result = orchestrator.process_frame(landmarks)
                if result is not None:
                    console.emit(result)
                    overlay.emit(result)

            # Record diagnostics
            if recorder:
                annotated = overlay.draw_frame(frame) if config.show_overlay else None
                recorder.record_frame(
                    frame=frame,
                    landmarks=landmarks,
                    result=result,
                    extract_ms=extractor.get_latency_ms(),
                    buffer_size=len(orchestrator._frame_buffer),
                    shoulder_width=extractor.last_shoulder_width,
                    annotated_frame=annotated,
                )

            if config.show_overlay:
                vis = overlay.draw_frame(frame)
                cv2.imshow("BSL Interpreter", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        if recorder:
            recorder.close()
        camera.close()
        extractor.close()
        cv2.destroyAllWindows()
        log.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
