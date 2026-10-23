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

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="BSL Interpreter")
    parser.add_argument("--config", default="configs/phase1.yaml", help="Config file path")
    parser.add_argument("--video", default=None, help="Video file (instead of camera)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Wire adapters
    if args.video:
        camera = VideoFileAdapter(args.video)
    else:
        camera = CameraOpenCVAdapter(config.camera.index)

    extractor = MediaPipeCompositeAdapter()
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

    # Open resources inside try block so cleanup always runs
    try:
        camera.open()
        extractor.open()

        log.info("Pipeline running. Press 'q' to quit.")

        while True:
            frame = camera.acquire_frame()
            if frame is None:
                break

            landmarks = extractor.extract(frame)
            if landmarks is not None:
                overlay.set_landmarks(landmarks)
                result = orchestrator.process_frame(landmarks)
                if result is not None:
                    console.emit(result)
                    overlay.emit(result)

            if config.show_overlay:
                vis = overlay.draw_frame(frame)
                cv2.imshow("BSL Interpreter", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Yield CPU when no GUI event loop
                time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        extractor.close()
        cv2.destroyAllWindows()
        log.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
