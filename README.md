# Sign Language Interpreter

Real-time British Sign Language (BSL) to English interpreter that runs entirely offline. Tracks hand landmarks via MediaPipe, classifies signs, and translates BSL grammar to natural English using a local language model.

![Hand tracking with 21-point skeleton overlay and real-time translation](assets/hand_tracking_demo.gif)

> **Work in progress.** Currently at hand detection and landmark extraction — the pipeline tracks 21-point hand skeletons at ~30 FPS with One Euro filtering. Sign classification and grammar translation are stubbed out pending BSL training data (BSL-1K / BOBSL, applied for academic access).

## Install

```bash
git clone https://github.com/ephemerallyCaden/SignLangInterpreter.git
cd SignLangInterpreter
pip install -e ".[dev]"
```

## Usage

```bash
# Live webcam
python -m src.main

# From a video file
python -m src.main --video path/to/video.mp4

# With diagnostic recording
python -m src.main --record
```

The overlay window draws 21-point hand skeletons and displays classification output at ~30 FPS. Press `q` to quit.

## Tech Stack

Python 3.11+ · MediaPipe · OpenCV · PyTorch · llama-cpp-python (planned)
