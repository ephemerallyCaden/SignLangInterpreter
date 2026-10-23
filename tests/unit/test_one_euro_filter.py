from __future__ import annotations

import numpy as np

from src.infrastructure.one_euro_filter import LandmarkSmoother, OneEuroFilter


def test_first_value_passes_through():
    f = OneEuroFilter(frequency=30, min_cutoff=1.0, beta=0.0)
    x = np.array([1.0, 2.0, 3.0])
    result = f.filter(x)
    np.testing.assert_array_equal(result, x)


def test_smooths_noisy_signal():
    f = OneEuroFilter(frequency=30, min_cutoff=0.05, beta=0.0)

    # Feed a constant signal with noise
    values = []
    for i in range(30):
        noisy = np.array([1.0 + np.random.normal(0, 0.1)])
        values.append(f.filter(noisy)[0])

    # After 30 frames, the smoothed value should be close to 1.0
    assert abs(values[-1] - 1.0) < 0.15


def test_responds_to_fast_movement():
    f = OneEuroFilter(frequency=30, min_cutoff=0.05, beta=40.0)

    # Start at 0
    f.filter(np.array([0.0]))

    # Jump to 10 (fast movement)
    for _ in range(5):
        result = f.filter(np.array([10.0]))

    # With high beta, should track the jump quickly
    assert result[0] > 8.0


def test_reset_clears_state():
    f = OneEuroFilter(frequency=30, min_cutoff=0.05, beta=0.0)
    f.filter(np.array([100.0]))
    f.reset()
    # After reset, next value should pass through unfiltered
    result = f.filter(np.array([5.0]))
    np.testing.assert_array_equal(result, np.array([5.0]))


def test_landmark_smoother_both_hands():
    smoother = LandmarkSmoother(frequency=30, min_cutoff=1.0, beta=0.0)
    left = np.random.rand(21, 3).astype(np.float32)
    right = np.random.rand(21, 3).astype(np.float32)

    sl, sr = smoother.smooth(left, right)
    assert sl is not None
    assert sr is not None
    assert sl.shape == (21, 3)
    assert sr.shape == (21, 3)


def test_landmark_smoother_one_hand_none():
    smoother = LandmarkSmoother(frequency=30, min_cutoff=1.0, beta=0.0)
    left = np.random.rand(21, 3).astype(np.float32)

    sl, sr = smoother.smooth(left, None)
    assert sl is not None
    assert sr is None


def test_landmark_smoother_resets_on_none():
    smoother = LandmarkSmoother(frequency=30, min_cutoff=0.05, beta=0.0)

    # Track a hand
    pts = np.ones((21, 3), dtype=np.float32)
    smoother.smooth(pts, None)
    smoother.smooth(pts, None)

    # Hand disappears and reappears at different position
    smoother.smooth(None, None)
    new_pts = np.ones((21, 3), dtype=np.float32) * 5.0
    sl, _ = smoother.smooth(new_pts, None)

    # After reset, new position should pass through (no lingering state from old position)
    np.testing.assert_array_almost_equal(sl, new_pts)
