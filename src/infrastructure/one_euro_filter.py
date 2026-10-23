from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    """One Euro Filter for real-time signal smoothing.

    Adaptive low-pass filter that smooths heavily when still and responds
    quickly during fast movement. Parameters from Google's BlazePose production
    config, tuned for 30fps hand landmark tracking.

    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(
        self,
        frequency: float = 30.0,
        min_cutoff: float = 0.05,
        beta: float = 40.0,
        derivate_cutoff: float = 1.0,
    ):
        self._frequency = frequency
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._derivate_cutoff = derivate_cutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self._frequency
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: np.ndarray) -> np.ndarray:
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x.copy()

        # Estimate derivative
        dx = (x - self._x_prev) * self._frequency
        a_d = self._alpha(self._derivate_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        # Adaptive cutoff based on speed
        speed = np.abs(dx_hat)
        cutoff = self._min_cutoff + self._beta * speed

        # Per-element alpha based on adaptive cutoff
        te = 1.0 / self._frequency
        tau = 1.0 / (2.0 * np.pi * cutoff)
        alpha = 1.0 / (1.0 + tau / te)

        x_hat = alpha * x + (1 - alpha) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat

        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None


class LandmarkSmoother:
    """Applies One Euro filtering to hand landmarks (21 points x 3 coords per hand)."""

    def __init__(self, **filter_kwargs):
        self._left_filter = OneEuroFilter(**filter_kwargs)
        self._right_filter = OneEuroFilter(**filter_kwargs)

    def smooth(
        self, left: np.ndarray | None, right: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        smoothed_left = None
        smoothed_right = None

        if left is not None:
            flat = left.reshape(-1)
            smoothed = self._left_filter.filter(flat)
            smoothed_left = smoothed.reshape(left.shape)
        else:
            self._left_filter.reset()

        if right is not None:
            flat = right.reshape(-1)
            smoothed = self._right_filter.filter(flat)
            smoothed_right = smoothed.reshape(right.shape)
        else:
            self._right_filter.reset()

        return smoothed_left, smoothed_right
