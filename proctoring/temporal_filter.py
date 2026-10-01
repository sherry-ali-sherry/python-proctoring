"""
proctoring/temporal_filter.py

Single-frame gaze decisions are noisy (spec section 18). This module
applies an exponential moving average (EMA) to the continuous signals
(screen_x, screen_y, confidence) before they reach the state machine.

Why EMA over a plain rolling window/median here: it needs O(1) memory
and time per frame (important for real-time performance, spec
section 29), naturally weights recent frames more than stale ones, and
its single alpha parameter is easy to expose as a configurable
responsiveness/smoothness trade-off. The *discrete label* coming out
of gaze_classifier is intentionally NOT smoothed directly (averaging
enum labels is meaningless) -- instead the underlying continuous
screen position and confidence are smoothed, and the label for state
machine purposes is re-derived from the smoothed position each frame.
Hysteresis (durations, not just smoothing) is handled separately in
state_manager.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import TemporalConfig


@dataclass
class SmoothedGaze:
    screen_x: Optional[float]
    screen_y: Optional[float]
    confidence: float


class TemporalFilter:
    def __init__(self, cfg: TemporalConfig):
        self._alpha = cfg.ema_alpha
        self._sx: Optional[float] = None
        self._sy: Optional[float] = None
        self._conf: float = 0.0

    def reset(self) -> None:
        self._sx = None
        self._sy = None
        self._conf = 0.0

    def update(
        self, screen_x: Optional[float], screen_y: Optional[float], confidence: float
    ) -> SmoothedGaze:
        a = self._alpha
        self._conf = a * confidence + (1 - a) * self._conf

        if screen_x is None or screen_y is None:
            # Don't let a single invalid-intersection frame yank the
            # smoothed position -- just decay confidence and hold
            # position, so a brief blink/occlusion doesn't cause a
            # position jump when the ray reappears.
            return SmoothedGaze(self._sx, self._sy, self._conf)

        if self._sx is None:
            self._sx, self._sy = screen_x, screen_y
        else:
            self._sx = a * screen_x + (1 - a) * self._sx
            self._sy = a * screen_y + (1 - a) * self._sy

        return SmoothedGaze(self._sx, self._sy, self._conf)
