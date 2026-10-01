"""
camera/camera_manager.py

Thin, swappable wrapper around cv2.VideoCapture. Isolated so the rest
of the pipeline never touches OpenCV capture APIs directly -- this is
what lets the camera be replaced (e.g. a different capture backend)
without touching face/gaze code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config import CameraConfig


@dataclass
class FrameResult:
    ok: bool
    frame_bgr: Optional[np.ndarray]
    timestamp: float
    frame_index: int


class CameraManager:
    """Manages webcam acquisition and basic resizing for processing."""

    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._cfg.device_index)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.capture_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.capture_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.target_fps)
        return True

    def read(self) -> FrameResult:
        if self._cap is None:
            raise RuntimeError("CameraManager.open() must be called first")
        ok, frame = self._cap.read()
        ts = time.time()
        if not ok:
            return FrameResult(False, None, ts, self._frame_index)
        self._frame_index += 1
        return FrameResult(True, frame, ts, self._frame_index)

    def processing_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return a resized copy for landmark inference plus the scale
        factor needed to map processing-space pixels back to the
        original captured frame's pixel space."""
        h, w = frame_bgr.shape[:2]
        target_w = self._cfg.processing_width
        if w <= target_w:
            return frame_bgr, 1.0
        scale = target_w / float(w)
        target_h = int(round(h * scale))
        resized = cv2.resize(frame_bgr, (target_w, target_h))
        return resized, 1.0 / scale

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
