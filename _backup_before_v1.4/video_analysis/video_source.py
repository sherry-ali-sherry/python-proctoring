"""
video_analysis/video_source.py

Reads a recorded video file frame by frame, the file-based counterpart
of camera/camera_manager.py. Every frame gets a timestamp in seconds
from the start of the video (media time, not wall-clock time), which is
what the temporal filter / hysteresis / incident timestamps run on.

Timestamps: the container's own presentation time (CAP_PROP_POS_MSEC)
is used when it is available and strictly increasing, which keeps
variable-frame-rate recordings (phones, screen recorders) accurate;
otherwise the time advances by one nominal frame period.

Frame indices count successfully decoded frames from 0. Every pass over
the video uses this same reader, so an index always refers to the same
frame in every pass (analysis, clip export).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

_DEFAULT_FPS = 30.0
_MIN_VALID_FPS = 1.0
_MAX_VALID_FPS = 240.0


class VideoError(Exception):
    """The video could not be opened or decoded."""


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float                 # nominal; _DEFAULT_FPS if the container reports none
    fps_reported: bool
    frame_count: int           # as reported by the container; 0 if unknown
    duration_s: float          # frame_count / fps; 0 if unknown
    codec: str

    def describe(self) -> str:
        duration = f"{self.duration_s:.1f} s" if self.duration_s > 0 else "unknown length"
        return f"{self.width}x{self.height}, {self.fps:.2f} fps, {duration}, codec {self.codec or 'unknown'}"


def _open_capture(path: str) -> cv2.VideoCapture:
    # FFmpeg gives consistent frame timestamps across formats; fall back
    # to OpenCV's default backend choice (e.g. Media Foundation on
    # Windows, which also copes with non-ASCII paths).
    for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
        cap = cv2.VideoCapture(path, backend)
        if cap.isOpened():
            return cap
        cap.release()
    raise VideoError(f"Could not open video file: {path}")


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    if code <= 0:
        return ""
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(c for c in chars if c.isprintable()).strip()


def probe_video(path: str) -> VideoInfo:
    """Open the file, check that at least one frame decodes, and return
    its metadata. Raises VideoError with a user-readable message."""
    p = Path(path)
    if not p.is_file():
        raise VideoError(f"Video file not found: {path}")
    if p.stat().st_size == 0:
        raise VideoError(f"Video file is empty: {path}")

    cap = _open_capture(str(p))
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError("The file opened but no video frame could be decoded. "
                             "It may be corrupt, audio-only, or use an unsupported codec.")
        height, width = frame.shape[:2]
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fps_ok = _MIN_VALID_FPS <= fps_raw <= _MAX_VALID_FPS
        fps = float(fps_raw) if fps_ok else _DEFAULT_FPS
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = frame_count if frame_count > 0 else 0
        return VideoInfo(
            path=str(p), width=width, height=height, fps=fps, fps_reported=fps_ok,
            frame_count=frame_count,
            duration_s=frame_count / fps if frame_count else 0.0,
            codec=_fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
        )
    finally:
        cap.release()


class VideoFrameReader:
    """Iterates (frame_index, time_s, frame_bgr) over a video file.
    Use as a context manager so the capture is always released."""

    def __init__(self, info: VideoInfo):
        self._info = info
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "VideoFrameReader":
        self._cap = _open_capture(self._info.path)
        return self

    def __exit__(self, *exc) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("VideoFrameReader must be used as a context manager")
        period = 1.0 / self._info.fps
        index = 0
        last_t: Optional[float] = None
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return
            pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            t = pos_ms / 1000.0 if pos_ms is not None and pos_ms >= 0 else -1.0
            if last_t is None:
                t = max(t, 0.0)
            elif not (t > last_t) or t - last_t > 10.0 * period + 1.0:
                # Missing, repeated, or implausibly jumping container time.
                t = last_t + period
            last_t = t
            yield index, t, frame
            index += 1
