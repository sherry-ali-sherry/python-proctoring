"""
video_analysis/video_writer.py

Writes browser-playable video (H.264 in MP4, yuv420p, "faststart" so a
browser can start playing before the whole file has downloaded).

OpenCV's own VideoWriter cannot produce H.264 on Windows without extra
DLLs, and its MPEG-4 Part 2 ("mp4v") output does not play in browsers,
which the web review UI needs. PyAV (FFmpeg bindings, bundled with its
wheels) is used instead, trying encoders in order: libx264, then the
Windows Media Foundation / GPU encoders. If none can be opened (PyAV
missing, or a build without H.264), it falls back to OpenCV mp4v so
analysis still produces playable-on-desktop clips.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import av
except ImportError:  # pragma: no cover - PyAV is in requirements.txt
    av = None

_H264_ENCODERS = ("libx264", "h264_mf", "h264_qsv", "h264_amf", "h264_nvenc")


def _even(n: int) -> int:
    return max(2, n - (n % 2))


class VideoWriter:
    """Frame-by-frame video writer. `path_base` has no extension; the
    final file is `path_base + .mp4` (or `.avi` in the MJPG last resort)."""

    def __init__(self, path_base: Path, fps: float, size: Tuple[int, int], max_height: Optional[int] = None):
        w, h = size
        if max_height and h > max_height:
            w, h = int(round(w * max_height / h)), max_height
        self.size = (_even(w), _even(h))
        self.fps = fps
        self.final_path: Optional[Path] = None
        self.browser_playable = False
        self._av = None
        self._stream = None
        self._n = 0
        self._cv: Optional[cv2.VideoWriter] = None
        self._temp_path: Optional[Path] = None
        if not self._open_pyav(path_base):
            self._open_opencv(path_base)

    # -- backends --------------------------------------------------------
    def _open_pyav(self, path_base: Path) -> bool:
        if av is None:
            return False
        final = path_base.parent / (path_base.name + ".mp4")
        rate = Fraction(self.fps).limit_denominator(1001)
        for encoder in _H264_ENCODERS:
            if encoder not in av.codecs_available:
                continue
            container = None
            try:
                container = av.open(str(final), mode="w", container_options={"movflags": "+faststart"})
                stream = container.add_stream(encoder, rate=rate)
                stream.width, stream.height = self.size
                stream.pix_fmt = "yuv420p"
                if encoder == "libx264":
                    stream.options = {"crf": "23", "preset": "veryfast"}
                # Hardware/OS encoders can be listed yet unusable on this
                # machine; opening the codec now surfaces that here.
                stream.codec_context.open()
            except Exception:
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        pass
                final.unlink(missing_ok=True)
                continue
            self._av, self._stream, self.final_path = container, stream, final
            self.browser_playable = True
            return True
        return False

    def _open_opencv(self, path_base: Path) -> None:
        for fourcc, ext in (("mp4v", ".mp4"), ("MJPG", ".avi")):
            final = path_base.parent / (path_base.name + ext)
            writer = cv2.VideoWriter(str(final), cv2.VideoWriter_fourcc(*fourcc), self.fps, self.size)
            if writer.isOpened():
                self._cv, self.final_path = writer, final
                return
            writer.release()
            # OpenCV cannot open some non-ASCII paths on Windows.
            fd, tmp = tempfile.mkstemp(prefix="examvision_clip_", suffix=ext)
            os.close(fd)
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*fourcc), self.fps, self.size)
            if writer.isOpened():
                self._cv, self.final_path, self._temp_path = writer, final, Path(tmp)
                return
            writer.release()
            Path(tmp).unlink(missing_ok=True)

    # -- public API ------------------------------------------------------
    @property
    def ok(self) -> bool:
        return self._av is not None or self._cv is not None

    def write(self, img: np.ndarray) -> None:
        if img.shape[1] != self.size[0] or img.shape[0] != self.size[1]:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        if self._av is not None:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="bgr24")
            frame.pts = self._n
            frame.time_base = self._stream.codec_context.time_base
            self._n += 1
            for packet in self._stream.encode(frame):
                self._av.mux(packet)
        elif self._cv is not None:
            self._cv.write(img)

    def close(self) -> Optional[Path]:
        """Finish the file; returns its path, or None if it came out empty."""
        if self._av is not None:
            try:
                for packet in self._stream.encode():
                    self._av.mux(packet)
            finally:
                self._av.close()
                self._av = None
        elif self._cv is not None:
            self._cv.release()
            self._cv = None
            if self._temp_path is not None:
                shutil.move(str(self._temp_path), str(self.final_path))
                self._temp_path = None
        else:
            return None
        if self.final_path.is_file() and self.final_path.stat().st_size > 0:
            return self.final_path
        return None

    def abort(self) -> None:
        try:
            if self._av is not None:
                self._av.close()
            if self._cv is not None:
                self._cv.release()
        except Exception:
            pass
        self._av = self._cv = None
        if self._temp_path is not None:
            self._temp_path.unlink(missing_ok=True)
        if self.final_path is not None:
            self.final_path.unlink(missing_ok=True)
