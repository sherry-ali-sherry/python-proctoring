"""
face/face_tracker.py

FaceTracker wraps MediaPipe's current Tasks API (`FaceLandmarker`) — the
legacy `mp.solutions.face_mesh` API this module used previously was removed
from the mediapipe PyPI package in 0.10.30+ (solutions.face_mesh no longer
exists; face_mesh.FaceMesh is gone). FaceLandmarker uses the same 478-point
mesh topology (468 face points + 10 iris points, indices 468-477) as the old
`refine_landmarks=True` FaceMesh, so nothing downstream — geometry, gaze,
proctoring — needs to change; this is the only file that imports mediapipe.

The task requires a model bundle file (`face_landmarker.task`) rather than
being bundled in the package. `_ensure_model` downloads it once to a local
cache directory the first time FaceTracker runs and reuses the cached copy
on every run after that, so normal operation is fully offline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from config import (
    ConfidenceConfig,
    FACE_MODEL_LANDMARK_IDS,
    LEFT_IRIS_RING,
    RIGHT_IRIS_RING,
    LEFT_EYE_LID,
    RIGHT_EYE_LID,
)

# Official model bundle, hosted by Google; "latest" always resolves to the
# newest float16 build of the face_landmarker model.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_CACHE_DIR = Path.home() / ".cache" / "examvision"
_MODEL_PATH = _MODEL_CACHE_DIR / "face_landmarker.task"

# Legacy FaceMesh(refine_landmarks=True) / current FaceLandmarker both use
# this many points: 0-467 face mesh, 468-477 left+right iris rings.
_EXPECTED_LANDMARK_COUNT = 478


def _ensure_model(path: Path = _MODEL_PATH, url: str = _MODEL_URL) -> str:
    """Return a local path to the FaceLandmarker model bundle, downloading
    it on first use. Subsequent calls (including future process runs) reuse
    the cached file and never touch the network."""
    if path.exists() and path.stat().st_size > 0:
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".task.part")
    print(f"[FaceTracker] Downloading FaceLandmarker model to {path} ...")
    try:
        urlretrieve(url, tmp_path)
    except (URLError, OSError) as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "FaceTracker needs the MediaPipe FaceLandmarker model bundle "
            f"but could not download it from {url} ({e}). Check your "
            f"internet connection, or manually download that file and "
            f"place it at {path}."
        ) from e
    tmp_path.replace(path)
    print("[FaceTracker] Model download complete.")
    return str(path)


@dataclass
class FaceObservation:
    """Everything downstream code needs about one detected face, for
    one frame, in normalized image coordinates (x, y in [0, 1])."""

    face_confidence: float
    landmarks_norm: np.ndarray          # (468 or 478, 3) x,y in [0,1], z relative
    image_width: int
    image_height: int
    # MediaPipe face blendshape scores in [0, 1] (e.g. "eyeLookOutLeft",
    # "eyeBlinkRight"); empty if the model did not return them.
    blendshapes: Dict[str, float] = field(default_factory=dict)

    def px(self, index: int) -> np.ndarray:
        """Landmark pixel position (x, y) for a single index."""
        lm = self.landmarks_norm[index]
        return np.array([lm[0] * self.image_width, lm[1] * self.image_height])

    def px_many(self, indices: List[int]) -> np.ndarray:
        return np.array([self.px(i) for i in indices])

    def stable_reference_px(self) -> dict:
        return {name: self.px(idx) for name, idx in FACE_MODEL_LANDMARK_IDS.items()}


@dataclass
class FaceTrackerResult:
    face_detected: bool
    num_faces: int
    faces: List[FaceObservation] = field(default_factory=list)


class FaceTracker:
    """Face landmark tracker backed by MediaPipe's FaceLandmarker (Tasks API).

    Public interface (constructor, `process`, `close`) is unchanged from
    the previous implementation, so nothing downstream needs to change.
    """

    def __init__(
        self,
        confidence_cfg: ConfidenceConfig,
        max_num_faces: int = 2,
        model_path: Optional[str] = None,
    ):
        self._cfg = confidence_cfg
        self._max_num_faces = max_num_faces

        resolved_model_path = model_path or _ensure_model()

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolved_model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=confidence_cfg.min_face_detection_confidence,
            min_face_presence_confidence=confidence_cfg.min_face_presence_confidence,
            min_tracking_confidence=confidence_cfg.min_tracking_confidence,
            # Blendshapes include learned eye-direction scores (eyeLookIn/Out/
            # Up/Down per eye). They need no calibration and, unlike the 3D
            # eye-geometry rays, respond reliably to reading beside or below
            # the screen -- see gaze/eye_direction.py.
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

        # VIDEO running mode requires strictly increasing timestamps per
        # call; monotonic clock avoids issues with wall-clock adjustments.
        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1
        print("[FaceTracker] Using MediaPipe FaceLandmarker (Tasks API)")

    def process(self, frame_rgb: np.ndarray, timestamp_ms: Optional[int] = None) -> FaceTrackerResult:
        """`timestamp_ms` is the frame's own media time (video files);
        when omitted, elapsed wall-clock time is used (live webcam)."""
        h, w = frame_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        if timestamp_ms is None:
            timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return FaceTrackerResult(face_detected=False, num_faces=0, faces=[])

        faces = []
        all_blendshapes = result.face_blendshapes or []
        for k, face_landmarks in enumerate(result.face_landmarks):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_landmarks],
                dtype=np.float64,
            )
            face_confidence = 1.0 if coords.shape[0] >= 468 else 0.0
            blendshapes = (
                {c.category_name: float(c.score) for c in all_blendshapes[k]}
                if k < len(all_blendshapes) else {}
            )
            faces.append(
                FaceObservation(
                    face_confidence=face_confidence,
                    landmarks_norm=coords,
                    image_width=w,
                    image_height=h,
                    blendshapes=blendshapes,
                )
            )
        return FaceTrackerResult(face_detected=True, num_faces=len(faces), faces=faces)

    @staticmethod
    def left_eye_landmarks(face: FaceObservation) -> dict:
        return {
            "outer": face.px(33),
            "inner": face.px(133),
            "lid_top": face.px(LEFT_EYE_LID["top"]),
            "lid_bottom": face.px(LEFT_EYE_LID["bottom"]),
        }

    @staticmethod
    def right_eye_landmarks(face: FaceObservation) -> dict:
        return {
            "outer": face.px(263),
            "inner": face.px(362),
            "lid_top": face.px(RIGHT_EYE_LID["top"]),
            "lid_bottom": face.px(RIGHT_EYE_LID["bottom"]),
        }

    @staticmethod
    def iris_landmarks(face: FaceObservation) -> dict:
        return {
            "left_ring": face.px_many(LEFT_IRIS_RING),
            "right_ring": face.px_many(RIGHT_IRIS_RING),
        }

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
