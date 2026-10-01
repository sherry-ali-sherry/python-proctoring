"""
detection/object_detector.py

Finds phones, books and people in video frames with Google's MediaPipe
ObjectDetector and an EfficientDet-Lite model trained on the COCO dataset
(Apache-2.0, runs offline, same MediaPipe package the face tracker uses).

COCO has the three classes ExamVision needs: "cell phone", "book" and
"person". The model file (~12 MB for EfficientDet-Lite2) is downloaded once
into ~/.cache/examvision/, like the face model, or can be placed there by
hand.

Measured on the project's test material (2026-09-30):
  * phones held in the hand in stock photos: found in 5 of 6 (0.48-0.86);
    the miss was a phone held flat and edge-on.
  * books being read: found in all 4 (0.47-0.63). Books on a shelf in the
    background are also found (0.2-0.35), which is why books need a higher
    score and why books that never move are reported separately
    (incidents.py) instead of flagged.
  * the six recorded test videos (no phone or book in them): no phone or
    book above 0.25 in any sampled frame; one person each.
  * ~110 ms per frame on a laptop CPU, so frames are sampled (2 per second
    by default) rather than every frame analysed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlretrieve

import cv2
import numpy as np

PHONE = "cell phone"
BOOK = "book"
PERSON = "person"
LABELS = (PHONE, BOOK, PERSON)

_MODEL_URLS = {
    "efficientdet_lite0": "https://storage.googleapis.com/mediapipe-models/object_detector/"
                          "efficientdet_lite0/float16/latest/efficientdet_lite0.tflite",
    "efficientdet_lite2": "https://storage.googleapis.com/mediapipe-models/object_detector/"
                          "efficientdet_lite2/float16/latest/efficientdet_lite2.tflite",
}
_MODEL_CACHE_DIR = Path.home() / ".cache" / "examvision"


@dataclass(frozen=True)
class DetectedObject:
    label: str                                 # PHONE | BOOK | PERSON
    score: float
    box: Tuple[float, float, float, float]     # x, y, w, h as fractions of the frame

    @property
    def area(self) -> float:
        return self.box[2] * self.box[3]


def model_path(model: str) -> Path:
    return _MODEL_CACHE_DIR / f"{model}.tflite"


def ensure_model(model: str) -> str:
    """Local path of the detector model, downloading it on first use."""
    if model not in _MODEL_URLS:
        raise ValueError(f"Unknown object model {model!r}; choose one of {sorted(_MODEL_URLS)}")
    path = model_path(model)
    if path.exists() and path.stat().st_size > 0:
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tflite.part")
    url = _MODEL_URLS[model]
    print(f"[ObjectDetector] Downloading {model} to {path} ...")
    try:
        urlretrieve(url, tmp)
    except (URLError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Phone/book/person detection needs the {model} model but it could not be downloaded "
            f"from {url} ({e}). Check the internet connection, or download that file and save it "
            f"as {path}.") from e
    tmp.replace(path)
    print("[ObjectDetector] Model download complete.")
    return str(path)


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class ObjectDetector:
    """Thin wrapper; the only file that imports the MediaPipe object API."""

    def __init__(self, model: str = "efficientdet_lite2", min_score: float = 0.2, max_results: int = 12):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import ObjectDetector as _MPDetector
        from mediapipe.tasks.python.vision import ObjectDetectorOptions, RunningMode
        import mediapipe as mp

        self._mp = mp
        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=ensure_model(model)),
            running_mode=RunningMode.IMAGE,
            score_threshold=min_score,
            max_results=max_results,
            category_allowlist=list(LABELS),
        )
        self._detector = _MPDetector.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedObject]:
        h, w = frame_bgr.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        out: List[DetectedObject] = []
        for d in self._detector.detect(image).detections:
            if not d.categories:
                continue
            c = d.categories[0]
            b = d.bounding_box
            out.append(DetectedObject(c.category_name, float(c.score),
                                      (b.origin_x / w, b.origin_y / h, b.width / w, b.height / h)))
        return out

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def count_people(objects: List[DetectedObject], min_score: float, min_area: float) -> int:
    """Distinct people: confident, not tiny, and not a duplicate box of the
    same person (overlap > 0.5)."""
    people = sorted((o for o in objects if o.label == PERSON and o.score >= min_score and o.area >= min_area),
                    key=lambda o: -o.score)
    kept: List[DetectedObject] = []
    for p in people:
        if all(iou(p.box, k.box) <= 0.5 for k in kept):
            kept.append(p)
    return len(kept)
