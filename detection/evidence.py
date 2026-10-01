"""
detection/evidence.py

Turns the sampled object detections of a whole video into per-frame
evidence: is a phone visible, is a book visible, how many people are in
the frame. Pure functions; no MediaPipe or OpenCV here.

Detections exist only on sampled frames (a few per second). Each sample's
result is held for the frames up to the next sample, but never longer than
`hold_s`, so a gap in sampling cannot stretch evidence.

Books that stay in the same place for most of the video (a bookshelf
behind the candidate) are treated as background: they are left out of the
evidence and returned as a note for the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from detection.object_detector import BOOK, PHONE, DetectedObject, count_people, iou


@dataclass
class FrameEvidence:
    phone: Optional[float] = None       # best phone score, if a phone counts on this frame
    book: Optional[float] = None        # best book score, if a book counts on this frame
    people: int = 0                     # people seen by the object detector
    objects: tuple = ()                 # detections to draw on clips (held sample)


@dataclass
class EvidenceResult:
    frames: List[FrameEvidence]
    samples: int
    notes: List[str] = field(default_factory=list)


def _static_book_boxes(samples: Sequence[List[DetectedObject]], min_score: float, fraction: float) -> list:
    """Book boxes present (same place, IoU > 0.5) in >= `fraction` of samples."""
    boxes = [o.box for s in samples for o in s if o.label == BOOK and o.score >= min_score]
    static: list = []
    n = len(samples)
    if not n:
        return static
    for box in boxes:
        if any(iou(box, s) > 0.5 for s in static):
            continue
        hits = sum(1 for s in samples if any(o.label == BOOK and o.score >= min_score and iou(o.box, box) > 0.5 for o in s))
        if hits / n >= fraction:
            static.append(box)
    return static


def build_evidence(times: Sequence[float], sampled: Sequence[Optional[List[DetectedObject]]], cfg) -> EvidenceResult:
    """times: per-frame video time. sampled: per frame, the detections if
    the frame was sampled, else None. cfg: ObjectDetectionConfig."""
    samples = [s for s in sampled if s is not None]
    static_books = _static_book_boxes(samples, cfg.book_min_score, cfg.static_book_fraction) if len(samples) >= 10 else []
    notes = []
    if static_books:
        notes.append(f"{len(static_books)} book(s) stayed in the same place for most of the video (for example "
                     f"on a shelf behind the candidate). They were treated as background and not flagged; "
                     f"check the video if they were within reach.")

    hold_s = 1.5 / max(cfg.samples_per_second, 0.1)
    frames: List[FrameEvidence] = []
    current: Optional[FrameEvidence] = None
    current_t = None
    for t, s in zip(times, sampled):
        if s is not None:
            phones = [o.score for o in s if o.label == PHONE and o.score >= cfg.phone_min_score]
            books = [o.score for o in s if o.label == BOOK and o.score >= cfg.book_min_score
                     and not any(iou(o.box, b) > 0.5 for b in static_books)]
            shown = tuple(o for o in s if (o.label == PHONE and o.score >= cfg.phone_min_score)
                          or (o.label == BOOK and o.score >= cfg.book_min_score)
                          or (o.label == "person" and o.score >= cfg.person_min_score))
            current = FrameEvidence(max(phones) if phones else None, max(books) if books else None,
                                    count_people(s, cfg.person_min_score, cfg.person_min_area), shown)
            current_t = t
        if current is not None and current_t is not None and t - current_t <= hold_s:
            frames.append(current)
        else:
            frames.append(FrameEvidence())
    return EvidenceResult(frames, len(samples), notes)
