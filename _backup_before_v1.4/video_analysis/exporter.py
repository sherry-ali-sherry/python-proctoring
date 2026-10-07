"""
video_analysis/exporter.py

Writes the visual evidence, in a single sequential pass over the
original (full-resolution) video:

    recordings/incident_NN_at_HHhMMmSSs_to_HHhMMmSSs.mp4
        the incident plus `clip_padding_s` of context on both sides,
        with a banner: red while the candidate is not looking at the
        screen, grey for the surrounding context.
    snapshots/incident_NN_at_HHhMMmSSs.jpg
        the middle frame of the incident, with the same banner.
    data/review_video.mp4                 (export_review_video)
        the whole video, unannotated, downscaled, for the web player.
    recordings/full_video_annotated.mp4   (export_annotated_video)

All videos are H.264 MP4 when possible (see video_writer.py), so they
play in a browser as well as in desktop players.

Frames are read sequentially rather than by seeking because seeking is
frame-inaccurate for many codecs; the reader numbers frames exactly as
it did during analysis, so incident frame indices line up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import AppConfig
from proctoring.state_manager import ProctorState
from video_analysis.incidents import (
    REASON_BOOK, REASON_PEOPLE, REASON_PHONE, FrameRecord, Incident, timecode,
)
from video_analysis.output_layout import ResultPaths
from video_analysis.video_source import VideoFrameReader, VideoInfo
from video_analysis.video_writer import VideoWriter

_RED = (40, 40, 210)
_GREEN = (60, 150, 40)
_GREY = (90, 90, 90)
_AMBER = (0, 150, 220)


@dataclass
class ExportResult:
    warnings: List[str] = field(default_factory=list)
    review_video: Optional[str] = None        # relative to the results folder
    browser_playable: bool = True             # all videos are H.264


def _file_timecode(seconds: float) -> str:
    total = int(max(seconds, 0.0))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def draw_banner(frame: np.ndarray, title: str, detail: str, color: Tuple[int, int, int]) -> np.ndarray:
    """Solid banner across the top of a copy of `frame`."""
    img = frame.copy()
    h, w = img.shape[:2]
    bar_h = max(40, int(h * 0.085))
    cv2.rectangle(img, (0, 0), (w, bar_h), color, -1)
    scale = max(0.45, min(1.2, w / 1100.0))
    thick = 2 if scale >= 0.7 else 1
    cv2.putText(img, title, (12, int(bar_h * 0.45)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thick + 1, cv2.LINE_AA)
    cv2.putText(img, detail, (12, int(bar_h * 0.85)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_jpeg(path: Path, img: np.ndarray) -> bool:
    # imencode + tofile handles non-ASCII paths, unlike cv2.imwrite.
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return False
    buf.tofile(str(path))
    return path.is_file() and path.stat().st_size > 0


@dataclass
class _ClipPlan:
    incident: Incident
    first: int
    last: int
    snapshot_frame: int
    base_path: Path
    writer: Optional[VideoWriter] = None


_OBJECT_TITLES = {REASON_PHONE: "PHONE VISIBLE", REASON_BOOK: "BOOK VISIBLE", REASON_PEOPLE: "MORE THAN ONE PERSON"}
_BOX_COLORS = {"cell phone": (40, 40, 230), "book": (0, 165, 255), "person": (200, 200, 200)}
_BOX_NAMES = {"cell phone": "phone", "book": "book", "person": "person"}


def incident_title(inc: Incident) -> str:
    if inc.is_object:
        return " + ".join(_OBJECT_TITLES[r] for r in inc.reasons if r in _OBJECT_TITLES)
    return "NOT LOOKING AT SCREEN"


def draw_objects(frame: np.ndarray, objects) -> np.ndarray:
    """Boxes around the phones, books and people found on this frame
    (normalized boxes from detection/object_detector.py)."""
    if not objects:
        return frame
    img = frame.copy()
    h, w = img.shape[:2]
    thick = max(2, w // 400)
    for o in objects:
        x, y, bw, bh = o.box
        p1 = (int(x * w), int(y * h))
        p2 = (int((x + bw) * w), int((y + bh) * h))
        color = _BOX_COLORS.get(o.label, (255, 255, 255))
        cv2.rectangle(img, p1, p2, color, thick if o.label != "person" else 1)
        label = f"{_BOX_NAMES.get(o.label, o.label)} {o.score:.0%}"
        cv2.putText(img, label, (p1[0] + 4, max(p1[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.45, w / 1800), color, max(1, thick - 1), cv2.LINE_AA)
    return img


def _incident_frame(frame: np.ndarray, idx: int, record: FrameRecord, inc: Incident) -> np.ndarray:
    tc = timecode(record.time_s)
    frame = draw_objects(frame, record.objects)
    if inc.start_frame <= idx <= inc.end_frame:
        detail = f"{tc}   {inc.reason_text()}" + (f"   ({inc.direction})" if inc.direction else "")
        return draw_banner(frame, f"{incident_title(inc)}  -  Incident #{inc.number}", detail, _RED)
    when = "before" if idx < inc.start_frame else "after"
    return draw_banner(frame, f"Context {when} incident #{inc.number}", tc, _GREY)


def _full_video_banner(record: Optional[FrameRecord], titles: Optional[Dict[int, str]] = None
                       ) -> Tuple[str, str, Tuple[int, int, int]]:
    if record is None:
        return "NOT ANALYZED", "", _GREY
    tc = timecode(record.time_s)
    if record.incident_number is not None:
        title = (titles or {}).get(record.incident_number, "NOT LOOKING AT SCREEN")
        return title, f"Incident #{record.incident_number}   {tc}", _RED
    if record.num_faces == 0:
        return "NO FACE VISIBLE", tc, _GREY
    if record.eyes_away or record.state in (ProctorState.POSSIBLE_AWAY, ProctorState.LOOKING_AWAY,
                                            ProctorState.POSSIBLE_SCREEN):
        return "GLANCING AWAY (below time limit)", tc, _AMBER
    if record.state == ProctorState.LOOKING_AT_SCREEN or record.eye_x is not None:
        return "LOOKING AT SCREEN", tc, _GREEN
    return "GAZE UNCERTAIN", tc, _GREY


class _OptionalWriter:
    """A full-length writer that is opened lazily on the first frame and
    disables itself (with a warning) if it cannot be created."""

    def __init__(self, base_path: Path, fps: float, max_height: Optional[int], label: str, warnings: List[str]):
        self._args = (base_path, fps, max_height)
        self._label = label
        self._warnings = warnings
        self.writer: Optional[VideoWriter] = None
        self.enabled = True

    def write(self, img: np.ndarray) -> None:
        if not self.enabled:
            return
        if self.writer is None:
            base_path, fps, max_height = self._args
            self.writer = VideoWriter(base_path, fps, (img.shape[1], img.shape[0]), max_height)
            if not self.writer.ok:
                self._warnings.append(f"Could not create the {self._label}.")
                self.enabled = False
                return
        self.writer.write(img)

    def close(self) -> Optional[Path]:
        if not self.enabled or self.writer is None:
            return None
        final = self.writer.close()
        if final is None:
            self._warnings.append(f"The {self._label} came out empty.")
        return final

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.abort()


def export_media(
    info: VideoInfo,
    incidents: Sequence[Incident],
    records: Sequence[FrameRecord],
    paths: ResultPaths,
    cfg: AppConfig,
    progress: Callable[[float, str], None],
    check_cancel: Callable[[], None],
) -> ExportResult:
    """Writes clips/snapshots/review video, fills in each incident's
    clip_file and snapshot_file, and returns warnings + file locations."""
    result = ExportResult()
    warnings = result.warnings
    total_frames = len(records)
    vcfg = cfg.video
    if total_frames == 0 or (not incidents and not vcfg.export_annotated_video and not vcfg.export_review_video):
        return result

    fps = info.fps
    pad = int(round(vcfg.clip_padding_s * fps))
    plans: List[_ClipPlan] = []
    for inc in incidents:
        name = f"incident_{inc.number:02d}_at_{_file_timecode(inc.start_s)}_to_{_file_timecode(inc.end_s)}"
        plans.append(_ClipPlan(
            incident=inc,
            first=max(0, inc.start_frame - pad),
            last=min(total_frames - 1, inc.end_frame + pad),
            snapshot_frame=(inc.start_frame + inc.end_frame) // 2,
            base_path=paths.recordings / name,
        ))
    starts: Dict[int, List[_ClipPlan]] = {}
    snapshots: Dict[int, List[_ClipPlan]] = {}
    for plan in plans:
        starts.setdefault(plan.first, []).append(plan)
        snapshots.setdefault(plan.snapshot_frame, []).append(plan)
    full_pass = vcfg.export_annotated_video or vcfg.export_review_video
    last_needed = total_frames - 1 if full_pass else max(p.last for p in plans)

    titles = {inc.number: incident_title(inc) for inc in incidents}
    annotated = (_OptionalWriter(paths.recordings / "full_video_annotated", fps, None,
                                 "annotated full-length video", warnings)
                 if vcfg.export_annotated_video else None)
    review = (_OptionalWriter(paths.data / "review_video", fps, vcfg.review_video_max_height,
                              "review video for the web player", warnings)
              if vcfg.export_review_video else None)

    active: List[_ClipPlan] = []
    try:
        with VideoFrameReader(info) as reader:
            for idx, _t, frame in reader.frames():
                check_cancel()
                if idx >= total_frames:
                    break
                record = records[idx]
                size = (frame.shape[1], frame.shape[0])

                for plan in starts.get(idx, []):
                    plan.writer = VideoWriter(plan.base_path, fps, size)
                    if plan.writer.ok:
                        active.append(plan)
                        result.browser_playable &= plan.writer.browser_playable
                    else:
                        warnings.append(f"Could not create a video file for incident #{plan.incident.number}.")

                for plan in active:
                    plan.writer.write(_incident_frame(frame, idx, record, plan.incident))

                for plan in snapshots.get(idx, []):
                    inc = plan.incident
                    snap = paths.snapshots / f"incident_{inc.number:02d}_at_{_file_timecode(record.time_s)}.jpg"
                    if write_jpeg(snap, _incident_frame(frame, idx, record, inc)):
                        inc.snapshot_file = paths.relative(snap)
                    else:
                        warnings.append(f"Could not save the snapshot for incident #{inc.number}.")

                for plan in [p for p in active if p.last == idx]:
                    active.remove(plan)
                    final = plan.writer.close()
                    if final is not None:
                        plan.incident.clip_file = paths.relative(final)
                    else:
                        warnings.append(f"The video clip for incident #{plan.incident.number} came out empty.")

                if annotated is not None:
                    annotated.write(draw_banner(draw_objects(frame, record.objects if record else ()),
                                                *_full_video_banner(record, titles)))
                if review is not None:
                    review.write(frame)

                if idx % 15 == 0:
                    progress(min(1.0, idx / max(last_needed, 1)), f"Exporting video: frame {idx + 1} of {last_needed + 1}")
                if idx >= last_needed:
                    break
    except BaseException:
        for plan in active:
            plan.writer.abort()
        for w in (annotated, review):
            if w is not None:
                w.abort()
        raise

    # The decoder stopped early (shorter than during analysis): finish
    # whatever was started so partial evidence is not lost.
    for plan in active:
        final = plan.writer.close()
        if final is not None:
            plan.incident.clip_file = paths.relative(final)
        warnings.append(f"The video ended early while exporting incident #{plan.incident.number}; its clip may be shorter.")
    if annotated is not None:
        annotated.close()
    if review is not None:
        final = review.close()
        if final is not None:
            result.review_video = paths.relative(final)
            result.browser_playable &= review.writer.browser_playable

    for plan in plans:
        if plan.writer is None:
            warnings.append(f"Incident #{plan.incident.number} was never reached while exporting clips.")
    if not result.browser_playable:
        warnings.append("H.264 encoding was unavailable, so videos were saved in a format some browsers "
                        "cannot play. They still open in desktop players (e.g. VLC).")
    progress(1.0, "Video exported")
    return result
