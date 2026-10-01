"""
video_analysis/incidents.py

Turns the per-frame decisions of a whole video into "not looking at the
screen" incidents with start/end timestamps, and into the final verdict.
Pure functions over plain records: no video, OpenCV, or MediaPipe here.

Three kinds of evidence, each gated by the existing persistence
durations in TemporalConfig so a single noisy frame never produces an
incident:

  GAZE_OFF_SCREEN   The hysteresis state machine (GazeStateManager)
                    confirmed LOOKING_AWAY. The incident starts where the
                    gaze first left the screen (the start of the
                    POSSIBLE_AWAY run that led to the confirmation, not
                    the later confirmation instant) and ends at the last
                    LOOKING_AWAY frame (trailing POSSIBLE_SCREEN frames
                    are the gaze already back on the screen).
  FACE_NOT_VISIBLE  No face in frame for >= seconds_to_confirm_face_lost.
  EYES_TURNED_AWAY  The eyes are held away from the candidate's own
                    screen-reading position in any of eight directions,
                    diagonals included (MediaPipe eye-direction scores,
                    gaze/eye_direction.py), for >= seconds_to_confirm_away,
                    where glances back shorter than merge_gap_s do not
                    restart the count. This is the reading-notes pattern:
                    eyes on the notes with brief looks back at the screen.
  HEAD_TURNED_AWAY  Head rotated beyond the head-turn thresholds, relative
                    to the candidate's own median pose, for >=
                    seconds_to_confirm_away. Covers turns so large that
                    the eyes can no longer be measured.

Overlapping or nearly adjacent intervals (gap <= merge_gap_s) are merged
into one incident that lists every reason that applied.
"""
from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from config import AppConfig
from gaze.directions import direction_label
from proctoring.state_manager import ProctorState

REASON_GAZE = "GAZE_OFF_SCREEN"
REASON_EYES = "EYES_TURNED_AWAY"
REASON_FACE = "FACE_NOT_VISIBLE"
REASON_HEAD = "HEAD_TURNED_AWAY"

REASON_TEXT = {
    REASON_GAZE: "Gaze off the screen",
    REASON_EYES: "Eyes turned away from the screen",
    REASON_FACE: "Face not visible to the camera",
    REASON_HEAD: "Head turned away from the screen",
}

VERDICT_CHEATING = "CHEATING"
VERDICT_NO_CHEATING = "NO CHEATING"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"

_AWAY_LIKE = (ProctorState.POSSIBLE_AWAY, ProctorState.LOOKING_AWAY, ProctorState.POSSIBLE_SCREEN)


@dataclass(slots=True)
class FrameRecord:
    """One analyzed video frame (the fields incident logic needs, plus
    what the frame-level CSV log reports)."""

    index: int
    time_s: float
    num_faces: int
    state: ProctorState
    away_direction: Optional[str] = None
    head_turned: bool = False
    head_yaw_offset: Optional[float] = None     # vs. the candidate's median pose
    head_pitch_offset: Optional[float] = None
    gaze_confidence: Optional[float] = None
    screen_x: Optional[float] = None
    screen_y: Optional[float] = None
    smoothed_x: Optional[float] = None
    smoothed_y: Optional[float] = None
    smoothed_confidence: Optional[float] = None
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    head_roll: Optional[float] = None
    eye_x: Optional[float] = None               # blendshape eye direction (see gaze/eye_direction.py)
    eye_up: Optional[float] = None
    eye_down: Optional[float] = None
    eye_blink: Optional[float] = None
    eyes_away: Optional[str] = None             # one of gaze.directions.DIRECTIONS when eyes are away this frame
    incident_number: Optional[int] = None


@dataclass
class Incident:
    number: int
    start_s: float
    end_s: float
    start_frame: int            # inclusive video frame indices
    end_frame: int
    reasons: List[str]
    direction: Optional[str]
    clip_file: Optional[str] = None       # relative to the results folder
    snapshot_file: Optional[str] = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def reason_text(self) -> str:
        return "; ".join(REASON_TEXT[r] for r in self.reasons)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "start_timecode": timecode(self.start_s),
            "end_timecode": timecode(self.end_s),
            "duration_s": round(self.duration_s, 3),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "reasons": list(self.reasons),
            "reason_text": self.reason_text(),
            "direction": self.direction,
            "clip_file": self.clip_file,
            "snapshot_file": self.snapshot_file,
        }


@dataclass
class AnalysisStats:
    frames_analyzed: int
    analyzed_duration_s: float
    frames_with_face: int
    frames_gaze_measured: int
    gaze_coverage: float                # gaze measured / frames with a face
    multiple_faces_s: float
    incident_count: int
    total_not_looking_s: float
    not_looking_fraction: float
    longest_incident_s: float

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class Verdict:
    label: str
    reasons: List[str] = field(default_factory=list)

    @property
    def is_cheating(self) -> bool:
        return self.label == VERDICT_CHEATING


def timecode(seconds: float, sep: str = ":") -> str:
    """HH:MM:SS.mmm (sep='-' gives a filename-safe variant)."""
    ms_total = int(round(max(seconds, 0.0) * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}{sep}{m:02d}{sep}{s:02d}.{ms:03d}"


def frame_period(times: Sequence[float], nominal_fps: float) -> float:
    if len(times) >= 2:
        diffs = np.diff(np.asarray(times, dtype=np.float64))
        diffs = diffs[diffs > 0]
        if diffs.size:
            return float(np.median(diffs))
    return 1.0 / nominal_fps


def _runs(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    """Inclusive (first, last) position pairs of each run of True."""
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


@dataclass
class _Interval:
    first: int      # positions in the records list, inclusive
    last: int
    reasons: set


def _interval_bounds(records: Sequence[FrameRecord], iv: _Interval, dt: float) -> Tuple[float, float]:
    return records[iv.first].time_s, records[iv.last].time_s + dt


def _gaze_intervals(records: Sequence[FrameRecord]) -> List[_Interval]:
    out = []
    for first, last in _runs([r.state in _AWAY_LIKE for r in records]):
        confirmed = [k for k in range(first, last + 1) if records[k].state == ProctorState.LOOKING_AWAY]
        if confirmed:
            out.append(_Interval(first, confirmed[-1], {REASON_GAZE}))
    return out


def _persistent_intervals(records, flags, min_duration_s, reason, dt) -> List[_Interval]:
    out = []
    for first, last in _runs(flags):
        if records[last].time_s + dt - records[first].time_s >= min_duration_s:
            out.append(_Interval(first, last, {reason}))
    return out


def _glance_tolerant_intervals(records, flags, merge_gap_s, min_duration_s, reason, dt) -> List[_Interval]:
    """Runs of True; runs separated by gaps <= merge_gap_s are joined
    BEFORE the persistence check, so short looks back (as when reading
    notes) do not restart the count."""
    joined: List[List[int]] = []
    for first, last in _runs(flags):
        if joined and records[first].time_s - (records[joined[-1][1]].time_s + dt) <= merge_gap_s:
            joined[-1][1] = last
        else:
            joined.append([first, last])
    return [_Interval(a, b, {reason}) for a, b in joined
            if records[b].time_s + dt - records[a].time_s >= min_duration_s]


def _summarize_directions(span: Sequence[FrameRecord], labels: List[Optional[str]], min_run_s: float = 0.8
                          ) -> Optional[str]:
    """Directions in the order they happened, e.g. "LEFT > UP > DOWN" for
    an incident where the candidate looked left, then above the screen,
    then down. Only stretches of one direction lasting >= min_run_s are
    listed (at most four); if none is that long, the most frequent label."""
    present = [l for l in labels if l]
    if not present:
        return None
    runs: List[Tuple[str, float, float]] = []      # (label, first_t, last_t)
    for r, lab in zip(span, labels):
        if not lab:
            continue
        if runs and runs[-1][0] == lab:
            runs[-1] = (lab, runs[-1][1], r.time_s)
        else:
            runs.append((lab, r.time_s, r.time_s))
    sequence: List[str] = []
    for lab, t0, t1 in runs:
        if t1 - t0 >= min_run_s and (not sequence or sequence[-1] != lab):
            sequence.append(lab)
    if not sequence:
        return Counter(present).most_common(1)[0][0]
    return " > ".join(sequence[:4]) + (" > ..." if len(sequence) > 4 else "")


def _direction(records: Sequence[FrameRecord], iv: _Interval, cfg: AppConfig) -> Optional[str]:
    """Dominant direction as seen in the video (camera view). The eye
    direction is preferred over the 3D gaze ray: it is measured relative
    to the candidate's own screen position and is the more reliable signal
    vertically."""
    span = records[iv.first:iv.last + 1]
    if any(r.eyes_away for r in span):
        return _summarize_directions(span, [r.eyes_away for r in span])
    if any(r.away_direction for r in span):
        return _summarize_directions(span, [r.away_direction for r in span])
    ic = cfg.video.incidents
    turned = [r for r in span if r.head_turned and r.head_yaw_offset is not None]
    if turned:
        yaw = float(np.median([r.head_yaw_offset for r in turned]))
        pitch = float(np.median([r.head_pitch_offset for r in turned]))
        # yaw > 0 appears as image right; pitch > 0 is a nod down.
        return direction_label(yaw / ic.head_turn_yaw_deg, -pitch / ic.head_turn_pitch_deg,
                               ic.eye_diagonal_min_ratio)
    return None


def extract_incidents(
    records: Sequence[FrameRecord], cfg: AppConfig, nominal_fps: float, gaze_available: bool = True,
) -> List[Incident]:
    if not records:
        return []
    temporal, ic = cfg.temporal, cfg.video.incidents
    dt = frame_period([r.time_s for r in records], nominal_fps)

    intervals: List[_Interval] = []
    if gaze_available:
        intervals += _gaze_intervals(records)
    if ic.count_face_lost_as_not_looking:
        intervals += _persistent_intervals(
            records, [r.num_faces == 0 for r in records],
            temporal.seconds_to_confirm_face_lost, REASON_FACE, dt)
    if ic.enable_head_turn_rule:
        intervals += _persistent_intervals(
            records, [r.num_faces > 0 and r.head_turned for r in records],
            temporal.seconds_to_confirm_away, REASON_HEAD, dt)
    if ic.enable_eye_direction_rule:
        intervals += _glance_tolerant_intervals(
            records, [r.num_faces > 0 and r.eyes_away is not None for r in records],
            ic.merge_gap_s, temporal.seconds_to_confirm_away, REASON_EYES, dt)

    intervals.sort(key=lambda iv: iv.first)
    merged: List[_Interval] = []
    for iv in intervals:
        if merged:
            cur = merged[-1]
            _, cur_end = _interval_bounds(records, cur, dt)
            if records[iv.first].time_s - cur_end <= ic.merge_gap_s:
                cur.last = max(cur.last, iv.last)
                cur.reasons |= iv.reasons
                continue
        merged.append(_Interval(iv.first, iv.last, set(iv.reasons)))

    incidents: List[Incident] = []
    reason_order = [REASON_GAZE, REASON_EYES, REASON_HEAD, REASON_FACE]
    for iv in merged:
        start_s, end_s = _interval_bounds(records, iv, dt)
        if end_s - start_s < ic.min_incident_duration_s:
            continue
        incidents.append(Incident(
            number=len(incidents) + 1,
            start_s=start_s,
            end_s=end_s,
            start_frame=records[iv.first].index,
            end_frame=records[iv.last].index,
            reasons=[r for r in reason_order if r in iv.reasons],
            direction=_direction(records, iv, cfg) if REASON_FACE not in iv.reasons or len(iv.reasons) > 1 else None,
        ))

    # Tag frames so the frame log shows which incident each belongs to.
    times = [r.time_s for r in records]
    for inc in incidents:
        lo = bisect.bisect_left(times, inc.start_s)
        hi = bisect.bisect_left(times, inc.end_s)
        for k in range(lo, hi):
            records[k].incident_number = inc.number
    return incidents


def compute_stats(records: Sequence[FrameRecord], incidents: Sequence[Incident], nominal_fps: float) -> AnalysisStats:
    dt = frame_period([r.time_s for r in records], nominal_fps)
    n = len(records)
    duration = (records[-1].time_s + dt - records[0].time_s) if n else 0.0
    with_face = sum(1 for r in records if r.num_faces > 0)
    # A frame counts as measured when either the 3D gaze or the eye
    # direction could be determined for it.
    measured = sum(1 for r in records if r.num_faces > 0
                   and (r.state != ProctorState.UNCERTAIN or r.eye_x is not None))
    multi_s = sum(1 for r in records if r.num_faces > 1) * dt
    total = sum(i.duration_s for i in incidents)
    return AnalysisStats(
        frames_analyzed=n,
        analyzed_duration_s=duration,
        frames_with_face=with_face,
        frames_gaze_measured=measured,
        gaze_coverage=measured / with_face if with_face else 0.0,
        multiple_faces_s=multi_s,
        incident_count=len(incidents),
        total_not_looking_s=total,
        not_looking_fraction=total / duration if duration > 0 else 0.0,
        longest_incident_s=max((i.duration_s for i in incidents), default=0.0),
    )


def compute_verdict(stats: AnalysisStats, cfg: AppConfig, gaze_available: bool) -> Verdict:
    vc = cfg.video.verdict
    reasons = []
    if stats.incident_count >= vc.min_incidents:
        reasons.append(f"{stats.incident_count} separate periods of not looking at the screen "
                       f"(limit: fewer than {vc.min_incidents}).")
    if stats.longest_incident_s >= vc.min_single_incident_s:
        reasons.append(f"Longest period of not looking at the screen lasted {stats.longest_incident_s:.1f} s "
                       f"(limit: under {vc.min_single_incident_s:.0f} s).")
    if stats.not_looking_fraction >= vc.min_not_looking_fraction:
        reasons.append(f"Not looking at the screen for {stats.not_looking_fraction:.0%} of the video "
                       f"(limit: under {vc.min_not_looking_fraction:.0%}).")
    if reasons:
        return Verdict(VERDICT_CHEATING, reasons)

    if not gaze_available:
        return Verdict(VERDICT_INCONCLUSIVE, [
            "The candidate's gaze could not be measured reliably enough to locate the screen, "
            "so only face-visibility and head-turn checks were possible. No rule was triggered "
            "by those checks, but that is not enough to clear the candidate."])
    if stats.gaze_coverage < vc.min_gaze_coverage:
        return Verdict(VERDICT_INCONCLUSIVE, [
            f"Gaze could be measured on only {stats.gaze_coverage:.0%} of the frames where the face "
            f"was visible (at least {vc.min_gaze_coverage:.0%} needed). Poor lighting, low resolution, "
            f"glare, or an off-angle camera are the usual causes. No rule was triggered, but the "
            f"video does not support a reliable NO CHEATING result."])

    detail = (f"{stats.incident_count} short period(s) of not looking at the screen, "
              f"{stats.total_not_looking_s:.1f} s in total, all below the limits."
              if stats.incident_count else "The candidate kept looking at the screen throughout.")
    return Verdict(VERDICT_NO_CHEATING, [detail])
