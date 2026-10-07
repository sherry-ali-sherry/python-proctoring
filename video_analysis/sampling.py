"""
video_analysis/sampling.py

Faster analysis by measuring only some frames, without changing results
(v1.4).

Face landmarks are the most expensive per-frame step. Most of a proctoring
video is steady (the candidate reads the screen, or holds a look away), and
there a few measurements per second carry all the information. Only around
CHANGES (a look starting or ending, a blink, the face leaving) and around
values close to a limit does the exact frame matter.

So the analyzer:
  1. measures every Nth frame (VideoConfig.analysis_fps per second),
  2. fills the frames in between with the NEAREST measured frame
     (hold_measurements), and makes the normal per-frame decisions on that
     full-rate timeline,
  3. finds the frames whose decision could differ if they were measured
     (refine_frames: next to a change, or near a limit), measures exactly
     those, and decides again.

Every later step (state machine, incidents, clips, frame log) still sees
one record per video frame at the video's own frame rate, so all the
time-based rules behave as before. Records filled in from a neighbour are
marked measured=False in the frame log.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from gaze.directions import outside_ellipse  # noqa: F401  (kept for callers' convenience)
from gaze.eye_direction import EyeBaseline, eye_deviation


def choose_sample_indices(times: Sequence[float], analysis_fps: float) -> List[int]:
    """Indices of the frames to measure in the first pass: the first frame,
    then one every 1/analysis_fps seconds of media time (the last frame too,
    so the end of the video is covered). analysis_fps <= 0 means all."""
    n = len(times)
    if n == 0:
        return []
    if analysis_fps <= 0:
        return list(range(n))
    every = 1.0 / analysis_fps
    out, next_t = [], times[0]
    for i, t in enumerate(times):
        if t >= next_t - 1e-6:
            out.append(i)
            next_t = t + every
    if out[-1] != n - 1:
        out.append(n - 1)
    return out


def nearest_measured(n: int, measured: Sequence[int]) -> np.ndarray:
    """For every frame index 0..n-1, the index of the nearest measured frame
    (ties go to the earlier one)."""
    m = np.asarray(sorted(measured), dtype=np.int64)
    idx = np.arange(n)
    pos = np.searchsorted(m, idx)
    left = m[np.clip(pos - 1, 0, len(m) - 1)]
    right = m[np.clip(pos, 0, len(m) - 1)]
    return np.where(np.abs(idx - left) <= np.abs(right - idx), left, right)


def hold_measurements(times: Sequence[float], measured: Dict[int, object]) -> list:
    """Full-rate list of measurements: measured frames as they are, every
    other frame a copy of its nearest measured frame stamped with its own
    time."""
    n = len(times)
    src = nearest_measured(n, list(measured.keys()))
    out = []
    for i in range(n):
        j = int(src[i])
        m = measured[j]
        out.append(m if j == i else dataclasses.replace(m, timestamp=times[i]))
    return out


def _flags(r, baseline: Optional[EyeBaseline], ic) -> tuple:
    """Everything a frame's decision depends on, coarsely: if any of these
    differ between two neighbouring measured frames, the frames between them
    are measured too."""
    return (
        min(r.num_faces, 2),
        r.eyes_away is not None,
        bool(r.head_turned),
        bool(r.facing_away),
        r.eye_blink is not None and r.eye_blink >= ic.eye_blink_ignore,
        getattr(r.state, "value", r.state),
    )


def _near_limit(r, baseline: Optional[EyeBaseline], ic, lo: float, hi: float, blink_band: float = 0.12) -> bool:
    """True when a measured value sits close to one of the limits, where a
    neighbouring frame could fall on the other side."""
    from gaze.eye_direction import EyeDirection
    if baseline is not None and r.eye_x is not None and r.eye_up is not None and r.eye_down is not None:
        e = EyeDirection(r.eye_x, r.eye_up, r.eye_down, r.eye_blink or 0.0)
        rx, ry = eye_deviation(e, baseline, ic)
        if lo <= float(np.hypot(rx, ry)) <= hi:
            return True
    if r.eye_blink is not None and abs(r.eye_blink - ic.eye_blink_ignore) <= blink_band:
        return True
    if r.head_yaw_offset is not None and r.head_pitch_offset is not None:
        rr = float(np.hypot(r.head_yaw_offset / ic.head_turn_yaw_deg, r.head_pitch_offset / ic.head_turn_pitch_deg))
        if lo <= rr <= hi:
            return True
    if r.head_cam_yaw is not None:
        rr = float(np.hypot(r.head_cam_yaw / ic.facing_away_yaw_deg,
                            (r.head_cam_pitch or 0.0) / ic.facing_away_pitch_deg))
        if lo <= rr <= hi:
            return True
    return False


def refine_frames(records: Sequence, measured: Set[int], baseline: Optional[EyeBaseline], ic,
                  near_lo: float = 0.8, near_hi: float = 1.25, blink_band: float = 0.12) -> List[int]:
    """Unmeasured frame indices whose decision could change if measured:
    every frame between two consecutive measured frames whose decisions
    differ, and every frame next to a measured frame that is close to a
    limit. Sorted."""
    ms = sorted(measured)
    want: Set[int] = set()
    for a, b in zip(ms, ms[1:]):
        if b - a <= 1:
            continue
        ra, rb = records[a], records[b]
        if (_flags(ra, baseline, ic) != _flags(rb, baseline, ic)
                or _near_limit(ra, baseline, ic, near_lo, near_hi, blink_band)
                or _near_limit(rb, baseline, ic, near_lo, near_hi, blink_band)):
            want.update(range(a + 1, b))
    return sorted(want - set(measured))
