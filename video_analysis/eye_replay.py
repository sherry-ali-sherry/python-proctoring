"""
video_analysis/eye_replay.py

Re-runs the eye-direction rule on a saved data/frame_log.csv, without
decoding the video or running MediaPipe again. Used by
tools/tune_eye_limits.py (to try many limit settings in seconds) and by the
regression tests.

Only the eye rule is replayed: the 3D-gaze, head-turn and face-missing
rules need the full per-frame measurements, which the log doesn't keep.
Frame logs written before v1.2 have no eye_blink column; blinks are then
treated as open eyes (slightly more "away" frames than the real analysis).
"""
from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config import AppConfig
from gaze.eye_direction import EyeBaseline, EyeDirection, fit_eye_baseline, label_eye_frames
from proctoring.state_manager import ProctorState
from video_analysis.incidents import FrameRecord, Incident, extract_incidents


@dataclass
class EyeLog:
    times: List[float]
    num_faces: List[int]
    eyes: List[Optional[EyeDirection]]
    has_blink: bool


class NoEyeData(ValueError):
    pass


def _num(v: Optional[str]) -> Optional[float]:
    return float(v) if v not in (None, "") else None


def load_eye_log(path: Path) -> EyeLog:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if not {"eye_x", "eye_up", "eye_down"} <= cols:
            raise NoEyeData(f"{path} has no eye-direction columns (analyzed before the eye rule existed); "
                            f"re-analyze the video to use it.")
        has_blink = "eye_blink" in cols
        times, faces, eyes = [], [], []
        for row in reader:
            times.append(float(row["time_s"]))
            n = int(row["num_faces"])
            faces.append(n)
            x, up, down = _num(row["eye_x"]), _num(row["eye_up"]), _num(row["eye_down"])
            blink = _num(row.get("eye_blink")) if has_blink else 0.0
            eyes.append(EyeDirection(x, up, down, blink or 0.0) if n > 0 and x is not None else None)
    return EyeLog(times, faces, eyes, has_blink)


@dataclass
class EyeReplay:
    baseline: Optional[EyeBaseline]
    labels: List[Optional[str]]          # per-frame eyes_away
    incidents: List[Incident]            # eye-rule-only incidents


def replay_eye_rule(log: EyeLog, cfg: AppConfig, nominal_fps: float = 30.0) -> EyeReplay:
    ic = cfg.video.incidents
    baseline = fit_eye_baseline([e for e, n in zip(log.eyes, log.num_faces) if n > 0],
                                cfg.video.auto_calibration, ic)
    labels = label_eye_frames(log.times, log.eyes, baseline, ic)
    records = [FrameRecord(index=i, time_s=t, num_faces=n, state=ProctorState.LOOKING_AT_SCREEN,
                           eye_x=e.x if e else None, eyes_away=lab)
               for i, (t, n, e, lab) in enumerate(zip(log.times, log.num_faces, log.eyes, labels))]
    only_eyes = dataclasses.replace(cfg, video=dataclasses.replace(
        cfg.video, incidents=dataclasses.replace(
            ic, enable_head_turn_rule=False, count_face_lost_as_not_looking=False,
            enable_eye_direction_rule=True)))
    incidents = extract_incidents(records, only_eyes, nominal_fps, gaze_available=False)
    return EyeReplay(baseline, labels, incidents)
