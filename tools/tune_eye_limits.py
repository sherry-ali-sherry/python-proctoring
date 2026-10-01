"""
tools/tune_eye_limits.py

Tune the eye-direction limits (sideways / up / down) on videos you have
labelled yourself. It replays the eye rule on each analysis's saved
data/frame_log.csv, so trying hundreds of settings takes seconds and the
videos are not re-analyzed.

Workflow
--------
1. Record short test videos where you know what happens (look at the
   screen, then at each corner, above the screen, at notes, at a phone...).
   Analyze each one normally (web app or main.py).

2. Label each one. Create a template and fill it in while watching the
   video (or the review player):

       python tools/tune_eye_limits.py template results/Ali_Khan

   This writes results/Ali_Khan/labels.csv:

       start_s,end_s,label
       0.0,4.5,SCREEN
       4.5,9.6,AWAY
       ...

   label is SCREEN or AWAY. Leave out moments you are unsure about; only
   labelled time is scored. tools/examples/sir_labels.csv is a finished
   example.

3. See how the current settings do, second by second:

       python tools/tune_eye_limits.py inspect results/Ali_Khan

4. Search for better limits over every labelled folder:

       python tools/tune_eye_limits.py tune results/*

   It prints the current score, the best settings, and a leave-one-video-
   out check (settings tuned without a video, scored on it). If that check
   is much worse than the tuned score, you need more labelled videos
   before trusting the new limits.

5. Save the best settings (writes settings.json, which the web app, CLI
   and desktop app all read):

       python tools/tune_eye_limits.py tune results/* --apply

Scores
------
Every labelled frame is compared with the analysis:
  frame  - is the eye rule "away" on this frame? (tests the limits directly)
  period - is this frame inside a reported period of not looking?
           (limits + the 3.5 s persistence and glance merging; this is
           what the verdict uses, and what the search maximizes)
precision = of the frames flagged away, how many really were away
recall    = of the frames really away, how many were flagged
F1        = balance of the two.
Only the eye rule is replayed; the 3D-gaze, head-turn and face-missing
rules can add more periods in a full analysis.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402

import settings_store  # noqa: E402
from config import AppConfig  # noqa: E402
from gaze.eye_direction import eye_deviation, fit_eye_baseline, label_eye_frames  # noqa: E402
from proctoring.state_manager import ProctorState  # noqa: E402
from video_analysis.eye_replay import EyeLog, NoEyeData, load_eye_log  # noqa: E402
from video_analysis.incidents import FrameRecord, extract_incidents  # noqa: E402

LIMIT_KEYS = ("eye_limit_side", "eye_limit_up", "eye_limit_down")
GRID = {
    "eye_limit_side": np.round(np.arange(0.15, 0.501, 0.05), 3),
    "eye_limit_up": np.round(np.arange(0.10, 0.401, 0.03), 3),
    "eye_limit_down": np.round(np.arange(0.20, 0.601, 0.05), 3),
}


# --------------------------------------------------------------------------- #
@dataclass
class Session:
    name: str
    folder: Path
    log: EyeLog
    truth: np.ndarray        # per frame: 1 away, 0 screen, -1 unlabelled
    fps: float


def read_labels(path: Path, times: Sequence[float]) -> np.ndarray:
    t = np.asarray(times)
    truth = np.full(len(t), -1, dtype=int)
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]
    if not rows or [c.strip().lower() for c in rows[0][:3]] != ["start_s", "end_s", "label"]:
        raise ValueError(f"{path}: first line must be 'start_s,end_s,label'")
    for n, row in enumerate(rows[1:], start=2):
        try:
            start, end, label = float(row[0]), float(row[1]), row[2].strip().upper()
        except (ValueError, IndexError):
            raise ValueError(f"{path} line {n}: expected start_s,end_s,label, got {row}")
        if label not in ("SCREEN", "AWAY") or end <= start:
            raise ValueError(f"{path} line {n}: label must be SCREEN or AWAY and end_s > start_s")
        truth[(t >= start) & (t < end)] = 1 if label == "AWAY" else 0
    return truth


def load_session(folder: Path) -> Optional[Session]:
    log_path, labels_path = folder / "data" / "frame_log.csv", folder / "labels.csv"
    if not log_path.is_file():
        print(f"  skip {folder.name}: no data/frame_log.csv")
        return None
    if not labels_path.is_file():
        print(f"  skip {folder.name}: no labels.csv (create one with the 'template' command)")
        return None
    try:
        log = load_eye_log(log_path)
    except NoEyeData as e:
        print(f"  skip {folder.name}: {e}")
        return None
    truth = read_labels(labels_path, log.times)
    if (truth >= 0).sum() == 0:
        print(f"  skip {folder.name}: labels.csv covers no frames")
        return None
    if not log.has_blink:
        print(f"  note {folder.name}: analyzed before v1.2 (no eye_blink column); re-analyze for exact results")
    dts = np.diff(log.times)
    fps = 1.0 / float(np.median(dts[dts > 0])) if len(dts) else 30.0
    return Session(folder.name, folder, log, truth, fps)


# --------------------------------------------------------------------------- #
def _with_limits(cfg: AppConfig, limits: Dict[str, float]) -> AppConfig:
    ic = dataclasses.replace(cfg.video.incidents, **limits,
                             enable_head_turn_rule=False, count_face_lost_as_not_looking=False,
                             enable_eye_direction_rule=True)
    return dataclasses.replace(cfg, video=dataclasses.replace(cfg.video, incidents=ic))


def predict(session: Session, cfg: AppConfig, baseline) -> Tuple[np.ndarray, np.ndarray]:
    """(frame_away, in_period) boolean arrays for this session."""
    ic = cfg.video.incidents
    log = session.log
    labels = label_eye_frames(log.times, log.eyes, baseline, ic)
    records = [FrameRecord(index=i, time_s=t, num_faces=n, state=ProctorState.LOOKING_AT_SCREEN,
                           eye_x=e.x if e else None, eyes_away=lab)
               for i, (t, n, e, lab) in enumerate(zip(log.times, log.num_faces, log.eyes, labels))]
    incidents = extract_incidents(records, cfg, session.fps, gaze_available=False)
    t = np.asarray(log.times)
    in_period = np.zeros(len(t), dtype=bool)
    for inc in incidents:
        in_period |= (t >= inc.start_s) & (t < inc.end_s)
    return np.array([lab is not None for lab in labels]), in_period


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, pred: np.ndarray, truth: np.ndarray) -> None:
        m = truth >= 0
        p, y = pred[m], truth[m] == 1
        self.tp += int((p & y).sum()); self.fp += int((p & ~y).sum())
        self.fn += int((~p & y).sum()); self.tn += int((~p & ~y).sum())

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def text(self) -> str:
        return f"F1 {self.f1:.3f}  (precision {self.precision:.2f}, recall {self.recall:.2f})"


def evaluate(sessions: Sequence[Session], cfg: AppConfig, baselines) -> Tuple[Score, Score]:
    frame, period = Score(), Score()
    for s in sessions:
        fa, ip = predict(s, cfg, baselines[s.name])
        frame.add(fa, s.truth)
        period.add(ip, s.truth)
    return frame, period


def grid_search(sessions: Sequence[Session], base: AppConfig, baselines):
    best = None
    for combo in itertools.product(*(GRID[k] for k in LIMIT_KEYS)):
        limits = dict(zip(LIMIT_KEYS, (float(v) for v in combo)))
        frame, period = evaluate(sessions, _with_limits(base, limits), baselines)
        key = (round(period.f1, 4), round(frame.f1, 4))
        if best is None or key > best[0]:
            best = (key, limits, frame, period)
    return best


def _current_limits(cfg: AppConfig) -> Dict[str, float]:
    return {k: getattr(cfg.video.incidents, k) for k in LIMIT_KEYS}


def _fmt(limits: Dict[str, float]) -> str:
    return "  ".join(f"{k.replace('eye_limit_', '')}={v:.2f}" for k, v in limits.items())


# --------------------------------------------------------------------------- #
def cmd_template(args) -> int:
    for folder in args.folders:
        folder = Path(folder)
        target = folder / "labels.csv"
        if target.exists() and not args.force:
            print(f"{target} already exists (use --force to overwrite)")
            continue
        duration = ""
        log_path = folder / "data" / "frame_log.csv"
        if log_path.is_file():
            with open(log_path, newline="", encoding="utf-8") as f:
                times = [float(r["time_s"]) for r in csv.DictReader(f)]
            duration = f"{times[-1]:.1f}" if times else ""
        target.write_text(
            "# Watch the video and list what the candidate was doing.\n"
            "# label = SCREEN (looking at the screen) or AWAY (anywhere else). Times in seconds.\n"
            "# Leave unclear moments out; only labelled time is scored. Delete the example rows.\n"
            "start_s,end_s,label\n"
            f"0.0,5.0,SCREEN\n5.0,{duration or '10.0'},AWAY\n", encoding="utf-8")
        print(f"wrote {target}" + (f" (video is {duration} s long)" if duration else ""))
    return 0


def cmd_inspect(args) -> int:
    cfg = settings_store.load_config()
    ic = cfg.video.incidents
    for folder in args.folders:
        folder = Path(folder)
        try:
            log = load_eye_log(folder / "data" / "frame_log.csv")
        except (NoEyeData, FileNotFoundError) as e:
            print(f"{folder.name}: {e}")
            continue
        baseline = fit_eye_baseline([e for e, n in zip(log.eyes, log.num_faces) if n > 0],
                                    cfg.video.auto_calibration, ic)
        if baseline is None:
            print(f"{folder.name}: no eye data")
            continue
        labels = label_eye_frames(log.times, log.eyes, baseline, ic)
        truth = read_labels(folder / "labels.csv", log.times) if (folder / "labels.csv").is_file() else None
        print(f"\n== {folder.name}   limits: {_fmt(_current_limits(cfg))}")
        print(f"   screen eye position: x={baseline.x:+.2f} v={baseline.v:+.2f}. {baseline.message}")
        print("   sec | side dev | vert dev | ellipse r | eyes away     | your label")
        t = np.asarray(log.times)
        for sec in range(int(t[-1]) + 1):
            idx = [i for i in np.where((t >= sec) & (t < sec + 1))[0] if log.eyes[i] is not None]
            if not idx:
                print(f"   {sec:3d} | (no face)")
                continue
            devs = np.array([eye_deviation(log.eyes[i], baseline, ic) for i in idx])
            rx, ry = np.median(devs, axis=0)
            dx, dv = np.median([[log.eyes[i].x - baseline.x, log.eyes[i].v - baseline.v] for i in idx], axis=0)
            labs = [labels[i] or "-" for i in idx]
            top = max(set(labs), key=labs.count)
            lab = ""
            if truth is not None:
                tv = truth[idx]
                lab = "AWAY" if (tv == 1).mean() > 0.5 else "SCREEN" if (tv == 0).mean() > 0.5 else ""
            print(f"   {sec:3d} | {dx:+8.2f} | {dv:+8.2f} | {np.hypot(rx, ry):9.2f} | "
                  f"{top:<10} {labs.count(top) / len(labs):3.0%} | {lab}")
        print("   side/vert dev: eye position minus the screen position (vert > 0 = up).")
        print("   ellipse r > 1 means beyond the limits (counted as away).")
    return 0


def cmd_tune(args) -> int:
    folders = [Path(f) for f in args.folders if Path(f).is_dir()]
    print(f"Loading {len(folders)} folder(s)")
    sessions = [s for s in (load_session(f) for f in folders) if s]
    if not sessions:
        print("No labelled analyses found. Create labels.csv files first (see the 'template' command).")
        return 1
    labelled = sum(int((s.truth >= 0).sum()) for s in sessions)
    away = sum(int((s.truth == 1).sum()) for s in sessions)
    print(f"Using {len(sessions)} labelled video(s), {labelled} labelled frames ({away} away).")
    if len(sessions) < 5:
        print("  Warning: with fewer than ~5 labelled videos the tuned limits will mostly fit these clips.")

    cfg = settings_store.load_config()
    baselines = {s.name: fit_eye_baseline([e for e, n in zip(s.log.eyes, s.log.num_faces) if n > 0],
                                          cfg.video.auto_calibration, cfg.video.incidents) for s in sessions}
    current = _current_limits(cfg)
    frame, period = evaluate(sessions, _with_limits(cfg, current), baselines)
    print(f"\nCurrent  {_fmt(current)}\n  frame : {frame.text()}\n  period: {period.text()}")
    for s in sessions:
        f1, p1 = evaluate([s], _with_limits(cfg, current), baselines)
        print(f"    {s.name:<28} period {p1.text()}")

    n = int(np.prod([len(v) for v in GRID.values()]))
    print(f"\nSearching {n} combinations...")
    _, best, bframe, bperiod = grid_search(sessions, cfg, baselines)
    print(f"Best     {_fmt(best)}\n  frame : {bframe.text()}\n  period: {bperiod.text()}")
    for s in sessions:
        _, p1 = evaluate([s], _with_limits(cfg, best), baselines)
        print(f"    {s.name:<28} period {p1.text()}")

    if len(sessions) >= 2:
        print("\nLeave-one-video-out check (tuned without the video, scored on it):")
        held = Score()
        for s in sessions:
            others = [o for o in sessions if o is not s]
            _, lim, _, _ = grid_search(others, cfg, baselines)
            _, p1 = evaluate([s], _with_limits(cfg, lim), baselines)
            held.tp += p1.tp; held.fp += p1.fp; held.fn += p1.fn; held.tn += p1.tn
            print(f"    {s.name:<28} {_fmt(lim)}  ->  period {p1.text()}")
        print(f"  overall held-out period {held.text()}")
        if held.f1 < bperiod.f1 - 0.05:
            print("  The held-out score is clearly lower: the best settings are fitted to these clips. "
                  "Label more videos before applying them.")

    if args.apply:
        overrides = settings_store.load_overrides()
        overrides.update({f"video.incidents.{k}": v for k, v in best.items()})
        clean, errors = settings_store.validate(overrides)
        if errors:
            print("\nNot saved: " + "; ".join(errors))
            return 1
        settings_store.save_overrides(clean)
        print(f"\nSaved to {settings_store.SETTINGS_FILE}. New analyses use these limits.")
    else:
        print("\nNothing saved. Re-run with --apply to save the best settings, or set them on the web Settings page.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tune ExamVision's eye-direction limits on labelled videos.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("template", help="create labels.csv in results folders")
    p.add_argument("folders", nargs="+")
    p.add_argument("--force", action="store_true", help="overwrite an existing labels.csv")
    p.set_defaults(func=cmd_template)
    p = sub.add_parser("inspect", help="second-by-second eye deviation and decision")
    p.add_argument("folders", nargs="+")
    p.set_defaults(func=cmd_inspect)
    p = sub.add_parser("tune", help="search the best limits over labelled folders")
    p.add_argument("folders", nargs="+")
    p.add_argument("--apply", action="store_true", help="save the best limits to settings.json")
    p.set_defaults(func=cmd_tune)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
