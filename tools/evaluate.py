"""
tools/evaluate.py

Measure how accurate ExamVision is on videos you have labelled yourself.
Scores the WHOLE analysis (every rule: eyes, head, facing away, face
missing, 3D gaze), not just the eye limits that tools/tune_eye_limits.py
tunes.

Labels
------
Put a labels.csv in each results folder (results/<name>/labels.csv), in the
same format as tune_eye_limits.py:

    start_s,end_s,label
    0.0,7.5,SCREEN
    8.0,12.8,AWAY
    ...

label is SCREEN or AWAY. Leave unclear moments out; only labelled time is
scored. A video where the candidate never looked away is just one SCREEN
row covering the whole video - those are important: they are where false
alarms show up.

Usage
-----
Score the analyses already in the folders (fast, nothing is re-run):

    python tools/evaluate.py results/*

Re-analyze each labelled folder's source video with the CURRENT settings
first (needs results/<name>/source/<video>; nothing in results/ is changed,
the new analyses go to a temporary folder):

    python tools/evaluate.py results/* --rerun

Scores
------
  periods caught   labelled AWAY periods of at least --min-away seconds
                   (default 3, about the 3.5 s persistence) that a reported
                   incident covers by at least half
  false alarms     reported incidents that are mostly (>= 70%) inside time
                   labelled SCREEN
  time precision   of the labelled time reported as not looking, the share
                   that really was AWAY
  time recall      of the labelled AWAY time, the share reported
  start/end error  median distance between a caught period's labelled and
                   reported start (and end), in seconds
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402

GRID_S = 0.05


def read_labels(path: Path) -> List[Tuple[float, float, str]]:
    """Sorted (start, end, label) with touching rows of the same label joined."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]
    if not rows or [c.strip().lower() for c in rows[0][:3]] != ["start_s", "end_s", "label"]:
        raise ValueError(f"{path}: first line must be 'start_s,end_s,label'")
    out: List[Tuple[float, float, str]] = []
    for n, row in enumerate(rows[1:], start=2):
        try:
            a, b, lab = float(row[0]), float(row[1]), row[2].strip().upper()
        except (ValueError, IndexError):
            raise ValueError(f"{path} line {n}: expected start_s,end_s,label, got {row}")
        if lab not in ("SCREEN", "AWAY") or b <= a:
            raise ValueError(f"{path} line {n}: label must be SCREEN or AWAY and end_s > start_s")
        out.append((a, b, lab))
    out.sort()
    joined: List[Tuple[float, float, str]] = []
    for a, b, lab in out:
        if joined and joined[-1][2] == lab and a - joined[-1][1] <= 0.05:
            joined[-1] = (joined[-1][0], max(b, joined[-1][1]), lab)
        else:
            joined.append((a, b, lab))
    return joined


@dataclass
class VideoScore:
    name: str
    periods: int = 0
    caught: int = 0
    false_alarms: int = 0
    tp: float = 0.0          # seconds
    fp: float = 0.0
    fn: float = 0.0
    start_err: List[float] = field(default_factory=list)
    end_err: List[float] = field(default_factory=list)
    missed: List[Tuple[float, float]] = field(default_factory=list)
    alarms: List[Tuple[float, float]] = field(default_factory=list)


def score_video(name: str, labels: Sequence[Tuple[float, float, str]],
                incidents: Sequence[Tuple[float, float]], min_away: float = 3.0) -> VideoScore:
    """incidents: (start_s, end_s) of the reported looking-away incidents."""
    s = VideoScore(name)
    end = max([b for _, b, _ in labels] + [b for _, b in incidents] + [0.0]) + 1.0
    g = np.arange(0.0, end, GRID_S)
    truth = np.full(len(g), -1, dtype=int)
    for a, b, lab in labels:
        truth[(g >= a) & (g < b)] = 1 if lab == "AWAY" else 0
    pred = np.zeros(len(g), dtype=bool)
    for a, b in incidents:
        pred |= (g >= a) & (g < b)
    m = truth >= 0
    y, p = truth[m] == 1, pred[m]
    s.tp = float((p & y).sum() * GRID_S)
    s.fp = float((p & ~y).sum() * GRID_S)
    s.fn = float((~p & y).sum() * GRID_S)

    for a, b, lab in labels:
        if lab != "AWAY" or b - a < min_away:
            continue
        s.periods += 1
        best = None
        for ia, ib in incidents:
            ov = min(b, ib) - max(a, ia)
            if ov >= 0.5 * (b - a) and (best is None or ov > best[0]):
                best = (ov, ia, ib)
        if best:
            s.caught += 1
            s.start_err.append(abs(best[1] - a))
            s.end_err.append(abs(best[2] - b))
        else:
            s.missed.append((a, b))
    for ia, ib in incidents:
        sel = (g >= ia) & (g < ib)
        t = truth[sel]
        t = t[t >= 0]
        if len(t) and (t == 0).mean() >= 0.7:
            s.false_alarms += 1
            s.alarms.append((ia, ib))
    return s


def incidents_from_report(report_json: Path) -> List[Tuple[float, float]]:
    with open(report_json, encoding="utf-8") as f:
        rep = json.load(f)
    return [(float(i["start_s"]), float(i["end_s"])) for i in rep.get("incidents", [])
            if i.get("kind", "gaze") == "gaze"]


def rerun(folder: Path, work: Path) -> Optional[Path]:
    """Re-analyze folder/source/<video> with the current settings into
    `work`; returns the new report.json or None."""
    import settings_store
    from video_analysis.analyzer import VideoAnalyzer
    src = sorted((folder / "source").glob("*.*")) if (folder / "source").is_dir() else []
    if not src:
        print(f"  skip {folder.name}: no source video in {folder / 'source'}")
        return None
    cfg = settings_store.load_config()
    cfg.video.export_review_video = False
    t0 = time.time()
    res = VideoAnalyzer(cfg, work).run(folder.name, str(src[0]))
    print(f"  re-analyzed {folder.name} in {time.time() - t0:.1f} s")
    return res.paths.root / "report" / "report.json"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score ExamVision against your labelled videos.")
    ap.add_argument("folders", nargs="+", help="results folders containing labels.csv")
    ap.add_argument("--rerun", action="store_true", help="re-analyze the source videos with the current settings first")
    ap.add_argument("--min-away", type=float, default=3.0, help="shortest labelled AWAY period that counts as an event (s)")
    args = ap.parse_args(argv)

    folders = [Path(f) for f in args.folders if Path(f).is_dir() and (Path(f) / "labels.csv").is_file()]
    if not folders:
        print("No results folders with a labels.csv found.")
        return 1
    work = Path(tempfile.mkdtemp(prefix="examvision_eval_")) if args.rerun else None
    scores: List[VideoScore] = []
    try:
        for folder in folders:
            report = rerun(folder, work) if args.rerun else folder / "report" / "report.json"
            if report is None or not report.is_file():
                print(f"  skip {folder.name}: no report/report.json")
                continue
            scores.append(score_video(folder.name, read_labels(folder / "labels.csv"),
                                      incidents_from_report(report), args.min_away))
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
    if not scores:
        return 1

    print(f"\n{'video':<22} caught  false   time P  time R   start/end err")
    for s in scores:
        p = s.tp / (s.tp + s.fp) if s.tp + s.fp else 1.0
        r = s.tp / (s.tp + s.fn) if s.tp + s.fn else 1.0
        err = (f"{np.median(s.start_err):.1f} / {np.median(s.end_err):.1f} s" if s.start_err else "-")
        print(f"{s.name:<22} {s.caught:>2}/{s.periods:<3} {s.false_alarms:>4}    {p:6.2f}  {r:6.2f}   {err}")
        for a, b in s.missed:
            print(f"{'':<22}   missed AWAY {a:.1f}-{b:.1f} s")
        for a, b in s.alarms:
            print(f"{'':<22}   false alarm {a:.1f}-{b:.1f} s")
    tp, fp, fn = (sum(getattr(s, k) for s in scores) for k in ("tp", "fp", "fn"))
    P = tp / (tp + fp) if tp + fp else 1.0
    R = tp / (tp + fn) if tp + fn else 1.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    caught, periods = sum(s.caught for s in scores), sum(s.periods for s in scores)
    alarms = sum(s.false_alarms for s in scores)
    print(f"\nOVERALL  periods caught {caught}/{periods}   false alarms {alarms}   "
          f"time precision {P:.3f}  recall {R:.3f}  F1 {F:.3f}   ({len(scores)} videos)")
    if len(scores) < 10:
        print("Note: fewer than ~10 labelled videos; label more (including clean, no-cheating videos) "
              "before trusting small differences.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
