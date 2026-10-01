"""
main.py

ExamVision entry point: analyze an UPLOADED exam video (no live webcam).

    1. Enter the candidate's name.
    2. Upload (choose) the recorded exam video.
    3. The video is analyzed frame by frame with the existing gaze
       pipeline and a CHEATING / NO CHEATING verdict is produced from
       whether the candidate was looking at the screen.
    4. Everything is saved to results/<Candidate_Name>/:
         recordings/  a clip of every period of not looking at the screen
         snapshots/   a still image of each period
         report/      report.html / .txt / .json and incidents.csv, with timestamps
         data/        frame-by-frame log, events, calibration used

Usage:
    python main.py                                   desktop app (default)
    python main.py --name "Ali Khan" --video exam.mp4
    python main.py --console                         text prompts, no window

The original real-time webcam mode now lives in live_webcam.py and is
only needed to create a saved calibration for "--calibration file".
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from config import AppConfig
from settings_store import load_config
from video_analysis.analyzer import (
    CALIBRATION_AUTO, CALIBRATION_FILE, AnalysisCancelled, AnalysisError, AnalysisResult, VideoAnalyzer,
)
from video_analysis.incidents import timecode
from video_analysis.output_layout import validate_candidate_name
from video_analysis.video_source import VideoError, probe_video

PROJECT_DIR = Path(__file__).resolve().parent


def _console_progress():
    last = {"t": 0.0, "len": 0}

    def report(fraction: float, message: str) -> None:
        now = time.time()
        if now - last["t"] < 0.25 and fraction < 1.0:
            return
        last["t"] = now
        filled = int(fraction * 30)
        line = f"\r[{'#' * filled}{'.' * (30 - filled)}] {fraction:6.1%}  {message}"
        sys.stdout.write(line.ljust(last["len"]))
        sys.stdout.flush()
        last["len"] = len(line)
    return report


def _print_result(result: AnalysisResult) -> None:
    v, s = result.verdict, result.stats
    print("\n")
    print("=" * 64)
    print(f"  VERDICT for {result.candidate_name}: {v.label}")
    print("=" * 64)
    for reason in v.reasons:
        print(f"  - {reason}")
    print(f"\n  Not looking at the screen: {s.incident_count} period(s), {s.total_not_looking_s:.1f} s "
          f"({s.not_looking_fraction:.1%} of {s.analyzed_duration_s:.0f} s)")
    for inc in result.incidents:
        print(f"    #{inc.number:<3} {timecode(inc.start_s)} -> {timecode(inc.end_s)}  "
              f"({inc.duration_s:.1f} s)  {inc.reason_text()}" + (f" [{inc.direction}]" if inc.direction else ""))
    for w in result.warnings:
        print(f"  WARNING: {w}")
    print(f"\n  Results folder: {result.paths.root}")
    print(f"  Report        : {result.report_html}")


def run_analysis_cli(cfg: AppConfig, name: str, video: str, calibration: str) -> int:
    analyzer = VideoAnalyzer(cfg, PROJECT_DIR)
    print(f"Analyzing '{video}' for candidate '{name}' ...")
    try:
        result = analyzer.run(name, video, calibration, progress=_console_progress())
    except AnalysisError as e:
        print(f"\nERROR: {e}")
        return 2
    except (KeyboardInterrupt, AnalysisCancelled):
        print("\nCancelled.")
        return 130
    _print_result(result)
    return 0


def _prompt_console(cfg: AppConfig) -> tuple[str, str]:
    while True:
        name = input("Candidate's full name: ").strip()
        error = validate_candidate_name(name)
        if not error:
            break
        print(f"  {error}")
    while True:
        # "Copy as path" in Windows Explorer wraps the path in quotes.
        video = input("Path of the exam video to upload: ").strip().strip('"').strip("'")
        try:
            print(f"  {probe_video(video).describe()}")
            return name, video
        except VideoError as e:
            print(f"  {e}")


def main(argv=None) -> int:
    from serve import check_environment
    check_environment()
    parser = argparse.ArgumentParser(description="ExamVision: detect looking away from the screen in an exam video.")
    parser.add_argument("--name", help="candidate's name (used for the results folder)")
    parser.add_argument("--video", help="path of the exam video to analyze")
    parser.add_argument("--calibration", choices=(CALIBRATION_AUTO, CALIBRATION_FILE), default=CALIBRATION_AUTO,
                        help="auto: locate the screen from the video (default); "
                             "file: use calibration_data.json from live_webcam.py")
    parser.add_argument("--annotated-video", action="store_true",
                        help="also export the full video with the verdict banner burned in")
    parser.add_argument("--copy-video", action="store_true", help="copy the original video into the results folder")
    parser.add_argument("--console", action="store_true", help="use text prompts instead of the desktop window")
    args = parser.parse_args(argv)

    cfg = load_config()   # config.py defaults + settings.json (web Settings page)
    cfg.video.export_annotated_video = cfg.video.export_annotated_video or args.annotated_video
    cfg.video.copy_source_video = cfg.video.copy_source_video or args.copy_video

    if args.name or args.video:
        if not (args.name and args.video):
            parser.error("--name and --video must be given together")
        return run_analysis_cli(cfg, args.name, args.video, args.calibration)

    if not args.console:
        try:
            from gui import run_gui
        except ImportError as e:  # tkinter missing from this Python build
            print(f"Desktop window unavailable ({e}); using text prompts instead.")
        else:
            run_gui(cfg, PROJECT_DIR)
            return 0

    name, video = _prompt_console(cfg)
    return run_analysis_cli(cfg, name, video, args.calibration)


if __name__ == "__main__":
    sys.exit(main())
