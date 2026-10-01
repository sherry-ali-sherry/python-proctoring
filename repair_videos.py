"""
repair_videos.py

Converts result videos that were saved in a format browsers can't play
(MPEG-4 Part 2 / MJPG, written when the app ran without PyAV) to H.264,
in place. Checks every results/<Name>/recordings/*.mp4|.avi and
results/<Name>/data/review_video.*.

    .\\venv\\Scripts\\python.exe repair_videos.py            convert everything that needs it
    .\\venv\\Scripts\\python.exe repair_videos.py --dry-run  only list what would change

Each file is written to a temporary name first and only replaces the
original once the new file is complete, so an interruption never loses a
video.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av

from serve import check_environment
from settings_store import load_config
from video_analysis.report import load_report, render_report_files
from video_analysis.video_writer import VideoWriter

PROJECT_DIR = Path(__file__).resolve().parent
_BROWSER_CODECS = {"h264"}
_H264_WARNING_PREFIX = "H.264 encoding was unavailable"


def video_codec(path: Path) -> str:
    with av.open(str(path)) as c:
        return c.streams.video[0].codec_context.name


def convert(path: Path) -> Path:
    """Re-encode `path` to H.264 MP4. Returns the new file's path (the
    extension becomes .mp4)."""
    with av.open(str(path)) as src:
        stream = src.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 30)
        tmp_base = path.parent / (path.stem + ".h264tmp")
        writer = None
        try:
            for frame in src.decode(stream):
                img = frame.to_ndarray(format="bgr24")
                if writer is None:
                    writer = VideoWriter(tmp_base, fps, (img.shape[1], img.shape[0]))
                    if not writer.browser_playable:
                        writer.abort()
                        raise RuntimeError("H.264 encoder unavailable in this Python")
                writer.write(img)
        except BaseException:
            if writer is not None:
                writer.abort()
            raise
    tmp = writer.close()
    if tmp is None:
        raise RuntimeError("conversion produced an empty file")
    target = path.with_suffix(".mp4")
    tmp.replace(target)                      # atomic swap on the same drive
    if target != path:
        path.unlink(missing_ok=True)         # e.g. old .avi replaced by .mp4
    return target


def repair_folder(folder: Path, dry_run: bool) -> int:
    candidates = sorted((folder / "recordings").glob("*.mp4")) + sorted((folder / "recordings").glob("*.avi"))
    candidates += sorted((folder / "data").glob("review_video.*"))
    renamed = {}
    changed = 0
    for path in candidates:
        codec = video_codec(path)
        if codec in _BROWSER_CODECS:
            continue
        rel = path.relative_to(folder).as_posix()
        print(f"  {rel}: {codec} -> h264" + ("  (dry run)" if dry_run else ""))
        if dry_run:
            changed += 1
            continue
        new = convert(path)
        changed += 1
        if new != path:
            renamed[rel] = new.relative_to(folder).as_posix()

    if changed and not dry_run and (folder / "report" / "report.json").is_file():
        report = load_report(folder)
        for inc in report["incidents"]:
            if inc.get("clip_file") in renamed:
                inc["clip_file"] = renamed[inc["clip_file"]]
        if report.get("review_video") in renamed:
            report["review_video"] = renamed[report["review_video"]]
        report["warnings"] = [w for w in report["warnings"] if not w.startswith(_H264_WARNING_PREFIX)]
        with open(folder / "report" / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        review_file = folder / "report" / "review.json"
        review = json.loads(review_file.read_text(encoding="utf-8")) if review_file.is_file() else None
        render_report_files(folder / "report", report, review)
    return changed


def main() -> None:
    check_environment()
    parser = argparse.ArgumentParser(description="Convert result videos to browser-playable H.264.")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be converted")
    args = parser.parse_args()
    results_root = PROJECT_DIR / load_config().video.results_dir
    total = 0
    for folder in sorted(p for p in results_root.iterdir() if p.is_dir()):
        print(f"{folder.name}:")
        n = repair_folder(folder, args.dry_run)
        if not n:
            print("  all videos already play in the browser")
        total += n
    print(f"\n{total} video(s) {'need converting' if args.dry_run else 'converted'}.")


if __name__ == "__main__":
    main()
