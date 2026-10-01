"""
video_analysis/output_layout.py

Candidate name validation and the per-candidate results folder:

    results/<Candidate_Name>/
        recordings/   one video clip per incident (not looking at the screen)
        snapshots/    one still image per incident
        report/       report.html, report.txt, report.json, incidents.csv
        data/         frame_log.csv, events.json, calibration.json
        source/       copy of the uploaded video (only if copy_source_video)

An existing, non-empty folder for the same name is never overwritten:
the new analysis goes to results/<Candidate_Name>_<YYYYmmdd_HHMMSS>/.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MAX_NAME_LENGTH = 80
_MAX_FOLDER_LENGTH = 60
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL",
                     *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def validate_candidate_name(name: str) -> Optional[str]:
    """Return a user-facing error message, or None if the name is usable."""
    stripped = (name or "").strip()
    if not stripped:
        return "Please enter the candidate's name."
    if len(stripped) > MAX_NAME_LENGTH:
        return f"The name is too long (maximum {MAX_NAME_LENGTH} characters)."
    if not any(ch.isalnum() for ch in stripped):
        return "The name must contain at least one letter or digit."
    if not sanitize_folder_name(stripped):
        return "The name contains only characters that cannot be used in a folder name."
    return None


def sanitize_folder_name(name: str) -> str:
    """Filesystem-safe folder name derived from the candidate's name.
    Unicode letters are kept; characters Windows forbids are removed and
    whitespace becomes single underscores."""
    cleaned = _INVALID_CHARS.sub("", name or "")
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    cleaned = cleaned[:_MAX_FOLDER_LENGTH].strip(" ._")
    if not cleaned:
        return ""
    if cleaned.split(".")[0].upper() in _WINDOWS_RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned


@dataclass
class ResultPaths:
    root: Path
    recordings: Path
    snapshots: Path
    report: Path
    data: Path
    source: Path

    def relative(self, path: Path) -> str:
        """Path relative to the results folder, with forward slashes
        (valid in both reports and HTML links)."""
        return Path(path).relative_to(self.root).as_posix()


def create_result_folders(results_root: Path, candidate_name: str, now: Optional[float] = None) -> ResultPaths:
    folder = sanitize_folder_name(candidate_name)
    if not folder:
        raise ValueError(f"Cannot build a folder name from candidate name {candidate_name!r}")
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    target = results_root / folder
    if target.exists() and any(target.iterdir()):
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now if now is not None else time.time()))
        target = results_root / f"{folder}_{stamp}"
        suffix = 2
        while target.exists():
            target = results_root / f"{folder}_{stamp}_{suffix}"
            suffix += 1

    paths = ResultPaths(
        root=target,
        recordings=target / "recordings",
        snapshots=target / "snapshots",
        report=target / "report",
        data=target / "data",
        source=target / "source",
    )
    for p in (paths.recordings, paths.snapshots, paths.report, paths.data):
        p.mkdir(parents=True, exist_ok=True)
    return paths
