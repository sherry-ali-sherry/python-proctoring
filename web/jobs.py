"""
web/jobs.py

Background analysis queue. One worker thread analyzes one uploaded
video at a time: face-landmark inference is CPU-bound, so running
several at once would only make each one slower.

Lifecycle of an upload:
    uploads/<id>/<video>  --queued-->  running  --done-->  moved to
    results/<Candidate>/source/<video>, report updated to point there.
Failed or cancelled analyses delete their upload. If the server stops
mid-analysis, the job is re-queued on the next start (its upload is
still on disk).
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from settings_store import load_config
from video_analysis.analyzer import AnalysisCancelled, AnalysisError, VideoAnalyzer
from video_analysis.report import load_report, render_report_files
from web.db import STATUS_CANCELLED, STATUS_FAILED, Store

AnalyzerFactory = Callable[..., VideoAnalyzer]


class JobRunner:
    def __init__(self, store: Store, project_dir: Path, uploads_dir: Path,
                 analyzer_factory: Optional[AnalyzerFactory] = None):
        self.store = store
        self.project_dir = Path(project_dir)
        self.uploads_dir = Path(uploads_dir)
        self._factory = analyzer_factory or (lambda cfg, project_dir: VideoAnalyzer(cfg, project_dir))
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._current: Optional[Tuple[str, threading.Event]] = None
        self._progress: Dict[str, Tuple[float, str]] = {}

    # -- control -----------------------------------------------------------
    def start(self) -> None:
        self.store.requeue_interrupted()
        self._thread = threading.Thread(target=self._loop, name="examvision-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stops taking new jobs. A running analysis is left as 'running'
        so the next server start re-queues it rather than losing it."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def wake(self) -> None:
        self._wake.set()

    def cancel(self, analysis_id: str) -> bool:
        if self.store.cancel_if_queued(analysis_id):
            self._delete_upload(analysis_id)
            return True
        with self._lock:
            if self._current and self._current[0] == analysis_id:
                self._current[1].set()
                return True
        return False

    def progress(self, analysis_id: str) -> Optional[Tuple[float, str]]:
        with self._lock:
            return self._progress.get(analysis_id)

    def wait_idle(self, timeout: float) -> bool:
        """Test helper: wait until nothing is queued or running."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                busy = self._current is not None
            if not busy and self.store.next_queued() is None:
                return True
            time.sleep(0.05)
        return False

    # -- worker ------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.store.next_queued()
            if job is None:
                self._wake.wait(timeout=2.0)
                self._wake.clear()
                continue
            if self.store.mark_running(job["id"]):
                self._run(job)

    def _run(self, job: dict) -> None:
        analysis_id = job["id"]
        cancel = threading.Event()
        with self._lock:
            self._current = (analysis_id, cancel)
            self._progress[analysis_id] = (0.0, "Starting")

        def on_progress(fraction: float, message: str) -> None:
            with self._lock:
                self._progress[analysis_id] = (fraction, message)

        try:
            cfg = load_config()
            cfg.video.copy_source_video = False   # the upload is moved in instead (no second copy)
            analyzer = self._factory(cfg, self.project_dir)
            result = analyzer.run(job["candidate_name"], job["upload_path"], job["calibration_mode"],
                                  progress=on_progress, cancel_event=cancel)
            self._adopt_upload(job, result.paths.root, result.paths.source)
            s = result.stats
            self.store.mark_done(analysis_id, str(result.paths.root), result.verdict.label, s.incident_count,
                                 s.total_not_looking_s, s.analyzed_duration_s)
        except AnalysisCancelled:
            self.store.mark_ended(analysis_id, STATUS_CANCELLED)
            self._delete_upload(analysis_id)
        except AnalysisError as e:
            self.store.mark_ended(analysis_id, STATUS_FAILED, str(e))
            self._delete_upload(analysis_id)
        except Exception as e:  # unexpected: record it and keep the worker alive
            traceback.print_exc()
            self.store.mark_ended(analysis_id, STATUS_FAILED, f"Unexpected error: {type(e).__name__}: {e}")
            self._delete_upload(analysis_id)
        finally:
            with self._lock:
                self._current = None
                self._progress.pop(analysis_id, None)

    def _adopt_upload(self, job: dict, results_root: Path, source_dir: Path) -> None:
        """Move the uploaded video into the results folder so the folder is
        self-contained, and point the report at its new location."""
        upload = Path(job["upload_path"])
        if not upload.is_file():
            return
        source_dir.mkdir(parents=True, exist_ok=True)
        target = source_dir / job["video_filename"]
        shutil.move(str(upload), str(target))
        self._delete_upload(job["id"])
        report = load_report(results_root)
        report["video"]["path"] = str(target)
        with open(results_root / "report" / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        render_report_files(results_root / "report", report)

    def _delete_upload(self, analysis_id: str) -> None:
        shutil.rmtree(self.uploads_dir / analysis_id, ignore_errors=True)


def sync_results_folder(store: Store, results_root: Path) -> int:
    """Registers results folders made by the CLI / desktop app, so they
    appear in the web app too. Returns how many were added."""
    added = 0
    results_root = Path(results_root)
    if not results_root.is_dir():
        return 0
    for folder in sorted(p for p in results_root.iterdir() if p.is_dir()):
        report_file = folder / "report" / "report.json"
        if not report_file.is_file():
            continue            # unfinished/partial folder
        try:
            report = load_report(folder)
            s = report["statistics"]
            created = time.mktime(time.strptime(report["analyzed_at"], "%Y-%m-%d %H:%M:%S"))
            added += store.import_finished(
                uuid.uuid4().hex, report["candidate_name"], Path(report["video"]["path"]).name, str(folder),
                created, report["verdict"], s["incident_count"], s["total_not_looking_s"], s["analyzed_duration_s"])
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            continue
    return added
