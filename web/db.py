"""
web/db.py

SQLite store for the web app (standard library only). Holds:
    analyses          one row per uploaded video: queue status, where its
                      results folder is, headline numbers, and the
                      reviewer's final decision
    incident_reviews  the reviewer's decision on each incident

The results folder (report/report.json) stays the source of truth for
the analysis itself; this database indexes it and holds review state,
which is also mirrored into report/review.json so the folder remains
self-contained.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id               TEXT PRIMARY KEY,
    candidate_name   TEXT NOT NULL,
    video_filename   TEXT NOT NULL,
    upload_path      TEXT,
    calibration_mode TEXT NOT NULL DEFAULT 'auto',
    status           TEXT NOT NULL,
    message          TEXT,
    error            TEXT,
    created_at       REAL NOT NULL,
    started_at       REAL,
    finished_at      REAL,
    results_dir      TEXT,
    auto_verdict     TEXT,
    incident_count   INTEGER,
    not_looking_s    REAL,
    duration_s       REAL,
    final_verdict    TEXT,
    verdict_note     TEXT,
    reviewer         TEXT,
    reviewed_at      REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS analyses_results_dir ON analyses(results_dir);
CREATE TABLE IF NOT EXISTS incident_reviews (
    analysis_id  TEXT NOT NULL REFERENCES analyses(id),
    number       INTEGER NOT NULL,
    status       TEXT NOT NULL,
    note         TEXT NOT NULL DEFAULT '',
    reviewer     TEXT NOT NULL,
    updated_at   REAL NOT NULL,
    PRIMARY KEY (analysis_id, number)
);
"""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def _rows(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # -- analyses ----------------------------------------------------------
    def create(self, analysis_id: str, candidate_name: str, video_filename: str, upload_path: str,
               calibration_mode: str) -> None:
        self._exec(
            "INSERT INTO analyses (id, candidate_name, video_filename, upload_path, calibration_mode, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (analysis_id, candidate_name, video_filename, upload_path, calibration_mode, STATUS_QUEUED, time.time()))

    def import_finished(self, analysis_id: str, candidate_name: str, video_filename: str, results_dir: str,
                        created_at: float, auto_verdict: str, incident_count: int, not_looking_s: float,
                        duration_s: float) -> bool:
        """Registers a results folder produced outside the web app (CLI /
        desktop). Returns False if that folder is already registered."""
        cur = self._exec(
            "INSERT OR IGNORE INTO analyses (id, candidate_name, video_filename, status, created_at, started_at, "
            "finished_at, results_dir, auto_verdict, incident_count, not_looking_s, duration_s) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (analysis_id, candidate_name, video_filename, STATUS_DONE, created_at, created_at, created_at,
             results_dir, auto_verdict, incident_count, not_looking_s, duration_s))
        return cur.rowcount == 1

    def get(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        return rows[0] if rows else None

    def list(self) -> List[Dict[str, Any]]:
        return self._rows("SELECT * FROM analyses ORDER BY created_at DESC")

    def next_queued(self) -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT * FROM analyses WHERE status = ? ORDER BY created_at LIMIT 1", (STATUS_QUEUED,))
        return rows[0] if rows else None

    def mark_running(self, analysis_id: str) -> bool:
        cur = self._exec("UPDATE analyses SET status = ?, started_at = ?, message = ? WHERE id = ? AND status = ?",
                         (STATUS_RUNNING, time.time(), "Starting", analysis_id, STATUS_QUEUED))
        return cur.rowcount == 1

    def mark_done(self, analysis_id: str, results_dir: str, auto_verdict: str, incident_count: int,
                  not_looking_s: float, duration_s: float) -> None:
        self._exec(
            "UPDATE analyses SET status = ?, finished_at = ?, results_dir = ?, auto_verdict = ?, incident_count = ?, "
            "not_looking_s = ?, duration_s = ?, upload_path = NULL, message = NULL WHERE id = ?",
            (STATUS_DONE, time.time(), results_dir, auto_verdict, incident_count, not_looking_s, duration_s, analysis_id))

    def mark_ended(self, analysis_id: str, status: str, error: Optional[str] = None) -> None:
        self._exec("UPDATE analyses SET status = ?, finished_at = ?, error = ?, upload_path = NULL, message = NULL "
                   "WHERE id = ?", (status, time.time(), error, analysis_id))

    def cancel_if_queued(self, analysis_id: str) -> bool:
        cur = self._exec("UPDATE analyses SET status = ?, finished_at = ? WHERE id = ? AND status = ?",
                         (STATUS_CANCELLED, time.time(), analysis_id, STATUS_QUEUED))
        return cur.rowcount == 1

    def requeue_interrupted(self) -> int:
        """Analyses that were running when the server stopped go back to
        the queue (their uploaded video is still on disk)."""
        cur = self._exec("UPDATE analyses SET status = ?, started_at = NULL, message = NULL WHERE status = ?",
                         (STATUS_QUEUED, STATUS_RUNNING))
        return cur.rowcount

    def delete(self, analysis_id: str) -> bool:
        """Removes an analysis and its incident reviews. Refuses queued or
        running analyses (they must be cancelled first)."""
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                row = self._conn.execute("SELECT status FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
                deletable = row is not None and row["status"] not in (STATUS_QUEUED, STATUS_RUNNING)
                if deletable:
                    # reviews first: they reference the analysis (foreign key)
                    self._conn.execute("DELETE FROM incident_reviews WHERE analysis_id = ?", (analysis_id,))
                    self._conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return deletable

    def set_final_verdict(self, analysis_id: str, final_verdict: Optional[str], note: str, reviewer: str) -> None:
        self._exec("UPDATE analyses SET final_verdict = ?, verdict_note = ?, reviewer = ?, reviewed_at = ? WHERE id = ?",
                   (final_verdict, note, reviewer, time.time(), analysis_id))

    # -- incident reviews --------------------------------------------------
    def set_incident_review(self, analysis_id: str, number: int, status: str, note: str, reviewer: str) -> None:
        self._exec(
            "INSERT INTO incident_reviews (analysis_id, number, status, note, reviewer, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(analysis_id, number) DO UPDATE SET status = excluded.status, note = excluded.note, "
            "reviewer = excluded.reviewer, updated_at = excluded.updated_at",
            (analysis_id, number, status, note, reviewer, time.time()))

    def incident_reviews(self, analysis_id: str) -> Dict[int, Dict[str, Any]]:
        rows = self._rows("SELECT * FROM incident_reviews WHERE analysis_id = ?", (analysis_id,))
        return {r["number"]: r for r in rows}
