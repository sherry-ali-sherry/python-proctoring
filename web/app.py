"""
web/app.py

FastAPI application: JSON API + the static single-page UI.

    GET    /api/status                               server/queue info
    GET    /api/analyses                             all analyses, newest first
    POST   /api/analyses                             upload a video (multipart: candidate_name, calibration_mode, file)
    GET    /api/analyses/{id}                        one analysis: status, report, reviews
    POST   /api/analyses/{id}/cancel                 cancel a queued/running analysis
    DELETE /api/analyses/{id}                        permanently delete an analysis and its results folder
    GET    /api/analyses/{id}/timeline               per-frame gaze state as time segments
    PUT    /api/analyses/{id}/incidents/{n}/review   reviewer: confirm / false alarm + note
    PUT    /api/analyses/{id}/review                 reviewer: final decision + note
    GET    /api/settings  PUT /api/settings  POST /api/settings/reset
    GET    /files/{id}/{path}                        files in that analysis' results folder
                                                     (supports Range requests for video seeking)

There is no login: the reviewer types their name in the UI and it is
recorded with each decision. Bind to 127.0.0.1 (the default in
serve.py) unless the network it is exposed to is trusted.
"""
from __future__ import annotations

import csv
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import settings_store
from video_analysis.analyzer import CALIBRATION_AUTO, CALIBRATION_FILE, VideoAnalyzer
from video_analysis.incidents import VERDICT_CHEATING, VERDICT_INCONCLUSIVE, VERDICT_NO_CHEATING
from video_analysis.output_layout import sanitize_folder_name, validate_candidate_name
from video_analysis.report import (
    INCIDENT_CONFIRMED, INCIDENT_FALSE_ALARM, INCIDENT_UNREVIEWED, load_report, save_review,
)
from video_analysis.video_source import VideoError, probe_video
from web.db import STATUS_CANCELLED, STATUS_DONE, STATUS_FAILED, STATUS_QUEUED, STATUS_RUNNING, Store
from web.jobs import JobRunner, sync_results_folder

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 20 * 1024 ** 3          # 20 GB
_CHUNK = 4 * 1024 * 1024
_MEDIA_TYPES = {".mp4": "video/mp4", ".avi": "video/x-msvideo", ".html": "text/html; charset=utf-8",
                ".jpg": "image/jpeg", ".csv": "text/csv; charset=utf-8", ".json": "application/json",
                ".txt": "text/plain; charset=utf-8"}
_VERDICTS = (VERDICT_CHEATING, VERDICT_NO_CHEATING, VERDICT_INCONCLUSIVE)
_ACCEPT = "accept"
_MAX_NOTE = 4000
_MAX_REVIEWER = 80


class IncidentReviewIn(BaseModel):
    status: str
    note: str = Field(default="", max_length=_MAX_NOTE)
    reviewer: str = Field(max_length=_MAX_REVIEWER)


class FinalReviewIn(BaseModel):
    decision: str                 # "accept" or one of _VERDICTS
    note: str = Field(default="", max_length=_MAX_NOTE)
    reviewer: str = Field(max_length=_MAX_REVIEWER)


class SettingsIn(BaseModel):
    values: Dict[str, Any]


def _iso(ts: Optional[float]) -> Optional[str]:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else None


def _clean_reviewer(name: str) -> str:
    name = " ".join((name or "").split())
    if not name:
        raise HTTPException(400, "Enter your name as the reviewer first.")
    return name


def create_app(project_dir: Path, db_path: Optional[Path] = None, analyzer_factory=None,
               start_worker: bool = True) -> FastAPI:
    project_dir = Path(project_dir).resolve()
    cfg0 = settings_store.load_config()
    results_root = project_dir / cfg0.video.results_dir
    uploads_dir = project_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    store = Store(db_path or project_dir / "examvision.db")
    runner = JobRunner(store, project_dir, uploads_dir, analyzer_factory)
    timeline_cache: Dict[str, tuple] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        sync_results_folder(store, results_root)
        if start_worker:
            runner.start()
        yield
        runner.stop()
        store.close()

    app = FastAPI(title="ExamVision", lifespan=lifespan)
    app.state.store = store
    app.state.runner = runner

    # -- helpers -----------------------------------------------------------
    def row_or_404(analysis_id: str) -> dict:
        row = store.get(analysis_id)
        if row is None:
            raise HTTPException(404, "Analysis not found.")
        return row

    def done_row(analysis_id: str) -> dict:
        row = row_or_404(analysis_id)
        if row["status"] != STATUS_DONE or not row["results_dir"] or not Path(row["results_dir"]).is_dir():
            raise HTTPException(409, "This analysis has no results yet.")
        return row

    thumbs_cache: Dict[str, tuple] = {}

    def thumbs(row: dict) -> List[str]:
        """URLs of up to 3 incident snapshots (cached per report.json mtime)."""
        if row["status"] != STATUS_DONE or not row["results_dir"]:
            return []
        report_file = Path(row["results_dir"]) / "report" / "report.json"
        try:
            mtime = report_file.stat().st_mtime
        except OSError:
            return []
        cached = thumbs_cache.get(row["id"])
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            incidents = load_report(Path(row["results_dir"]))["incidents"]
        except (OSError, ValueError, KeyError):
            return []
        urls = [f"/files/{row['id']}/{i['snapshot_file']}" for i in incidents if i.get("snapshot_file")][:3]
        thumbs_cache[row["id"]] = (mtime, urls)
        return urls

    def summary(row: dict) -> dict:
        prog = runner.progress(row["id"]) if row["status"] == STATUS_RUNNING else None
        reviewed = row["reviewer"] is not None
        return {
            "id": row["id"], "candidate_name": row["candidate_name"], "video_filename": row["video_filename"],
            "status": row["status"], "calibration_mode": row["calibration_mode"],
            "progress": prog[0] if prog else (1.0 if row["status"] == STATUS_DONE else 0.0),
            "message": prog[1] if prog else row["message"], "error": row["error"],
            "created_at": _iso(row["created_at"]), "finished_at": _iso(row["finished_at"]),
            "auto_verdict": row["auto_verdict"], "final_verdict": row["final_verdict"],
            "effective_verdict": (row["final_verdict"] or row["auto_verdict"]) if reviewed else row["auto_verdict"],
            "reviewed": reviewed, "reviewer": row["reviewer"], "reviewed_at": _iso(row["reviewed_at"]),
            "verdict_note": row["verdict_note"], "incident_count": row["incident_count"],
            "not_looking_s": row["not_looking_s"], "duration_s": row["duration_s"],
            "thumbs": thumbs(row),
        }

    def review_dict(row: dict) -> dict:
        reviews = store.incident_reviews(row["id"])
        latest = max((r["updated_at"] for r in reviews.values()), default=None)
        reviewer = row["reviewer"]
        reviewed_at = row["reviewed_at"]
        if reviewer is None and reviews:
            last = max(reviews.values(), key=lambda r: r["updated_at"])
            reviewer, reviewed_at = last["reviewer"], latest
        elif latest and reviewed_at:
            reviewed_at = max(reviewed_at, latest)
        return {
            "reviewer": reviewer, "reviewed_at": _iso(reviewed_at),
            "final_verdict": row["final_verdict"], "verdict_note": row["verdict_note"] or "",
            "decision_made": row["reviewer"] is not None,
            "incidents": {str(n): {"status": r["status"], "note": r["note"], "reviewer": r["reviewer"],
                                   "updated_at": _iso(r["updated_at"])} for n, r in reviews.items()},
        }

    def persist_review(analysis_id: str) -> None:
        row = store.get(analysis_id)
        save_review(Path(row["results_dir"]), review_dict(row))

    # -- API: analyses ------------------------------------------------------
    @app.get("/api/status")
    def status() -> dict:
        rows = store.list()
        analyzer = VideoAnalyzer(settings_store.load_config(), project_dir)
        return {
            "queued": sum(r["status"] == STATUS_QUEUED for r in rows),
            "running": sum(r["status"] == STATUS_RUNNING for r in rows),
            "saved_calibration_available": analyzer.saved_calibration_available(),
            "video_extensions": list(cfg0.video.video_extensions),
            "max_upload_bytes": MAX_UPLOAD_BYTES,
        }

    @app.get("/api/analyses")
    def list_analyses() -> List[dict]:
        return [summary(r) for r in store.list()]

    @app.post("/api/analyses", status_code=201)
    def create_analysis(candidate_name: str = Form(...), calibration_mode: str = Form(CALIBRATION_AUTO),
                        file: UploadFile = File(...)) -> dict:
        candidate_name = " ".join(candidate_name.split())
        error = validate_candidate_name(candidate_name)
        if error:
            raise HTTPException(400, error)
        if calibration_mode not in (CALIBRATION_AUTO, CALIBRATION_FILE):
            raise HTTPException(400, "Unknown calibration mode.")
        original = Path(file.filename or "").name
        ext = Path(original).suffix.lower()
        if ext not in cfg0.video.video_extensions:
            raise HTTPException(400, f"Unsupported file type '{ext or '(none)'}'. "
                                     f"Accepted: {', '.join(cfg0.video.video_extensions)}")
        stem = sanitize_folder_name(Path(original).stem) or "video"
        safe_name = f"{stem}{ext}"

        analysis_id = uuid.uuid4().hex
        target_dir = uploads_dir / analysis_id
        target_dir.mkdir(parents=True)
        target = target_dir / safe_name
        written = 0
        try:
            with open(target, "wb") as out:
                while chunk := file.file.read(_CHUNK):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"The video is larger than {MAX_UPLOAD_BYTES // 1024 ** 3} GB.")
                    out.write(chunk)
            try:
                probe_video(str(target))
            except VideoError as e:
                raise HTTPException(400, str(e))
        except BaseException:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise

        store.create(analysis_id, candidate_name, safe_name, str(target), calibration_mode)
        runner.wake()
        return summary(store.get(analysis_id))

    @app.get("/api/analyses/{analysis_id}")
    def get_analysis(analysis_id: str) -> dict:
        row = row_or_404(analysis_id)
        out = summary(row)
        if row["status"] == STATUS_DONE and row["results_dir"] and Path(row["results_dir"]).is_dir():
            try:
                out["report"] = load_report(Path(row["results_dir"]))
            except (OSError, ValueError) as e:
                out["error"] = f"The report in {row['results_dir']} is missing or damaged ({type(e).__name__})."
                return out
            out["review"] = review_dict(row)
            out["files_url"] = f"/files/{analysis_id}/"
            out["results_dir"] = row["results_dir"]
        elif row["status"] == STATUS_DONE:
            out["error"] = f"The results folder is missing: {row['results_dir']}"
        return out

    @app.post("/api/analyses/{analysis_id}/cancel")
    def cancel_analysis(analysis_id: str) -> dict:
        row = row_or_404(analysis_id)
        if row["status"] not in (STATUS_QUEUED, STATUS_RUNNING) or not runner.cancel(analysis_id):
            raise HTTPException(409, "Only queued or running analyses can be cancelled.")
        return summary(store.get(analysis_id))

    @app.delete("/api/analyses/{analysis_id}", status_code=204)
    def delete_analysis(analysis_id: str) -> None:
        """Permanently deletes an analysis: its results folder (clips,
        snapshots, report, the uploaded video) and its review records."""
        row = row_or_404(analysis_id)
        if row["status"] in (STATUS_QUEUED, STATUS_RUNNING):
            raise HTTPException(409, "This analysis is still queued or running. Cancel it first.")
        folder = Path(row["results_dir"]).resolve() if row["results_dir"] else None
        if folder is not None and folder.exists():
            root = results_root.resolve()
            # Never delete anything outside results/, or results/ itself.
            if folder == root or not folder.is_relative_to(root):
                raise HTTPException(409, f"Refusing to delete a folder outside {root}: {folder}")
            shutil.rmtree(folder, ignore_errors=True)
            if folder.exists():
                raise HTTPException(409, "Some files could not be deleted because they are open in another "
                                         "program (for example a video player). Close them and try again. "
                                         f"Folder: {folder}")
        store.delete(analysis_id)
        timeline_cache.pop(analysis_id, None)
        thumbs_cache.pop(analysis_id, None)

    @app.get("/api/analyses/{analysis_id}/timeline")
    def timeline(analysis_id: str) -> dict:
        row = done_row(analysis_id)
        log = Path(row["results_dir"]) / "data" / "frame_log.csv"
        if not log.is_file():
            raise HTTPException(404, "Frame log not found.")
        mtime = log.stat().st_mtime
        cached = timeline_cache.get(analysis_id)
        if cached and cached[0] == mtime:
            return cached[1]
        data = _build_timeline(log)
        timeline_cache[analysis_id] = (mtime, data)
        return data

    # -- API: review ----------------------------------------------------------
    @app.put("/api/analyses/{analysis_id}/incidents/{number}/review")
    def review_incident(analysis_id: str, number: int, body: IncidentReviewIn) -> dict:
        row = done_row(analysis_id)
        if body.status not in (INCIDENT_UNREVIEWED, INCIDENT_CONFIRMED, INCIDENT_FALSE_ALARM):
            raise HTTPException(400, "Unknown incident review status.")
        numbers = {i["number"] for i in load_report(Path(row["results_dir"]))["incidents"]}
        if number not in numbers:
            raise HTTPException(404, "Incident not found.")
        store.set_incident_review(analysis_id, number, body.status, body.note.strip(), _clean_reviewer(body.reviewer))
        persist_review(analysis_id)
        return review_dict(store.get(analysis_id))

    @app.put("/api/analyses/{analysis_id}/review")
    def review_final(analysis_id: str, body: FinalReviewIn) -> dict:
        done_row(analysis_id)
        if body.decision != _ACCEPT and body.decision not in _VERDICTS:
            raise HTTPException(400, "Decision must be 'accept' or a verdict.")
        reviewer = _clean_reviewer(body.reviewer)
        final = None if body.decision == _ACCEPT else body.decision
        store.set_final_verdict(analysis_id, final, body.note.strip(), reviewer)
        persist_review(analysis_id)
        row = store.get(analysis_id)
        return {**summary(row), "review": review_dict(row)}

    # -- API: settings --------------------------------------------------------
    @app.get("/api/settings")
    def get_settings() -> List[dict]:
        return settings_store.describe()

    @app.put("/api/settings")
    def put_settings(body: SettingsIn):
        clean, errors = settings_store.validate(body.values)
        unknown = [k for k in body.values if k not in {f.key for f in settings_store.FIELDS}]
        if unknown:
            errors.append(f"Unknown setting(s): {', '.join(unknown)}")
        if errors:
            return JSONResponse({"detail": errors}, status_code=422)
        merged = {**settings_store.load_overrides(), **clean}
        settings_store.save_overrides(merged)
        return settings_store.describe()

    @app.post("/api/settings/reset")
    def reset_settings() -> List[dict]:
        settings_store.save_overrides({})
        return settings_store.describe()

    # -- files ------------------------------------------------------------
    @app.get("/files/{analysis_id}/{rel_path:path}")
    def get_file(analysis_id: str, rel_path: str):
        row = done_row(analysis_id)
        root = Path(row["results_dir"]).resolve()
        target = (root / rel_path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(404, "File not found.")
        media_type = _MEDIA_TYPES.get(target.suffix.lower())
        return FileResponse(target, media_type=media_type)

    # -- UI -----------------------------------------------------------------
    # Browsers may otherwise keep an old styles.css / app.js after an
    # update; "no-cache" makes them re-check (a cheap 304 when unchanged).
    @app.middleware("http")
    async def _revalidate_static(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8",
                            headers={"Cache-Control": "no-cache"})

    return app


_AWAY_STATES = {"POSSIBLE_AWAY", "LOOKING_AWAY", "POSSIBLE_SCREEN"}


def _frame_kind(row: dict) -> str:
    if row["incident_number"]:
        return "incident"
    if row["num_faces"] == "0":
        return "no_face"
    if row["state"] in _AWAY_STATES or row.get("eyes_away"):
        return "glance"
    if row["state"] == "LOOKING_AT_SCREEN" or row.get("eye_x"):
        return "screen"
    return "uncertain"


def _build_timeline(log: Path) -> dict:
    """Run-length encodes the frame log into [start_s, end_s, kind]
    segments; kinds: screen, glance, incident, no_face, uncertain."""
    segments: List[list] = []
    last_t, period = 0.0, 0.0
    with open(log, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = float(row["time_s"])
            if segments and t > last_t:
                period = t - last_t
            kind = _frame_kind(row)
            if segments and segments[-1][2] == kind:
                segments[-1][1] = t
            else:
                if segments:
                    segments[-1][1] = t
                segments.append([t, t, kind])
            last_t = t
    if segments:
        segments[-1][1] = last_t + period
    duration = segments[-1][1] if segments else 0.0
    return {"duration": round(duration, 3), "segments": [[round(a, 3), round(b, 3), k] for a, b, k in segments]}
