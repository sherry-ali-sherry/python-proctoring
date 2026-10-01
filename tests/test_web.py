"""
tests/test_web.py

Web API end to end with FastAPI's TestClient. The analyzer is replaced
by a fast fake that writes a real results folder (via the real report /
folder code), so the queue, upload handling, review flow, file serving
and settings are exercised without running MediaPipe.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import settings_store
from proctoring.state_manager import ProctorState as S
from video_analysis.analyzer import AnalysisCancelled, AnalysisResult
from video_analysis.incidents import FrameRecord, compute_stats, compute_verdict, extract_incidents
from video_analysis.output_layout import create_result_folders
from video_analysis.report import ReportContext, write_data_files, write_reports
from video_analysis.video_source import probe_video
from web.app import create_app
from web.db import STATUS_QUEUED, STATUS_RUNNING, Store

FPS = 10.0


def _make_video(path: Path, seconds: float = 2.0) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    for i in range(int(seconds * FPS)):
        w.write(np.full((48, 64, 3), i % 255, np.uint8))
    w.release()
    return path


class FakeAnalyzer:
    def __init__(self, cfg, project_dir):
        self.cfg, self.project_dir = cfg, Path(project_dir)

    def run(self, name, video, mode, progress=None, cancel_event=None):
        progress(0.5, "Halfway")
        if cancel_event is not None and cancel_event.is_set():
            raise AnalysisCancelled()
        info = probe_video(video)
        spec = [(10, S.LOOKING_AT_SCREEN), (35, S.POSSIBLE_AWAY), (60, S.LOOKING_AWAY), (15, S.LOOKING_AT_SCREEN)]
        records, idx = [], 0
        for n, state in spec:
            for _ in range(n):
                records.append(FrameRecord(index=idx, time_s=idx / FPS, num_faces=1, state=state,
                                           away_direction="LEFT" if state != S.LOOKING_AT_SCREEN else None))
                idx += 1
        incidents = extract_incidents(records, self.cfg, FPS)
        stats = compute_stats(records, incidents, FPS)
        verdict = compute_verdict(stats, self.cfg, True)
        paths = create_result_folders(self.project_dir / self.cfg.video.results_dir, name)
        write_data_files(paths, records, [], {"mode": mode})
        ctx = ReportContext(name, info, time.time(), 1.0, mode, "fake", True, stats, verdict, incidents)
        html = write_reports(paths, ctx, self.cfg)
        return AnalysisResult(name, paths, html, verdict, stats, incidents, [])


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", tmp_path / "settings.json")
    return tmp_path


def _client(project: Path, start_worker=True) -> TestClient:
    app = create_app(project, db_path=project / "test.db", analyzer_factory=FakeAnalyzer, start_worker=start_worker)
    return TestClient(app)


def _upload(client, video: Path, name="Ali Khan", filename=None):
    with open(video, "rb") as f:
        return client.post("/api/analyses", data={"candidate_name": name, "calibration_mode": "auto"},
                           files={"file": (filename or video.name, f, "video/mp4")})


def _wait_done(client, analysis_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = client.get(f"/api/analyses/{analysis_id}").json()
        if d["status"] not in (STATUS_QUEUED, STATUS_RUNNING):
            return d
        time.sleep(0.05)
    raise AssertionError("analysis did not finish")


def test_upload_analyze_review_flow(env):
    video = _make_video(env / "exam.mp4")
    with _client(env) as client:
        r = _upload(client, video, name="  Ali   Khan ")
        assert r.status_code == 201, r.text
        created = r.json()
        assert created["candidate_name"] == "Ali Khan" and created["status"] == STATUS_QUEUED

        d = _wait_done(client, created["id"])
        assert d["status"] == "done", d
        assert d["auto_verdict"] == "CHEATING" and d["incident_count"] == 1 and not d["reviewed"]
        results = Path(d["results_dir"])
        assert results == env / "results" / "Ali_Khan"
        # upload moved into the results folder, report points at it
        assert (results / "source" / "exam.mp4").is_file()
        assert not any((env / "uploads").iterdir())
        assert d["report"]["video"]["path"] == str(results / "source" / "exam.mp4")

        # listing
        rows = client.get("/api/analyses").json()
        assert [r["id"] for r in rows] == [created["id"]]

        # timeline
        tl = client.get(f"/api/analyses/{created['id']}/timeline").json()
        kinds = [s[2] for s in tl["segments"]]
        assert kinds == ["screen", "incident", "screen"]
        assert tl["duration"] == pytest.approx(12.0)

        # incident review
        base = f"/api/analyses/{created['id']}"
        assert client.put(f"{base}/incidents/1/review", json={"status": "confirmed", "note": "", "reviewer": " "}).status_code == 400
        assert client.put(f"{base}/incidents/9/review", json={"status": "confirmed", "note": "", "reviewer": "R"}).status_code == 404
        assert client.put(f"{base}/incidents/1/review", json={"status": "bogus", "note": "", "reviewer": "R"}).status_code == 400
        r = client.put(f"{base}/incidents/1/review", json={"status": "confirmed", "note": "Looked at a phone", "reviewer": "Dr  Sara"})
        assert r.status_code == 200 and r.json()["incidents"]["1"]["status"] == "confirmed"

        # final decision overrides the automatic verdict
        r = client.put(f"{base}/review", json={"decision": "NO CHEATING", "note": "Reading allowed notes", "reviewer": "Dr Sara"})
        assert r.status_code == 200
        s = r.json()
        assert s["reviewed"] and s["final_verdict"] == "NO CHEATING" and s["effective_verdict"] == "NO CHEATING"
        review = json.loads((results / "report" / "review.json").read_text(encoding="utf-8"))
        assert review["final_verdict"] == "NO CHEATING" and review["incidents"]["1"]["note"] == "Looked at a phone"
        html = (results / "report" / "report.html").read_text(encoding="utf-8")
        assert "Reviewer decision" in html and "Dr Sara" in html and "Confirmed by reviewer" in html
        assert "confirmed" in (results / "report" / "incidents.csv").read_text(encoding="utf-8")

        # accepting the automatic verdict clears the override
        s = client.put(f"{base}/review", json={"decision": "accept", "note": "", "reviewer": "Dr Sara"}).json()
        assert s["final_verdict"] is None and s["effective_verdict"] == "CHEATING" and s["reviewed"]
        assert client.put(f"{base}/review", json={"decision": "MAYBE", "note": "", "reviewer": "x"}).status_code == 400

        # files: serving, range requests, traversal protection
        f = client.get(f"/files/{created['id']}/report/report.html")
        assert f.status_code == 200 and f.headers["content-type"].startswith("text/html")
        part = client.get(f"/files/{created['id']}/source/exam.mp4", headers={"Range": "bytes=0-99"})
        assert part.status_code == 206 and len(part.content) == 100
        assert client.get(f"/files/{created['id']}/../test.db").status_code == 404
        assert client.get(f"/files/{created['id']}/..%2F..%2Ftest.db").status_code == 404

        # second analysis for the same name never overwrites the first
        d2 = _wait_done(client, _upload(client, video).json()["id"])
        assert Path(d2["results_dir"]) != results and (results / "report" / "review.json").is_file()


def test_upload_validation(env):
    video = _make_video(env / "exam.mp4")
    (env / "notes.txt").write_text("hello")
    (env / "fake.mp4").write_bytes(b"not really a video")
    with _client(env, start_worker=False) as client:
        assert _upload(client, video, name="   ").status_code == 400
        assert _upload(client, video, name="???").status_code == 400
        r = _upload(client, env / "notes.txt")
        assert r.status_code == 400 and "Unsupported file type" in r.json()["detail"]
        r = _upload(client, env / "fake.mp4")
        assert r.status_code == 400
        assert not any((env / "uploads").iterdir())        # rejected uploads are removed
        assert client.get("/api/analyses").json() == []
        # path components in the client's filename are ignored
        r = _upload(client, video, filename="..\\..\\evil name.mp4")
        assert r.status_code == 201
        assert r.json()["video_filename"] == "evil_name.mp4"


def test_cancel_and_remove_queued(env):
    video = _make_video(env / "exam.mp4")
    with _client(env, start_worker=False) as client:
        a = _upload(client, video).json()
        assert (env / "uploads" / a["id"]).is_dir()
        assert client.delete(f"/api/analyses/{a['id']}").status_code == 409
        r = client.post(f"/api/analyses/{a['id']}/cancel")
        assert r.status_code == 200 and r.json()["status"] == "cancelled"
        assert not (env / "uploads" / a["id"]).exists()
        assert client.post(f"/api/analyses/{a['id']}/cancel").status_code == 409
        assert client.get(f"/api/analyses/{a['id']}/timeline").status_code == 409
        assert client.delete(f"/api/analyses/{a['id']}").status_code == 204
        assert client.get(f"/api/analyses/{a['id']}").status_code == 404


def test_delete_finished_analysis(env):
    video = _make_video(env / "exam.mp4")
    with _client(env) as client:
        keep = _wait_done(client, _upload(client, video, name="Keep Me").json()["id"])
        gone = _wait_done(client, _upload(client, video, name="Delete Me").json()["id"])
        base = f"/api/analyses/{gone['id']}"
        client.put(f"{base}/incidents/1/review", json={"status": "confirmed", "note": "x", "reviewer": "R"})
        client.put(f"{base}/review", json={"decision": "accept", "note": "", "reviewer": "R"})
        folder = Path(gone["results_dir"])
        assert folder.is_dir()

        assert client.delete(base).status_code == 204
        assert not folder.exists()
        assert client.get(base).status_code == 404
        assert client.get(f"/files/{gone['id']}/report/report.html").status_code == 404
        assert client.app.state.store.incident_reviews(gone["id"]) == {}
        assert client.delete(base).status_code == 404
        # the other candidate is untouched
        assert Path(keep["results_dir"]).is_dir()
        assert [r["candidate_name"] for r in client.get("/api/analyses").json()] == ["Keep Me"]
    with _client(env, start_worker=False) as client:      # not re-imported from disk after a restart
        assert [r["candidate_name"] for r in client.get("/api/analyses").json()] == ["Keep Me"]


def test_delete_refuses_folders_outside_results(env):
    outside = env / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("do not delete")
    store = Store(env / "test.db")
    store.import_finished("evil", "X", "x.mp4", str(outside), time.time(), "NO CHEATING", 0, 0.0, 1.0)
    store.close()
    with _client(env, start_worker=False) as client:
        r = client.delete("/api/analyses/evil")
        assert r.status_code == 409 and "outside" in r.json()["detail"]
        assert (outside / "keep.txt").read_text() == "do not delete"
        assert client.get("/api/analyses/evil").status_code == 200


def test_interrupted_analysis_is_requeued(env):
    store = Store(env / "test.db")
    store.create("abc123", "Ali", "exam.mp4", str(_make_video(env / "exam.mp4")), "auto")
    store.mark_running("abc123")
    assert store.requeue_interrupted() == 1
    assert store.get("abc123")["status"] == STATUS_QUEUED
    store.close()


def test_existing_results_folders_are_imported(env):
    video = _make_video(env / "exam.mp4")
    cfg = settings_store.load_config()
    FakeAnalyzer(cfg, env).run("Desktop Made", str(video), "auto", progress=lambda *a: None)
    (env / "results" / "partial" / "report").mkdir(parents=True)     # unfinished folder: ignored
    with _client(env, start_worker=False) as client:
        rows = client.get("/api/analyses").json()
        assert [r["candidate_name"] for r in rows] == ["Desktop Made"]
        assert rows[0]["status"] == "done"
    with _client(env, start_worker=False) as client:                  # not imported twice
        assert len(client.get("/api/analyses").json()) == 1


def test_settings_api(env):
    with _client(env, start_worker=False) as client:
        fields = {f["key"]: f for f in client.get("/api/settings").json()}
        assert fields["video.verdict.min_incidents"]["value"] == 3
        r = client.put("/api/settings", json={"values": {"video.verdict.min_incidents": 5,
                                                         "video.verdict.min_not_looking_fraction": 0.2}})
        assert r.status_code == 200
        fields = {f["key"]: f for f in r.json()}
        assert fields["video.verdict.min_incidents"]["value"] == 5
        assert settings_store.load_config().video.verdict.min_not_looking_fraction == 0.2

        bad = client.put("/api/settings", json={"values": {"video.verdict.min_incidents": 0}})
        assert bad.status_code == 422 and "at least" in bad.json()["detail"][0]
        assert client.put("/api/settings", json={"values": {"video.verdict.min_incidents": 2.5}}).status_code == 422
        assert client.put("/api/settings", json={"values": {"video.clip_padding_s": "3"}}).status_code == 422
        assert client.put("/api/settings", json={"values": {"camera.device_index": 1}}).status_code == 422
        assert settings_store.load_config().video.verdict.min_incidents == 5     # unchanged by bad writes

        fields = {f["key"]: f for f in client.post("/api/settings/reset").json()}
        assert fields["video.verdict.min_incidents"]["value"] == 3


def test_index_and_static(env):
    with _client(env, start_worker=False) as client:
        assert "ExamVision" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/styles.css").status_code == 200
