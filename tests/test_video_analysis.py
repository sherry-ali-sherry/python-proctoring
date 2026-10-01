"""
tests/test_video_analysis.py

Auto-calibration, results-folder layout, and a replay of the real
GazePipeline.decide() (classifier + temporal filter + hysteresis) over
synthetic measurements with video timestamps. No MediaPipe needed.
"""
import numpy as np
import pytest

from config import AppConfig
from face.head_pose import HeadPoseAngles
from gaze.gaze_ray import GazeRay
from pipeline import FrameMeasurement, GazePipeline
from proctoring.state_manager import ProctorState
from video_analysis.auto_calibration import fit_auto_calibration
from video_analysis.incidents import extract_incidents
from video_analysis.output_layout import create_result_folders, sanitize_folder_name, validate_candidate_name

EYE = np.array([0.0, 0.0, 3000.0])      # eye ~3000 model units in front of the camera


def _ray_towards(yaw_deg, pitch_deg, conf=0.9):
    """Gaze ray from EYE, rotated from 'straight at the camera' by
    yaw (+ = image right) and pitch (+ = up)."""
    d = np.array([np.tan(np.radians(yaw_deg)), -np.tan(np.radians(pitch_deg)), -1.0])
    return GazeRay(EYE.copy(), d / np.linalg.norm(d), conf)


def _m(t, yaw_deg, pitch_deg, faces=1, conf=0.9):
    return FrameMeasurement(
        timestamp=t, num_faces=faces, pose_ok=faces > 0, face_confidence=1.0 if faces else 0.0,
        reprojection_error_px=2.0, head_pose=HeadPoseAngles(0.0, 10.0, 0.0) if faces else None,
        combined_ray=_ray_towards(yaw_deg, pitch_deg, conf) if faces else None,
    )


# The screen sits below the webcam: looking at it means gaze ~12 deg down.
SCREEN_PITCH = -12.0


def test_auto_calibration_centers_on_dominant_gaze():
    cfg = AppConfig()
    rng = np.random.default_rng(0)
    ms = [_m(i / 30, rng.normal(0, 3), SCREEN_PITCH + rng.normal(0, 2)) for i in range(900)]
    ms += [_m(30 + i / 30, 45.0, SCREEN_PITCH) for i in range(100)]      # a minority looking far right
    summary = fit_auto_calibration(ms, cfg)
    assert summary.success and summary.result.quality == "AUTO"
    assert summary.screen_consistency > 0.85

    from gaze.screen_plane import intersect_ray_plane
    plane_distance = summary.result.plane_distance_mm
    from gaze.screen_plane import make_default_screen_plane
    plane = make_default_screen_plane(plane_distance)

    def to_screen(yaw, pitch):
        hit = intersect_ray_plane(_ray_towards(yaw, pitch), plane)
        return summary.result.affine_matrix @ np.array([hit.plane_u, hit.plane_v]) + summary.result.affine_offset

    sx, sy = to_screen(0.0, SCREEN_PITCH)
    assert abs(sx) < 0.1 and abs(sy) < 0.1
    sx, _ = to_screen(cfg.video.auto_calibration.screen_half_angle_x_deg, SCREEN_PITCH)
    assert sx == pytest.approx(1.0, abs=0.05)
    sx, _ = to_screen(45.0, SCREEN_PITCH)
    assert sx > 1.5


def test_auto_calibration_fails_without_enough_confident_frames():
    cfg = AppConfig()
    ms = [_m(i / 30, 0, 0, conf=0.1) for i in range(500)] + [_m(20 + i / 30, 0, 0) for i in range(10)]
    summary = fit_auto_calibration(ms, cfg)
    assert not summary.success and summary.result is None and summary.samples_used == 10


def test_decide_replay_detects_look_away_with_video_time():
    cfg = AppConfig()
    fps = 30.0
    timeline = [(0.0, 20.0)] * 1 + [(40.0, 6.0)] + [(0.0, 20.0)]   # (yaw, seconds)
    ms, t = [], 0.0
    for yaw, seconds in timeline:
        for _ in range(int(seconds * fps)):
            ms.append(_m(t, yaw, SCREEN_PITCH))
            t += 1 / fps

    pipeline = GazePipeline(cfg)
    summary = fit_auto_calibration(ms, cfg)
    pipeline.calibration.result = summary.result

    from video_analysis.incidents import FrameRecord
    records = []
    for i, m in enumerate(ms):
        d = pipeline.decide(m, m.timestamp)
        records.append(FrameRecord(index=i, time_s=m.timestamp, num_faces=1, state=d.state,
                                   away_direction=d.gaze_result.away_direction if d.gaze_result else None))
    assert records[int(10 * fps)].state == ProctorState.LOOKING_AT_SCREEN
    assert records[int(25 * fps)].state == ProctorState.LOOKING_AWAY

    incs = extract_incidents(records, cfg, fps)
    assert len(incs) == 1
    inc = incs[0]
    # Gaze left at 20.0 s and returned at 26.0 s; EMA smoothing delays
    # both edges by a few frames.
    assert inc.start_s == pytest.approx(20.0, abs=0.3)
    assert inc.end_s == pytest.approx(26.0, abs=0.3)
    assert inc.direction == "RIGHT"
    # LOOKING_AWAY_STARTED is time-stamped in video time, not wall-clock.
    started = [e for e in pipeline.event_engine.events if e.event_type == "LOOKING_AWAY_STARTED"]
    assert len(started) == 1 and 23.0 < started[0].timestamp < 24.0


@pytest.mark.parametrize("name,expected", [
    ("Ali Khan", "Ali_Khan"),
    ("  Sara   O'Neil  ", "Sara_O'Neil"),
    ('bad<>:"/\\|?*name', "badname"),
    ("CON", "_CON"),
    ("Zoë Müller", "Zoë_Müller"),
    ("...", ""),
])
def test_sanitize_folder_name(name, expected):
    assert sanitize_folder_name(name) == expected


def test_validate_candidate_name():
    assert validate_candidate_name("") is not None
    assert validate_candidate_name("   ") is not None
    assert validate_candidate_name("???") is not None
    assert validate_candidate_name("x" * 81) is not None
    assert validate_candidate_name("Ali Khan") is None


def test_result_folders_never_overwrite(tmp_path):
    first = create_result_folders(tmp_path, "Ali Khan")
    assert first.root == tmp_path / "Ali_Khan"
    for sub in (first.recordings, first.snapshots, first.report, first.data):
        assert sub.is_dir()
    (first.report / "report.html").write_text("x")
    second = create_result_folders(tmp_path, "Ali Khan", now=0)
    assert second.root != first.root and second.root.name.startswith("Ali_Khan_")
    assert (first.report / "report.html").read_text() == "x"
    third = create_result_folders(tmp_path, "Ali Khan", now=0)
    assert third.root not in (first.root, second.root)
