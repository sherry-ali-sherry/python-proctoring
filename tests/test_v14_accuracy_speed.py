"""Tests for the v1.4 accuracy and speed changes: mesh head pose, iris
offset, facing-away rule, eye smoothing, per-candidate limit floor,
frame sampling + refinement, rate-adjusted calibration, review-video copy,
and the evaluation tool."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from config import AppConfig  # noqa: E402
from face.mesh_pose import head_pose_from_transform  # noqa: E402
from gaze.eye_direction import (  # noqa: E402
    EyeBaseline, EyeDirection, effective_limits, eye_deviation, iris_offset, label_eye_frames, smooth_eyes,
)
from proctoring.state_manager import ProctorState  # noqa: E402
from video_analysis.incidents import REASON_FACING, FrameRecord, extract_incidents  # noqa: E402
from video_analysis.sampling import (  # noqa: E402
    choose_sample_indices, hold_measurements, nearest_measured, refine_frames,
)


def _rot_y(deg):
    a = np.radians(deg)
    m = np.eye(4)
    m[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    return m


def _rot_x(deg):
    a = np.radians(deg)
    m = np.eye(4)
    m[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    return m


# --------------------------------------------------------------------------- #
# Head pose from the face-mesh transform
# --------------------------------------------------------------------------- #
def test_mesh_pose_frontal_is_zero():
    hp = head_pose_from_transform(np.eye(4))
    assert abs(hp.yaw_deg) < 1e-6 and abs(hp.pitch_deg) < 1e-6 and abs(hp.roll_deg) < 1e-6


def test_mesh_pose_signs():
    # Face's forward axis rotated toward +x (image right) -> yaw > 0.
    assert head_pose_from_transform(_rot_y(30)).yaw_deg == pytest.approx(30, abs=0.1)
    # Forward axis tilted toward -y (down) -> pitch > 0 (chin down).
    assert head_pose_from_transform(_rot_x(20)).pitch_deg == pytest.approx(20, abs=0.1)
    assert head_pose_from_transform(None) is None
    assert head_pose_from_transform(np.full((4, 4), np.nan)) is None


# --------------------------------------------------------------------------- #
# Iris offset
# --------------------------------------------------------------------------- #
def _eye_landmarks(iris_dx=0.0, iris_dy=0.0):
    lm = np.zeros((478, 3))
    # left eye corners 33 (outer, image-left) / 133 (inner); right eye 362 / 263
    lm[33, :2], lm[133, :2] = (0.30, 0.40), (0.40, 0.40)
    lm[362, :2], lm[263, :2] = (0.60, 0.40), (0.70, 0.40)
    lm[468, :2] = (0.35 + iris_dx, 0.40 + iris_dy)
    lm[473, :2] = (0.65 + iris_dx, 0.40 + iris_dy)
    return lm


def test_iris_offset_directions():
    h, v = iris_offset(_eye_landmarks(), 1000, 1000)
    assert abs(h) < 1e-9 and abs(v) < 1e-9
    h, v = iris_offset(_eye_landmarks(iris_dx=0.02), 1000, 1000)
    assert h == pytest.approx(0.4)          # 20 px of a 50 px half-width, image right
    h, v = iris_offset(_eye_landmarks(iris_dy=-0.01), 1000, 1000)
    assert v == pytest.approx(0.2)          # moving up in the image is positive
    assert iris_offset(np.zeros((10, 3)), 100, 100) is None


# --------------------------------------------------------------------------- #
# Eye rule: smoothing, per-candidate floor, head term
# --------------------------------------------------------------------------- #
BASE = EyeBaseline(x=-0.25, v=-0.5, frames_used=100, prior_fraction=0.8, fallback=False, message="",
                   spread_x=0.06, spread_v=0.05)


def test_effective_limits_floor_only_for_noisy_candidates():
    ic = AppConfig().video.incidents
    assert effective_limits(BASE, ic)[0] == pytest.approx(ic.eye_limit_side)
    noisy = dataclasses.replace(BASE, spread_x=0.2)
    assert effective_limits(noisy, ic)[0] == pytest.approx(ic.eye_limit_spread_k * 0.2)


def test_head_term_adds_and_cancels():
    ic = dataclasses.replace(AppConfig().video.incidents, head_eye_deg_per_unit=40.0)
    e = EyeDirection(x=-0.25 + 0.2, up=0.0, down=0.5, blink=0.1)
    rx_alone, _ = eye_deviation(e, BASE, ic)
    rx_same, _ = eye_deviation(e, BASE, ic, head=(8.0, 0.0))       # head turned the same way
    rx_back, _ = eye_deviation(e, BASE, ic, head=(-8.0, 0.0))      # eyes compensating a head turn
    assert rx_same > rx_alone > abs(rx_back)
    assert abs(rx_back) < 1e-9


def test_smoothing_bridges_single_frame_dips():
    """'sidee' 19.5-20.7 s: eyes clearly to the side but jittering, so single
    frames dipped under the limit and the flicker filter erased the start."""
    ic = AppConfig().video.incidents
    t = [i / 30 for i in range(90)]
    xs = [(-0.25 - 0.42) if i % 3 else (-0.25 - 0.26) for i in range(90)]   # every 3rd frame inside
    eyes = [EyeDirection(x, 0.0, 0.5, 0.1) for x in xs]
    raw = label_eye_frames(t, eyes, BASE, dataclasses.replace(ic, eye_smoothing_s=0.0))
    smooth = label_eye_frames(t, eyes, BASE, dataclasses.replace(ic, eye_smoothing_s=0.2))
    assert sum(lab is not None for lab in raw) < 10
    assert sum(lab is not None for lab in smooth) > 80


def test_smooth_eyes_keeps_blinks_and_none():
    t = [0.0, 0.1, 0.2, 0.3]
    eyes = [EyeDirection(0.1, 0, 0, 0.1), None, EyeDirection(0.1, 0, 0, 0.9), EyeDirection(0.3, 0, 0, 0.1)]
    out = smooth_eyes(t, eyes, 0.25, 0.5)
    assert out[1] is None and out[2].blink == 0.9


# --------------------------------------------------------------------------- #
# Facing-away rule
# --------------------------------------------------------------------------- #
def test_facing_away_rule_makes_an_incident():
    cfg = AppConfig()
    recs = [FrameRecord(index=i, time_s=i / 30, num_faces=1, state=ProctorState.LOOKING_AT_SCREEN,
                        facing_away=(60 <= i < 240), head_cam_yaw=60.0 if 60 <= i < 240 else 0.0, head_cam_pitch=0.0)
            for i in range(300)]
    incs = extract_incidents(recs, cfg, 30.0, gaze_available=False)
    assert len(incs) == 1 and REASON_FACING in incs[0].reasons
    assert incs[0].start_s == pytest.approx(2.0, abs=0.05) and incs[0].direction == "RIGHT"


def test_camera_relative_angle_corrects_face_position():
    from face.head_pose import HeadPoseAngles
    from pipeline import FrameMeasurement
    from video_analysis.analyzer import _camera_relative
    tan_h = np.tan(np.radians(65.0) / 2)
    # Face at the left edge looking into the lens appears turned toward the
    # image right by the camera-ray angle; relative to the camera that is ~0.
    ray = np.degrees(np.arctan(0.35 * 2 * tan_h))
    m = FrameMeasurement(timestamp=0.0, num_faces=1, face_cx=0.15, face_cy=0.5)
    yaw, pitch = _camera_relative(HeadPoseAngles(ray, 0.0, 0.0), m, tan_h)
    assert abs(yaw) < 1e-6 and abs(pitch) < 1e-6


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def test_choose_sample_indices():
    t = [i / 30 for i in range(91)]
    idx = choose_sample_indices(t, 5.0)
    assert idx[0] == 0 and idx[-1] == 90
    assert all(5 <= b - a <= 7 for a, b in zip(idx, idx[1:-1]))
    assert choose_sample_indices(t, 0) == list(range(91))


def test_nearest_and_hold():
    assert list(nearest_measured(7, [0, 6])) == [0, 0, 0, 0, 6, 6, 6]
    from pipeline import FrameMeasurement
    t = [i / 30 for i in range(7)]
    held = hold_measurements(t, {0: FrameMeasurement(0.0, 1), 6: FrameMeasurement(t[6], 0)})
    assert [m.num_faces for m in held] == [1, 1, 1, 1, 0, 0, 0]
    assert held[2].timestamp == pytest.approx(t[2])


def test_refine_frames_only_around_changes():
    ic = AppConfig().video.incidents
    recs = [FrameRecord(index=i, time_s=i / 30, num_faces=1, state=ProctorState.LOOKING_AT_SCREEN,
                        eyes_away=("LEFT" if i >= 12 else None)) for i in range(25)]
    todo = refine_frames(recs, {0, 6, 12, 18, 24}, None, ic)
    assert todo == [7, 8, 9, 10, 11]


def test_rate_adjusted_config_scales_frame_counts():
    from video_analysis.analyzer import rate_adjusted_config
    cfg = AppConfig()
    assert rate_adjusted_config(cfg, [i / 30 for i in range(100)]) is cfg
    slow = rate_adjusted_config(cfg, [i / 5 for i in range(100)])
    assert slow.video.auto_calibration.min_samples == max(6, round(cfg.video.auto_calibration.min_samples * 5 / 30))
    assert cfg.video.auto_calibration.min_samples == 30      # the original is not modified


# --------------------------------------------------------------------------- #
# Review video copy
# --------------------------------------------------------------------------- #
def test_remux_copies_browser_ready_h264(tmp_path):
    av = pytest.importorskip("av")
    from fractions import Fraction
    from video_analysis.video_writer import remux_for_browser
    if "libx264" not in av.codecs_available:
        pytest.skip("no libx264")
    src = tmp_path / "src.mp4"
    with av.open(str(src), "w") as out:
        s = out.add_stream("libx264", rate=Fraction(30))
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(10):
            f = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 20, np.uint8), format="rgb24")
            f.pts = i
            for p in s.encode(f):
                out.mux(p)
        for p in s.encode():
            out.mux(p)
    got = remux_for_browser(str(src), tmp_path / "review", 720)
    assert got is not None and got.stat().st_size > 0
    with av.open(str(got)) as c:
        assert sum(1 for _ in c.decode(video=0)) == 10
    assert remux_for_browser(str(src), tmp_path / "small", 24) is None     # taller than allowed


# --------------------------------------------------------------------------- #
# Evaluation tool
# --------------------------------------------------------------------------- #
def test_evaluate_scores_events_and_false_alarms(tmp_path):
    import evaluate
    labels = [(0.0, 8.0, "SCREEN"), (8.0, 13.0, "AWAY"), (13.0, 20.0, "SCREEN"), (20.0, 25.0, "AWAY"),
              (25.0, 40.0, "SCREEN")]
    s = evaluate.score_video("x", labels, [(8.2, 12.9), (30.0, 34.0)])
    assert (s.periods, s.caught, s.false_alarms) == (2, 1, 1)
    assert s.missed == [(20.0, 25.0)]
    assert s.start_err[0] == pytest.approx(0.2) and s.end_err[0] == pytest.approx(0.1)
    p = tmp_path / "labels.csv"
    p.write_text("start_s,end_s,label\n0,1,SCREEN\n1,2,SCREEN\n2,3,AWAY\n")
    assert evaluate.read_labels(p) == [(0.0, 2.0, "SCREEN"), (2.0, 3.0, "AWAY")]
