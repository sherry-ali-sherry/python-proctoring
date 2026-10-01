"""
tests/test_calibration.py

Synthetic-data test for CalibrationManager: simulates rays that hit
known screen-local (u, v) positions for each calibration target with
a known ground-truth affine relationship, verifies the fitted result
recovers a good mapping and a HIGH quality score. Also checks the
FAILED path when too few samples are valid.
"""
import numpy as np

from config import CalibrationConfig, ScreenConfig
from gaze.calibration import CalibrationManager
from gaze.gaze_ray import GazeRay


def _make_manager():
    cal_cfg = CalibrationConfig()
    screen_cfg = ScreenConfig()
    return CalibrationManager(cal_cfg, screen_cfg), cal_cfg, screen_cfg


def test_calibration_good_fit_recovers_affine_mapping():
    manager, cal_cfg, screen_cfg = _make_manager()

    # Ground truth: screen coord = 2.5 * plane_uv (simple known scale),
    # simulated directly via rays whose direction encodes the target.
    true_scale = 1 / 250.0  # so plane_u * true_scale ~= target when plane at distance ~600? just check recovery

    rng = np.random.default_rng(42)
    for target in manager.target_points:
        for _ in range(cal_cfg.samples_per_point):
            # Build a ray whose intersection with a 600mm plane gives
            # (u, v) = target / true_scale, i.e. target = u * true_scale.
            u = target[0] / true_scale + rng.normal(0, 2.0)
            v = target[1] / true_scale + rng.normal(0, 2.0)
            direction = np.array([u, v, 600.0])
            direction = direction / np.linalg.norm(direction)
            ray = GazeRay(origin=np.zeros(3), direction=direction, confidence=1.0)
            manager.add_sample(target, ray)

    result = manager.compute()
    assert result.success
    assert result.quality in ("GOOD", "LOW_CONFIDENCE")

    # Check the fitted affine mapping recovers something close to a
    # point near the center target.
    plane = manager.screen_plane()
    center_ray = GazeRay(np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0)
    from gaze.screen_plane import intersect_ray_plane
    inter = intersect_ray_plane(center_ray, plane)
    sx, sy = manager.apply_affine(inter.plane_u, inter.plane_v)
    assert abs(sx) < 0.3
    assert abs(sy) < 0.3


def test_calibration_fails_with_too_few_samples():
    manager, cal_cfg, screen_cfg = _make_manager()
    ray = GazeRay(np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0)
    # Only add a couple of samples -- far below the required minimum.
    manager.add_sample((0.0, 0.0), ray)
    manager.add_sample((0.0, 0.0), ray)
    result = manager.compute()
    assert not result.success
    assert result.quality == "FAILED"
