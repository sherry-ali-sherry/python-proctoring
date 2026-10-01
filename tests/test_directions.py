"""
tests/test_directions.py

Eight-way directions from the 3D-gaze classifier and the head-turn rule.
"""
import numpy as np
import pytest

from config import AppConfig
from gaze.calibration import CalibrationManager, CalibrationResult
from gaze.gaze_classifier import GazeClassifier, RawGazeLabel
from gaze.gaze_ray import GazeRay
from gaze.directions import outside_ellipse


def _classifier():
    cfg = AppConfig()
    cal = CalibrationManager(cfg.calibration, cfg.screen)
    cal.result = CalibrationResult(True, "GOOD", 1.0, "", np.eye(2), np.zeros(2), 0.0)
    return GazeClassifier(cfg, cal)


def _direction_at(sx, sy):
    # Ray straight at the camera plane (z = 0), hitting it at (u, v) = (sx, sy).
    ray = GazeRay(origin=np.array([sx, -sy, 600.0]), direction=np.array([0.0, 0.0, -1.0]), confidence=1.0)
    r = _classifier().classify(ray, None, 1.0, 0.0)
    return r.label, r.away_direction


@pytest.mark.parametrize("sx, sy, expected", [
    (0.0, 0.0, None),
    (1.6, 0.0, "RIGHT"),
    (0.0, 1.6, "UP"),
    (-1.6, -1.5, "DOWN-LEFT"),      # past the lower-left corner
    (1.5, 1.4, "UP-RIGHT"),
    (1.8, -1.05, "RIGHT"),          # far right, barely below: not a corner
])
def test_gaze_classifier_reports_eight_directions(sx, sy, expected):
    label, direction = _direction_at(sx, sy)
    assert direction == expected
    assert label == (RawGazeLabel.SCREEN if expected is None else RawGazeLabel.AWAY)


def test_head_turn_ellipse_catches_diagonal_turns():
    ic = AppConfig().video.incidents
    # 25 deg sideways and 22 deg down: neither axis alone passes 35 / 30.
    assert outside_ellipse(25 / ic.head_turn_yaw_deg, 22 / ic.head_turn_pitch_deg)
    assert not outside_ellipse(20 / ic.head_turn_yaw_deg, 10 / ic.head_turn_pitch_deg)
