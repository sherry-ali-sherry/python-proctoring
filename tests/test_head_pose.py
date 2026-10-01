"""
tests/test_head_pose.py

HeadPoseEstimator must report a frontal face as (0, 0, 0) and recover
known yaw/pitch/roll from a solvePnP-style rotation (face model -> OpenCV
camera coords, which includes a 180-degree flip about X for a frontal
face).
"""
import numpy as np
import pytest

from face.head_pose import HeadPoseEstimator

_FLIP = np.diag([1.0, -1.0, -1.0])


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _pnp_rotation(yaw, pitch, roll):
    viewer = _ry(np.radians(yaw)) @ _rx(np.radians(pitch)) @ _rz(np.radians(roll))
    return _FLIP @ viewer


def test_frontal_face_is_zero():
    a = HeadPoseEstimator.angles_from_rotation(_FLIP)
    assert a.yaw_deg == pytest.approx(0.0, abs=1e-9)
    assert a.pitch_deg == pytest.approx(0.0, abs=1e-9)
    assert a.roll_deg == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("yaw,pitch,roll", [
    (30, 0, 0), (-45, 0, 0), (0, 20, 0), (0, -25, 0), (0, 0, 15), (25, -15, 10), (-60, 30, -5),
])
def test_recovers_known_angles(yaw, pitch, roll):
    a = HeadPoseEstimator.angles_from_rotation(_pnp_rotation(yaw, pitch, roll))
    assert a.yaw_deg == pytest.approx(yaw, abs=1e-6)
    assert a.pitch_deg == pytest.approx(pitch, abs=1e-6)
    assert a.roll_deg == pytest.approx(roll, abs=1e-6)


def test_turn_is_yaw_not_pitch():
    """Regression: the previous decomposition reported a pure left/right
    head turn as pitch."""
    a = HeadPoseEstimator.angles_from_rotation(_pnp_rotation(40, 0, 0))
    assert abs(a.yaw_deg) == pytest.approx(40, abs=1e-6)
    assert abs(a.pitch_deg) < 1e-6
