"""
tests/test_gaze_fixes.py

Regression tests for three defects that made the gaze pipeline report
UNCERTAIN on almost every frame of a clear, frontal face:
  1. iris validity computed over a "ring" that included the iris CENTER
     landmark, capping validity at ~0.5;
  2. the combined ray originating at the (unstable) fixation point;
  3. a binocular disagreement penalty scaled for millimetres at 600 mm,
     not for the face model's units at real viewing distances.
"""
import numpy as np
import pytest

from config import ConfidenceConfig, LEFT_IRIS_CONTOUR, LEFT_IRIS_RING, RIGHT_IRIS_CONTOUR, RIGHT_IRIS_RING
from face.face_geometry import FaceGeometryEstimator
from gaze.binocular_gaze import combine_gaze_rays
from gaze.gaze_ray import GazeRay
from gaze.iris_tracker import IrisTracker


def _mediapipe_style_iris(center, rx=7.8, ry=6.4):
    """Center point followed by 4 contour points (right, top, left, bottom)."""
    cx, cy = center
    return np.array([[cx, cy], [cx + rx, cy], [cx, cy - ry], [cx - rx, cy], [cx, cy + ry]])


def test_contour_indices_exclude_center():
    assert LEFT_IRIS_CONTOUR == LEFT_IRIS_RING[1:]
    assert RIGHT_IRIS_CONTOUR == RIGHT_IRIS_RING[1:]


def test_iris_validity_uses_contour_only():
    ring = _mediapipe_style_iris((300.0, 200.0))
    _, with_center = IrisTracker._robust_center(ring)
    center, contour_only = IrisTracker._robust_center(ring[1:])
    assert with_center < 0.55               # the old, capped behaviour
    assert contour_only > 0.85
    assert np.allclose(center, [300.0, 200.0])
    assert contour_only >= ConfidenceConfig().min_iris_validity


def _ray(origin, target, conf=0.9):
    d = np.asarray(target, float) - np.asarray(origin, float)
    return GazeRay(np.asarray(origin, float), d / np.linalg.norm(d), conf)


def test_combined_origin_is_between_the_eyes():
    left_eye, right_eye = [-200.0, 0.0, 3500.0], [200.0, 0.0, 3500.0]
    # slight vertical skew: converge near the camera but not exactly
    left = _ray(left_eye, [0.0, 10.0, 0.0])
    right = _ray(right_eye, [0.0, -10.0, 0.0])
    combined = combine_gaze_rays(left, right)
    assert np.allclose(combined.origin, [0.0, 0.0, 3500.0])


def test_vergence_on_screen_is_not_penalized():
    """Both eyes fixating the same point: rays intersect, no penalty."""
    left = _ray([-200.0, 0.0, 3500.0], [0.0, 300.0, 0.0])
    right = _ray([200.0, 0.0, 3500.0], [0.0, 300.0, 0.0])
    combined = combine_gaze_rays(left, right)
    assert combined.confidence == pytest.approx(0.9, abs=1e-6)


def test_disagreement_penalty_is_scale_invariant():
    """The same skew angle gives the same confidence whatever the unit
    scale / viewing distance."""
    def conf(scale):
        left = _ray(np.array([-200.0, 0.0, 3500.0]) * scale, np.array([0.0, 60.0, 0.0]) * scale)
        right = _ray(np.array([200.0, 0.0, 3500.0]) * scale, np.array([0.0, -60.0, 0.0]) * scale)
        return combine_gaze_rays(left, right).confidence
    assert conf(1.0) == pytest.approx(conf(0.2), abs=1e-9)
    assert 0.45 < conf(1.0) < 0.9


def test_small_vertical_noise_keeps_confidence_above_threshold():
    """~2 degrees of vertical disagreement (typical landmark noise) must
    not push a good binocular measurement under min_gaze_confidence."""
    left = _ray([-200.0, 0.0, 3500.0], [0.0, 60.0, 0.0])
    right = _ray([200.0, 0.0, 3500.0], [0.0, -60.0, 0.0])
    assert combine_gaze_rays(left, right).confidence > ConfidenceConfig().min_gaze_confidence + 0.2
