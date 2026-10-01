"""
tests/test_geometry.py

Pure-geometry unit tests -- no webcam/MediaPipe required. Validates
the mathematical core: ray/ray closest approach, ray/plane
intersection, and gaze-ray construction, against hand-computable
cases.
"""
import numpy as np
import pytest

from gaze.gaze_ray import build_gaze_ray, GazeRay
from gaze.binocular_gaze import closest_approach, combine_gaze_rays
from gaze.screen_plane import make_default_screen_plane, intersect_ray_plane


def test_build_gaze_ray_direction_and_normalization():
    eye_center = np.array([0.0, 0.0, 500.0])
    iris = np.array([10.0, 0.0, 500.0])
    ray = build_gaze_ray(eye_center, iris, confidence=0.9)
    assert ray is not None
    assert np.isclose(np.linalg.norm(ray.direction), 1.0)
    assert np.allclose(ray.direction, np.array([1.0, 0.0, 0.0]))


def test_build_gaze_ray_degenerate_returns_none():
    eye_center = np.array([1.0, 2.0, 3.0])
    ray = build_gaze_ray(eye_center, eye_center.copy(), confidence=0.9)
    assert ray is None


def test_closest_approach_intersecting_rays():
    # Two rays that genuinely cross at (0,0,10).
    left = GazeRay(np.array([-10.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0]) / np.sqrt(2), 1.0)
    right = GazeRay(np.array([10.0, 0.0, 0.0]), np.array([-1.0, 0.0, 1.0]) / np.sqrt(2), 1.0)
    result = closest_approach(left, right)
    assert np.allclose(result.midpoint, np.array([0.0, 0.0, 10.0]), atol=1e-6)
    assert result.separation_distance < 1e-6


def test_closest_approach_parallel_rays_falls_back_to_origin_midpoint():
    left = GazeRay(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 1.0)
    right = GazeRay(np.array([10.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 1.0)
    result = closest_approach(left, right)
    assert np.allclose(result.midpoint, np.array([5.0, 0.0, 0.0]))


def test_combine_gaze_rays_monocular_fallback():
    left = GazeRay(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0.8)
    combined = combine_gaze_rays(left, None)
    assert combined is not None
    assert np.allclose(combined.direction, left.direction)
    assert combined.confidence == pytest.approx(0.8 * 0.7)


def test_ray_plane_intersection_center_hit():
    plane = make_default_screen_plane(distance_mm=600.0)
    ray = GazeRay(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 1.0)
    result = intersect_ray_plane(ray, plane)
    assert result.valid
    assert np.isclose(result.plane_u, 0.0)
    assert np.isclose(result.plane_v, 0.0)
    assert np.allclose(result.point_3d, np.array([0.0, 0.0, 600.0]))


def test_ray_plane_intersection_parallel_is_invalid():
    plane = make_default_screen_plane(distance_mm=600.0)
    ray = GazeRay(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 1.0)
    result = intersect_ray_plane(ray, plane)
    assert not result.valid
    assert result.ray_plane_parallel


def test_ray_plane_intersection_offset_point():
    plane = make_default_screen_plane(distance_mm=600.0)
    # Ray angled slightly in +X, +Y (image-down convention -> plane v axis is -Y)
    direction = np.array([50.0, -30.0, 600.0])
    direction = direction / np.linalg.norm(direction)
    ray = GazeRay(np.array([0.0, 0.0, 0.0]), direction, 1.0)
    result = intersect_ray_plane(ray, plane)
    assert result.valid
    assert result.point_3d[2] > 0
    # plane_u should be positive (same sign as X displacement)
    assert result.plane_u > 0
    # plane_v should be positive too (v_axis = -Y, and displacement Y is negative)
    assert result.plane_v > 0
