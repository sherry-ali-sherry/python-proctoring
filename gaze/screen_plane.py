"""
gaze/screen_plane.py

Represents the virtual screen as a plane in camera coordinates and
performs ray/plane intersection.

Documented assumption (see README "Known Limitations"): a single
uncalibrated monocular webcam cannot observe the true tilt of the
physical screen relative to the camera. We therefore assume the
screen plane is parallel to the camera's image plane (normal along
-Z) at a distance solved during calibration, and let the calibration
step's affine correction (see calibration.py) absorb the residual
error from any real tilt/offset. This is a standard practical
simplification for monocular webcam gaze systems and is explicitly
NOT presented as metrically exact screen geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gaze.gaze_ray import GazeRay


@dataclass
class ScreenPlane:
    origin: np.ndarray          # P: a point on the plane, camera coords
    normal: np.ndarray          # N: normalized plane normal, camera coords
    u_axis: np.ndarray          # screen "horizontal" unit axis, camera coords
    v_axis: np.ndarray          # screen "vertical" unit axis, camera coords


@dataclass
class Intersection:
    valid: bool
    point_3d: Optional[np.ndarray]     # camera coords, on the plane
    plane_u: float                     # local plane coordinate (unscaled)
    plane_v: float
    ray_plane_parallel: bool


def intersect_ray_plane(ray: GazeRay, plane: ScreenPlane, min_dot: float = 1e-3) -> Intersection:
    """t = ((P - O) . N) / (D . N)"""
    denom = float(np.dot(ray.direction, plane.normal))
    if abs(denom) < min_dot:
        return Intersection(False, None, 0.0, 0.0, True)

    t = float(np.dot(plane.origin - ray.origin, plane.normal)) / denom
    if t < 0:
        # Screen is behind the ray's forward direction -- invalid.
        return Intersection(False, None, 0.0, 0.0, False)

    point = ray.point_at(t)
    rel = point - plane.origin
    u = float(np.dot(rel, plane.u_axis))
    v = float(np.dot(rel, plane.v_axis))
    return Intersection(True, point, u, v, False)


def make_default_screen_plane(distance_mm: float) -> ScreenPlane:
    """Initial (pre-calibration) plane: parallel to the camera sensor
    plane, `distance_mm` in front of the camera along +Z."""
    return ScreenPlane(
        origin=np.array([0.0, 0.0, distance_mm]),
        normal=np.array([0.0, 0.0, -1.0]),   # points back toward the camera
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, -1.0, 0.0]),   # image Y is down; screen "up" is -Y
    )
