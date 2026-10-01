"""
gaze/iris_tracker.py

Estimates a robust 2D iris center from the MediaPipe iris landmark
ring (5 points per eye), validates it, and unprojects it into 3D
camera coordinates.

3D unprojection method: the eyeball is modeled as a sphere of radius
`eyeball_radius_mm` centered at the eye center (from eye_model.py).
The iris center's 3D position is taken as the intersection of the
camera ray through the 2D iris-center pixel with that sphere -- i.e.
genuine ray-sphere geometry, not an arbitrary depth guess. Of the (up
to) two intersections, the one closer to the camera is used (the
front, visible surface of the eyeball).

This keeps the iris 3D position physically constrained to lie on the
modeled eyeball surface, which is what makes the resulting gaze
DIRECTION (iris - eye_center) meaningful rather than an artifact of an
assumed constant depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import ConfidenceConfig
from face.face_geometry import FacePose, FaceGeometryEstimator


@dataclass
class IrisEstimate:
    valid: bool
    center_px: Optional[np.ndarray]
    center_3d: Optional[np.ndarray]   # camera coords
    validity_score: float             # 0..1, ring-consistency based


class IrisTracker:
    def __init__(self, confidence_cfg: ConfidenceConfig, geometry: FaceGeometryEstimator):
        self._cfg = confidence_cfg
        self._geometry = geometry

    @staticmethod
    def _robust_center(ring_px: np.ndarray) -> tuple[np.ndarray, float]:
        """Robust centroid + a validity score derived from how tightly
        the ring points cluster around a plausible circle (closed eyes,
        occlusion, or landmark failure make the ring degenerate)."""
        center = np.mean(ring_px, axis=0)
        radii = np.linalg.norm(ring_px - center, axis=1)
        mean_r = float(np.mean(radii))
        if mean_r < 1e-6:
            return center, 0.0
        # Coefficient of variation of the ring radius: low CV => the
        # 5 points form a plausible small circle => trustworthy.
        cv = float(np.std(radii) / mean_r)
        validity = float(np.clip(1.0 - cv, 0.0, 1.0))
        return center, validity

    def estimate(
        self,
        ring_px: np.ndarray,
        eye_center_3d: np.ndarray,
        eyeball_radius_mm: float,
        pose: FacePose,
    ) -> IrisEstimate:
        if ring_px is None or len(ring_px) == 0:
            return IrisEstimate(False, None, None, 0.0)

        center_px, validity = self._robust_center(ring_px)
        if validity < self._cfg.min_iris_validity:
            return IrisEstimate(False, center_px, None, validity)

        direction = self._geometry.pixel_to_camera_ray(pose, center_px)
        # Camera at origin: ray = t * direction, solve for sphere
        # centered at eye_center_3d with given radius:
        #   |t*d - C|^2 = r^2  =>  t^2 - 2t(d.C) + (|C|^2 - r^2) = 0
        c = eye_center_3d
        b = -2.0 * np.dot(direction, c)
        cc = float(np.dot(c, c) - eyeball_radius_mm ** 2)
        discriminant = b * b - 4.0 * cc
        if discriminant < 0:
            # Ray doesn't hit the modeled eyeball at all -- landmark
            # noise or a bad eye-center estimate this frame.
            return IrisEstimate(False, center_px, None, validity)

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0
        t = min(t1, t2) if min(t1, t2) > 0 else max(t1, t2)
        if t <= 0:
            return IrisEstimate(False, center_px, None, validity)

        point_3d = direction * t
        return IrisEstimate(True, center_px, point_3d, validity)
