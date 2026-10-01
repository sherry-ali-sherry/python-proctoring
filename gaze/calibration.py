"""
gaze/calibration.py

CalibrationManager runs a multipoint calibration sequence, collects
per-point gaze-ray samples, estimates the best-fit screen plane
distance, fits an affine correction from raw plane-intersection
coordinates to normalized screen coordinates, and computes a
calibration-quality metric.

Why an affine correction on top of the assumed plane (see
screen_plane.py for the plane assumption): fitting
    [sx, sy] ~= A @ [u, v] + t
from >= 3 calibration points (we use up to 9) via least squares lets
the calibration absorb the *real* screen's scale, offset, and small
rotation/shear relative to the simplified assumed plane -- this is
what makes the system "personalized" (per spec section 15): it
accounts for the individual's face geometry, distance from the
camera, and camera/screen placement, because all of those show up as
systematic bias in the raw (u, v) measurements that the fit corrects.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from config import CalibrationConfig, ScreenConfig
from gaze.gaze_ray import GazeRay
from gaze.screen_plane import ScreenPlane, make_default_screen_plane, intersect_ray_plane


@dataclass
class CalibrationSample:
    target_xy: Tuple[float, float]
    plane_u: float
    plane_v: float


@dataclass
class CalibrationResult:
    success: bool
    quality: str            # "GOOD" | "LOW_CONFIDENCE" | "FAILED"
    quality_score: float    # 0..1
    message: str
    affine_matrix: Optional[np.ndarray] = None   # (2,2)
    affine_offset: Optional[np.ndarray] = None   # (2,)
    plane_distance_mm: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class CalibrationManager:
    def __init__(self, cal_cfg: CalibrationConfig, screen_cfg: ScreenConfig):
        self._cal_cfg = cal_cfg
        self._screen_cfg = screen_cfg
        self._samples: List[CalibrationSample] = []
        self.result: Optional[CalibrationResult] = None
        self._plane_distance = screen_cfg.initial_plane_distance_mm

    @property
    def target_points(self):
        return self._cal_cfg.points

    def reset(self) -> None:
        self._samples.clear()
        self.result = None

    def add_sample(self, target_xy: Tuple[float, float], ray: GazeRay) -> bool:
        """Intersect the given combined gaze ray with the current
        (uncorrected) plane at self._plane_distance and record it."""
        plane = make_default_screen_plane(self._plane_distance)
        intersection = intersect_ray_plane(ray, plane)
        if not intersection.valid:
            return False
        self._samples.append(
            CalibrationSample(target_xy, intersection.plane_u, intersection.plane_v)
        )
        return True

    def _fit_for_distance(self, distance_mm: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Re-project stored samples' rays is not possible after the
        fact (we only stored plane u,v at the ORIGINAL distance), so
        instead we scale u,v analytically: for a plane parallel to the
        image plane, doubling the distance scales (u, v) linearly with
        distance for a fixed ray direction/origin. So u(d) = u(d0) *
        (d / d0) is a valid re-projection without re-intersecting.
        """
        d0 = self._plane_distance
        scale = distance_mm / d0
        uv = np.array([[s.plane_u * scale, s.plane_v * scale] for s in self._samples])
        targets = np.array([s.target_xy for s in self._samples])

        # Solve targets ~= uv @ A^T + t  (affine, least squares).
        ones = np.ones((uv.shape[0], 1))
        design = np.hstack([uv, ones])   # (N, 3)
        # Solve for each output dimension independently.
        coeffs, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
        # coeffs: (3, 2) -> rows [a_x, b_x_coupled?, offset]; build A, t
        a_row_x = coeffs[0:2, 0]
        a_row_y = coeffs[0:2, 1]
        A = np.array([a_row_x, a_row_y])       # (2,2), maps (u,v) -> (sx, sy) linear part
        t = coeffs[2, :]                        # (2,)

        predicted = uv @ A.T + t
        residuals = predicted - targets
        rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
        return A, t, rmse

    def compute(self) -> CalibrationResult:
        n_targets = len(set(self._cal_cfg.points))
        n_collected = len(self._samples)
        min_needed = max(3, int(self._cal_cfg.min_valid_sample_fraction * n_targets * self._cal_cfg.samples_per_point))

        if n_collected < min_needed:
            self.result = CalibrationResult(
                success=False,
                quality="FAILED",
                quality_score=0.0,
                message=(
                    f"Not enough valid calibration samples "
                    f"({n_collected}/{min_needed} needed). "
                    f"Improve lighting, keep your face centered and visible, "
                    f"and follow the calibration point with your eyes."
                ),
            )
            return self.result

        # Search a small range of candidate plane distances and keep
        # the one minimizing calibration RMSE (this is the "solve for
        # depth via calibration" step referenced in screen_plane.py).
        best = None
        for candidate_distance in np.linspace(300.0, 1200.0, 19):
            A, t, rmse = self._fit_for_distance(candidate_distance)
            if best is None or rmse < best[2]:
                best = (A, t, rmse, candidate_distance)

        A, t, rmse, distance_mm = best

        # Quality scoring: combine fit RMSE (normalized coords, so
        # RMSE is directly comparable to the [-1,1] target range) with
        # the fraction of attempted samples that were valid.
        expected_samples = n_targets * self._cal_cfg.samples_per_point
        completeness = n_collected / max(expected_samples, 1)
        rmse_score = float(np.clip(1.0 - rmse / self._cal_cfg.max_sample_std_normalized, 0.0, 1.0))
        quality_score = float(np.clip(0.6 * rmse_score + 0.4 * completeness, 0.0, 1.0))

        if quality_score < 0.35:
            quality = "FAILED"
            message = "Calibration quality too low. Please recalibrate with better lighting and a stable head position."
            success = False
        elif quality_score < 0.65:
            quality = "LOW_CONFIDENCE"
            message = "Calibration succeeded but with low confidence. Consider recalibrating."
            success = True
        else:
            quality = "GOOD"
            message = "Calibration succeeded."
            success = True

        self.result = CalibrationResult(
            success=success,
            quality=quality,
            quality_score=quality_score,
            message=message,
            affine_matrix=A,
            affine_offset=t,
            plane_distance_mm=distance_mm,
        )
        return self.result

    def screen_plane(self) -> ScreenPlane:
        distance = self.result.plane_distance_mm if (self.result and self.result.success) else self._screen_cfg.initial_plane_distance_mm
        return make_default_screen_plane(distance)

    def apply_affine(self, plane_u: float, plane_v: float) -> Tuple[float, float]:
        if self.result is None or not self.result.success:
            return plane_u, plane_v
        uv = np.array([plane_u, plane_v])
        sx, sy = self.result.affine_matrix @ uv + self.result.affine_offset
        return float(sx), float(sy)

    def save(self, path: str) -> None:
        if self.result is None:
            raise RuntimeError("No calibration result to save yet.")
        data = {
            "success": self.result.success,
            "quality": self.result.quality,
            "quality_score": self.result.quality_score,
            "affine_matrix": self.result.affine_matrix.tolist(),
            "affine_offset": self.result.affine_offset.tolist(),
            "plane_distance_mm": self.result.plane_distance_mm,
            "timestamp": self.result.timestamp,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> bool:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        self.result = CalibrationResult(
            success=data["success"],
            quality=data["quality"],
            quality_score=data["quality_score"],
            message="Loaded from disk.",
            affine_matrix=np.array(data["affine_matrix"]),
            affine_offset=np.array(data["affine_offset"]),
            plane_distance_mm=data["plane_distance_mm"],
            timestamp=data["timestamp"],
        )
        return True
