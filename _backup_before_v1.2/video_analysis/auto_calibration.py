"""
video_analysis/auto_calibration.py

Self-calibration for recorded videos.

Live mode calibrates by asking the user to look at 9 known dots. A
recorded exam video has no such dots, so this module locates the screen
from the video itself, under one documented assumption: the candidate
looks at the screen for the majority of the recording (true of any
normal exam attempt).

Procedure, over the confident frames of the whole video:
  1. Intersect each combined gaze ray with the screen plane (the plane
     through the camera, parallel to the sensor, i.e. where a
     monitor/laptop-mounted webcam's screen physically is).
  2. Screen center = per-axis MEDIAN of those intersections. The median
     is unaffected by a minority of look-away frames, and absorbs the
     generic eye model's constant bias for this particular person and
     camera placement (the same role as the affine offset of the live
     9-point calibration).
  3. Screen half-size = median eye-to-plane depth * tan(half-angle),
     with the half-angles from AutoCalibrationConfig. The size is a prior
     rather than fitted from the data, because a candidate who never
     looks away would otherwise produce a tiny "screen" that flags
     ordinary noise as looking away.

The result is an ordinary CalibrationResult (diagonal affine + offset,
plane distance), so GazeClassifier uses it exactly like a live
calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from config import AppConfig
from gaze.calibration import CalibrationResult
from gaze.eye_direction import eyes_centered
from gaze.screen_plane import make_default_screen_plane, intersect_ray_plane
from pipeline import FrameMeasurement


@dataclass
class AutoCalibrationSummary:
    success: bool
    message: str
    samples_used: int
    screen_consistency: float = 0.0   # fraction of samples inside the fitted screen
    center_u: Optional[float] = None
    center_v: Optional[float] = None
    half_width: Optional[float] = None
    half_height: Optional[float] = None
    median_depth: Optional[float] = None
    result: Optional[CalibrationResult] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "message": self.message,
            "samples_used": self.samples_used,
            "screen_consistency": self.screen_consistency,
            "center_u": self.center_u,
            "center_v": self.center_v,
            "half_width": self.half_width,
            "half_height": self.half_height,
            "median_depth": self.median_depth,
        }


def _frame_gaze_confidence(m: FrameMeasurement, cfg: AppConfig) -> float:
    """Same confidence the classifier assigns before head-pose weighting."""
    conf = m.face_confidence
    if m.reprojection_error_px > cfg.confidence.max_pnp_reprojection_error_px:
        conf *= 0.4
    return conf * m.combined_ray.confidence


def fit_auto_calibration(measurements: Iterable[FrameMeasurement], cfg: AppConfig) -> AutoCalibrationSummary:
    ac = cfg.video.auto_calibration
    hp = cfg.head_pose
    ic = cfg.video.incidents
    plane = make_default_screen_plane(ac.plane_distance_mm)
    measurements = list(measurements)

    # Anchor the screen to frames where the eyes are demonstrably centred
    # (blendshape eye direction), not to whatever the candidate looked at
    # most. Without eye data (older models), every confident frame is used.
    have_eye_data = any(m.eye is not None for m in measurements)
    use_centered_only = have_eye_data and ic.enable_eye_direction_rule

    us, vs, depths = [], [], []
    for m in measurements:
        if not m.pose_ok or m.combined_ray is None:
            continue
        if use_centered_only and not eyes_centered(m.eye, ac.centered_eye_fraction, ic.eye_side_threshold,
                                                  ic.eye_down_threshold, ic.eye_up_threshold, ic.eye_blink_ignore):
            continue
        if _frame_gaze_confidence(m, cfg) < cfg.confidence.min_gaze_confidence:
            continue
        if m.head_pose is not None and (
            abs(m.head_pose.yaw_deg) > hp.max_reliable_yaw_deg
            or abs(m.head_pose.pitch_deg) > hp.max_reliable_pitch_deg
        ):
            continue
        hit = intersect_ray_plane(m.combined_ray, plane)
        depth = float(m.combined_ray.origin[2]) - ac.plane_distance_mm
        if not hit.valid or depth <= 0:
            continue
        us.append(hit.plane_u)
        vs.append(hit.plane_v)
        depths.append(depth)

    n = len(us)
    if n < ac.min_samples:
        why = (f"Only {n} frames had the candidate's eyes centred and measurable"
               if use_centered_only else f"Only {n} frames had a reliable gaze measurement")
        return AutoCalibrationSummary(
            success=False,
            samples_used=n,
            message=(
                f"{why} (at least {ac.min_samples} needed), so the screen position could not be "
                f"located and the 3D gaze check is disabled for this video. The eye-direction, "
                f"face-visibility and head-turn checks still apply."
            ),
        )

    u = np.asarray(us)
    v = np.asarray(vs)
    center_u = float(np.median(u))
    center_v = float(np.median(v))
    median_depth = float(np.median(depths))
    half_w = median_depth * float(np.tan(np.radians(ac.screen_half_angle_x_deg)))
    half_h = median_depth * float(np.tan(np.radians(ac.screen_half_angle_y_deg)))

    # (sx, sy) = A @ (u, v) + t  maps the fitted screen to [-1, 1]^2.
    A = np.array([[1.0 / half_w, 0.0], [0.0, 1.0 / half_h]])
    t = -A @ np.array([center_u, center_v])

    limit = 1.0 + cfg.screen.margin_fraction
    sx = (u - center_u) / half_w
    sy = (v - center_v) / half_h
    consistency = float(np.mean((np.abs(sx) <= limit) & (np.abs(sy) <= limit)))

    if consistency >= ac.min_screen_consistency:
        quality = "AUTO"
        message = (f"Screen located from {n} confident frames; "
                   f"{consistency:.0%} of them fall on the fitted screen.")
    else:
        quality = "AUTO_LOW_CONFIDENCE"
        message = (f"Screen located from {n} confident frames, but only {consistency:.0%} of them "
                   f"fall on the fitted screen. Gaze was unusually scattered, so the "
                   f"'mostly looking at the screen' assumption may not hold; review the clips.")

    result = CalibrationResult(
        success=True,
        quality=quality,
        quality_score=consistency,
        message=message,
        affine_matrix=A,
        affine_offset=t,
        plane_distance_mm=ac.plane_distance_mm,
    )
    return AutoCalibrationSummary(
        success=True, message=message, samples_used=n, screen_consistency=consistency,
        center_u=center_u, center_v=center_v, half_width=half_w, half_height=half_h,
        median_depth=median_depth, result=result,
    )
