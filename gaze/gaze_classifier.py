"""
gaze/gaze_classifier.py

Per-frame fusion layer. Combines:
    - screen intersection (from the combined gaze ray + calibrated
      plane + affine correction)
    - gaze/iris/face confidence
    - head pose (SUPPORTING signal only)
into a single-frame raw label + confidence. Temporal smoothing and
the hysteresis state machine live in proctoring/, not here -- this
module only judges a single frame.

Fusion logic (spec section 17): eye gaze dominates. Head pose can
only modestly raise or lower confidence in the eye-based verdict; it
never flips a clear eye-gaze verdict on its own, and large,
unexplained head rotation with poor eye evidence pushes toward
UNCERTAIN rather than a confident guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from config import AppConfig
from face.head_pose import HeadPoseAngles
from gaze.gaze_ray import GazeRay
from gaze.screen_plane import ScreenPlane, Intersection, intersect_ray_plane
from gaze.calibration import CalibrationManager
from gaze.directions import direction_label


class RawGazeLabel(str, Enum):
    SCREEN = "LOOKING_AT_SCREEN"
    AWAY = "LOOKING_AWAY"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class GazeFrameResult:
    label: RawGazeLabel
    confidence: float
    screen_x: Optional[float]   # normalized [-1, 1], None if invalid
    screen_y: Optional[float]
    intersection_valid: bool
    away_direction: Optional[str]  # one of gaze.directions.DIRECTIONS, or None


class GazeClassifier:
    def __init__(self, cfg: AppConfig, calibration: CalibrationManager):
        self._cfg = cfg
        self._calibration = calibration

    def classify(
        self,
        combined_ray: Optional[GazeRay],
        head_pose: Optional[HeadPoseAngles],
        face_confidence: float,
        pnp_reprojection_error_px: float,
    ) -> GazeFrameResult:
        conf_cfg = self._cfg.confidence

        base_confidence = face_confidence
        if pnp_reprojection_error_px > conf_cfg.max_pnp_reprojection_error_px:
            base_confidence *= 0.4

        if combined_ray is None:
            return GazeFrameResult(RawGazeLabel.UNCERTAIN, 0.0, None, None, False, None)

        plane = self._calibration.screen_plane()
        intersection = intersect_ray_plane(combined_ray, plane)

        gaze_confidence = base_confidence * combined_ray.confidence

        if not intersection.valid:
            return GazeFrameResult(RawGazeLabel.UNCERTAIN, gaze_confidence * 0.3, None, None, False, None)

        sx, sy = self._calibration.apply_affine(intersection.plane_u, intersection.plane_v)

        if gaze_confidence < conf_cfg.min_gaze_confidence:
            return GazeFrameResult(RawGazeLabel.UNCERTAIN, gaze_confidence, sx, sy, True, None)

        margin = self._cfg.screen.margin_fraction
        inside = (-1.0 - margin) <= sx <= (1.0 + margin) and (-1.0 - margin) <= sy <= (1.0 + margin)

        # Head pose: supporting evidence only. It nudges confidence,
        # weighted by head_pose_influence_weight, and can only push
        # a BORDERLINE eye verdict toward UNCERTAIN when head pose
        # disagrees strongly with reliable eye geometry -- never
        # override a clear eye-based verdict by itself.
        head_penalty = 0.0
        if head_pose is not None:
            hp_cfg = self._cfg.head_pose
            extreme = (
                abs(head_pose.yaw_deg) > hp_cfg.max_reliable_yaw_deg
                or abs(head_pose.pitch_deg) > hp_cfg.max_reliable_pitch_deg
            )
            if extreme:
                head_penalty = hp_cfg.head_pose_influence_weight

        final_confidence = float(np.clip(gaze_confidence * (1.0 - head_penalty), 0.0, 1.0))

        # Eight-way direction from BOTH axes (the old if/elif reported a
        # corner as LEFT/RIGHT only). Each axis contributes only the part
        # beyond the screen edge, so a point far right and just below the
        # screen stays RIGHT, while one clearly past a corner is diagonal.
        direction = None
        if not inside:
            over_x = float(np.sign(sx) * max(abs(sx) - 1.0, 0.0))
            over_y = float(np.sign(sy) * max(abs(sy) - 1.0, 0.0))
            direction = direction_label(over_x, over_y, self._cfg.video.incidents.eye_diagonal_min_ratio)

        if final_confidence < conf_cfg.min_gaze_confidence:
            label = RawGazeLabel.UNCERTAIN
        else:
            label = RawGazeLabel.SCREEN if inside else RawGazeLabel.AWAY

        return GazeFrameResult(label, final_confidence, sx, sy, True, direction)
