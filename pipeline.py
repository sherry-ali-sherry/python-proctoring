"""
pipeline.py

The per-frame gaze pipeline shared by the video analyzer (main entry
point) and the optional live-webcam tool. Extracted from the original
main.py without changing its logic, and split into two stages:

    measure(frame, t)  -- pure per-frame geometry, no temporal state:
        FaceTracker -> FaceGeometryEstimator -> HeadPoseEstimator
        -> EyeModel -> IrisTracker -> gaze rays -> binocular combine

    decide(measurement, t)  -- everything that depends on calibration
        and on time:
        GazeClassifier -> TemporalFilter -> GazeStateManager -> EventEngine

Live mode calls both back to back each frame. Video mode runs measure()
over the whole file first, fits a calibration from those measurements
(a recorded video has no calibration dots), then replays decide() over
the stored measurements with the video's own timestamps. Both modes
therefore run the identical classification/hysteresis logic.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import config as config_module  # for LEFT/RIGHT iris contour landmark indices
from config import AppConfig
from face.face_tracker import FaceTracker, FaceObservation
from face.face_geometry import FaceGeometryEstimator, FacePose
from face.head_pose import HeadPoseEstimator, HeadPoseAngles
from gaze.eye_direction import EyeDirection, eye_direction_from_blendshapes
from gaze.eye_model import EyeModel, EyeCenters
from gaze.iris_tracker import IrisTracker
from gaze.gaze_ray import GazeRay, build_gaze_ray
from gaze.binocular_gaze import combine_gaze_rays
from gaze.calibration import CalibrationManager
from gaze.gaze_classifier import GazeClassifier, GazeFrameResult, RawGazeLabel
from proctoring.temporal_filter import TemporalFilter, SmoothedGaze
from proctoring.state_manager import GazeStateManager, ProctorState
from proctoring.event_engine import EventEngine


@dataclass
class FrameMeasurement:
    """Output of GazePipeline.measure(). The heavy fields (face .. right_ray)
    exist for overlay drawing; `compact()` drops them so long videos can
    keep one measurement per frame in memory."""

    timestamp: float
    num_faces: int
    pose_ok: bool = False
    face_confidence: float = 0.0
    reprojection_error_px: float = 0.0
    head_pose: Optional[HeadPoseAngles] = None
    combined_ray: Optional[GazeRay] = None
    iris_left_valid: bool = False
    iris_right_valid: bool = False
    eye: Optional[EyeDirection] = None      # blendshape eye-in-head direction
    # Heavy, overlay-only fields:
    face: Optional[FaceObservation] = None
    pose: Optional[FacePose] = None
    eye_centers: Optional[EyeCenters] = None
    left_ray: Optional[GazeRay] = None
    right_ray: Optional[GazeRay] = None

    def compact(self) -> "FrameMeasurement":
        return dataclasses.replace(
            self, face=None, pose=None, eye_centers=None, left_ray=None, right_ray=None
        )


@dataclass
class FrameDecision:
    """Output of GazePipeline.decide()."""

    state: ProctorState
    gaze_result: Optional[GazeFrameResult] = None
    smoothed: Optional[SmoothedGaze] = None
    smoothed_label: RawGazeLabel = RawGazeLabel.UNCERTAIN


class GazePipeline:
    def __init__(self, cfg: AppConfig, face_tracker: Optional[FaceTracker] = None):
        self.cfg = cfg
        self._face_tracker = face_tracker
        self.geometry = FaceGeometryEstimator()
        self.head_pose_estimator = HeadPoseEstimator()
        self.eye_model = EyeModel(cfg.eye)
        self.iris_tracker = IrisTracker(cfg.confidence, self.geometry)
        self.calibration = CalibrationManager(cfg.calibration, cfg.screen)
        self.classifier = GazeClassifier(cfg, self.calibration)
        self.temporal_filter = TemporalFilter(cfg.temporal)
        self.state_manager = GazeStateManager(cfg.temporal)
        self.event_engine = EventEngine(cfg.temporal)

    @property
    def face_tracker(self) -> FaceTracker:
        # Created lazily: decide()-only use (replaying stored measurements,
        # unit tests) never needs MediaPipe or its model download.
        if self._face_tracker is None:
            self._face_tracker = FaceTracker(self.cfg.confidence)
        return self._face_tracker

    def reset_temporal_state(self) -> None:
        """Fresh filter/state machine/events, keeping calibration."""
        self.temporal_filter = TemporalFilter(self.cfg.temporal)
        self.state_manager = GazeStateManager(self.cfg.temporal)
        self.event_engine = EventEngine(self.cfg.temporal)

    # ------------------------------------------------------------------ #
    def measure(self, frame_bgr: np.ndarray, now: float, timestamp_ms: Optional[int] = None) -> FrameMeasurement:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.face_tracker.process(frame_rgb, timestamp_ms)

        if not result.face_detected or result.num_faces == 0:
            return FrameMeasurement(timestamp=now, num_faces=0)

        # Use the highest-confidence face as the primary subject when
        # multiple faces are present; MULTIPLE_FACES_DETECTED is still
        # emitted in decide() regardless of which face we track for gaze.
        face = max(result.faces, key=lambda f: f.face_confidence)
        m = FrameMeasurement(
            timestamp=now, num_faces=result.num_faces,
            face=face, face_confidence=face.face_confidence,
            eye=eye_direction_from_blendshapes(face.blendshapes),
        )

        pose = self.geometry.estimate(face)
        if pose is None:
            return m
        m.pose = pose
        m.pose_ok = True
        m.reprojection_error_px = pose.reprojection_error_px

        m.head_pose = self.head_pose_estimator.estimate(pose)
        eye_centers = self.eye_model.estimate(pose)
        m.eye_centers = eye_centers

        left_iris = self.iris_tracker.estimate(
            face.px_many(config_module.LEFT_IRIS_CONTOUR),
            eye_centers.left_eye_center_3d,
            self.eye_model.eyeball_radius_mm,
            pose,
        )
        right_iris = self.iris_tracker.estimate(
            face.px_many(config_module.RIGHT_IRIS_CONTOUR),
            eye_centers.right_eye_center_3d,
            self.eye_model.eyeball_radius_mm,
            pose,
        )
        m.iris_left_valid = left_iris.valid
        m.iris_right_valid = right_iris.valid

        m.left_ray = (
            build_gaze_ray(eye_centers.left_eye_center_3d, left_iris.center_3d, left_iris.validity_score)
            if left_iris.valid else None
        )
        m.right_ray = (
            build_gaze_ray(eye_centers.right_eye_center_3d, right_iris.center_3d, right_iris.validity_score)
            if right_iris.valid else None
        )
        m.combined_ray = combine_gaze_rays(
            m.left_ray, m.right_ray, self.cfg.confidence.max_binocular_disagreement_deg
        )
        return m

    # ------------------------------------------------------------------ #
    def decide(self, m: FrameMeasurement, now: float) -> FrameDecision:
        self.event_engine.on_face_count(m.num_faces, now)

        if m.num_faces == 0 or not m.pose_ok:
            self.state_manager.update(RawGazeLabel.UNCERTAIN, now)
            return FrameDecision(state=self.state_manager.state)

        gaze_result = self.classifier.classify(
            m.combined_ray, m.head_pose, m.face_confidence, m.reprojection_error_px
        )
        smoothed = self.temporal_filter.update(gaze_result.screen_x, gaze_result.screen_y, gaze_result.confidence)

        # Re-derive a raw label from the SMOOTHED signal for the state
        # machine, so hysteresis operates on stable data.
        if smoothed.confidence < self.cfg.confidence.min_gaze_confidence or smoothed.screen_x is None:
            smoothed_label = RawGazeLabel.UNCERTAIN
        else:
            margin = self.cfg.screen.margin_fraction
            inside = (-1.0 - margin) <= smoothed.screen_x <= (1.0 + margin) and \
                     (-1.0 - margin) <= smoothed.screen_y <= (1.0 + margin)
            smoothed_label = RawGazeLabel.SCREEN if inside else RawGazeLabel.AWAY

        self.state_manager.update(smoothed_label, now)
        self.event_engine.on_state(self.state_manager.state, gaze_result.away_direction, smoothed.confidence, now)

        return FrameDecision(
            state=self.state_manager.state,
            gaze_result=gaze_result,
            smoothed=smoothed,
            smoothed_label=smoothed_label,
        )

    def close(self) -> None:
        if self._face_tracker is not None:
            self._face_tracker.close()
            self._face_tracker = None
