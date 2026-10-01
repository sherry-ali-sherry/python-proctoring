"""
visualization/overlay.py

Real-time OpenCV debug overlay (spec sections 23-24). Every element
is behind a VisualizationConfig toggle. Drawing here never affects
the underlying geometry -- it only reads already-computed results.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config import VisualizationConfig
from face.face_geometry import FacePose
from face.head_pose import HeadPoseAngles
from face.face_tracker import FaceObservation
from gaze.eye_model import EyeCenters
from gaze.gaze_ray import GazeRay
from gaze.gaze_classifier import GazeFrameResult
from proctoring.state_manager import ProctorState


def _project_point(point_3d: np.ndarray, pose: FacePose) -> Optional[tuple]:
    """Project a camera-coordinate 3D point into image pixels using
    the same intrinsics used for solvePnP (identity extrinsics, since
    point_3d is already in camera coords)."""
    if point_3d[2] <= 1e-6:
        return None
    fx, fy = pose.camera_matrix[0, 0], pose.camera_matrix[1, 1]
    cx, cy = pose.camera_matrix[0, 2], pose.camera_matrix[1, 2]
    u = fx * point_3d[0] / point_3d[2] + cx
    v = fy * point_3d[1] / point_3d[2] + cy
    return int(round(u)), int(round(v))


class OverlayRenderer:
    def __init__(self, cfg: VisualizationConfig):
        self._cfg = cfg

    def draw(
        self,
        frame_bgr: np.ndarray,
        face: Optional[FaceObservation],
        pose: Optional[FacePose],
        head_pose: Optional[HeadPoseAngles],
        eye_centers: Optional[EyeCenters],
        left_ray: Optional[GazeRay],
        right_ray: Optional[GazeRay],
        combined_ray: Optional[GazeRay],
        gaze_result: Optional[GazeFrameResult],
        proctor_state: ProctorState,
        num_faces: int,
        calibration_status: str,
        fps: float,
    ) -> np.ndarray:
        img = frame_bgr

        if self._cfg.show_landmarks and face is not None:
            for i in range(0, face.landmarks_norm.shape[0], 3):
                p = face.px(i)
                cv2.circle(img, (int(p[0]), int(p[1])), 1, (80, 180, 80), -1)

        if self._cfg.show_iris and face is not None:
            from config import LEFT_IRIS_RING, RIGHT_IRIS_RING
            for idx in LEFT_IRIS_RING + RIGHT_IRIS_RING:
                p = face.px(idx)
                cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)

        if pose is not None:
            if self._cfg.show_eye_centers and eye_centers is not None:
                for c, color in (
                    (eye_centers.left_eye_center_3d, (255, 0, 255)),
                    (eye_centers.right_eye_center_3d, (255, 0, 255)),
                ):
                    p = _project_point(c, pose)
                    if p:
                        cv2.circle(img, p, 4, color, -1)

            if self._cfg.show_gaze_rays:
                for ray, color in ((left_ray, (0, 128, 255)), (right_ray, (0, 128, 255))):
                    if ray is None:
                        continue
                    p0 = _project_point(ray.origin, pose)
                    p1 = _project_point(ray.point_at(200.0), pose)
                    if p0 and p1:
                        cv2.arrowedLine(img, p0, p1, color, 2, tipLength=0.08)
                if combined_ray is not None:
                    p0 = _project_point(combined_ray.origin, pose)
                    p1 = _project_point(combined_ray.point_at(220.0), pose)
                    if p0 and p1:
                        cv2.arrowedLine(img, p0, p1, (0, 0, 255), 2, tipLength=0.08)

            if self._cfg.show_head_axes:
                axis_len = 80.0
                origin_3d = pose.translation_vector.reshape(3)
                for axis_vec, color in (
                    (np.array([axis_len, 0, 0]), (0, 0, 255)),
                    (np.array([0, axis_len, 0]), (0, 255, 0)),
                    (np.array([0, 0, axis_len]), (255, 0, 0)),
                ):
                    tip_model = axis_vec
                    tip_cam = pose.rotation_matrix @ tip_model.reshape(3, 1) + pose.translation_vector
                    p0 = _project_point(origin_3d, pose)
                    p1 = _project_point(tip_cam.reshape(3), pose)
                    if p0 and p1:
                        cv2.line(img, p0, p1, color, 2)

        if self._cfg.show_numeric_overlay:
            self._draw_text_panel(img, head_pose, gaze_result, proctor_state, num_faces, calibration_status, fps)

        return img

    def _draw_text_panel(self, img, head_pose, gaze_result: Optional[GazeFrameResult],
                          proctor_state: ProctorState, num_faces: int,
                          calibration_status: str, fps: float) -> None:
        lines = [f"STATE: {proctor_state.value}"]
        if gaze_result is not None:
            lines.append(f"GAZE CONFIDENCE: {gaze_result.confidence:.2f}")
        if head_pose is not None:
            lines.append(f"HEAD  YAW:{head_pose.yaw_deg:6.1f}  PITCH:{head_pose.pitch_deg:6.1f}  ROLL:{head_pose.roll_deg:6.1f}")
        if gaze_result is not None and gaze_result.screen_x is not None:
            lines.append(f"SCREEN  X:{gaze_result.screen_x:5.2f}  Y:{gaze_result.screen_y:5.2f}  VALID:{gaze_result.intersection_valid}")
        lines.append(f"FACES: {num_faces}   CALIBRATION: {calibration_status}")
        if self._cfg.show_fps:
            lines.append(f"FPS: {fps:.1f}")

        y = 24
        for line in lines:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22

    def draw_calibration_target(self, img: np.ndarray, norm_xy: tuple, progress: float) -> None:
        h, w = img.shape[:2]
        cx = int((norm_xy[0] + 1.0) / 2.0 * w)
        cy = int((1.0 - (norm_xy[1] + 1.0) / 2.0) * h)
        cv2.circle(img, (cx, cy), 22, (255, 255, 255), 2)
        cv2.circle(img, (cx, cy), max(2, int(18 * progress)), (0, 200, 0), -1)
        cv2.putText(img, "Look at the dot", (cx - 70, cy - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
