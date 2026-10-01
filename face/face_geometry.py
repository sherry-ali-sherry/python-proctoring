"""
face/face_geometry.py

Builds the stable 3D face coordinate system via solvePnP, using ONLY
landmarks that do not shift when the eyes rotate (nose tip, chin, eye
OUTER corners, mouth corners). This is the anchor for every later 3D
computation (eye centers, gaze rays, screen intersection all live in
the resulting camera-coordinate frame).

Coordinate systems (see README for full derivation):
    - face model coords: FACE_MODEL_3D in config.py, origin at nose tip.
    - camera coords: standard OpenCV convention, origin at the camera's
      optical center, produced by solvePnP as (R, t) that maps
      face-model points into camera coords: X_cam = R @ X_model + t.
    - image coords: 2D pixel positions, related to camera coords by
      the pinhole projection model (used only for solvePnP input and
      overlay drawing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import FACE_MODEL_3D, FACE_MODEL_LANDMARK_IDS
from face.face_tracker import FaceObservation


@dataclass
class FacePose:
    success: bool
    rotation_matrix: np.ndarray      # (3,3) face -> camera
    rotation_vector: np.ndarray      # (3,1) Rodrigues form
    translation_vector: np.ndarray   # (3,1) face-origin position in camera coords
    reprojection_error_px: float     # quality metric
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray

    def face_to_camera(self, point_model: np.ndarray) -> np.ndarray:
        """Transform a 3D point from face-model coords to camera coords."""
        point_model = np.asarray(point_model, dtype=np.float64).reshape(3, 1)
        cam = self.rotation_matrix @ point_model + self.translation_vector
        return cam.reshape(3)


class FaceGeometryEstimator:
    """Solves the rigid face pose (R, t) each frame via solvePnP."""

    def __init__(self):
        self._model_points = np.array(
            [FACE_MODEL_3D[name] for name in FACE_MODEL_LANDMARK_IDS.keys()],
            dtype=np.float64,
        )
        self._landmark_names = list(FACE_MODEL_LANDMARK_IDS.keys())
        self._landmark_ids = [FACE_MODEL_LANDMARK_IDS[n] for n in self._landmark_names]

    @staticmethod
    def build_camera_matrix(image_width: int, image_height: int) -> np.ndarray:
        # Approximate intrinsics: focal length ~ image width (typical
        # webcam FOV assumption), principal point at image center.
        # Documented limitation: not a true calibrated intrinsic
        # matrix -- adequate for pose/gaze geometry, not for metric
        # depth accuracy.
        focal_length = image_width
        cx, cy = image_width / 2.0, image_height / 2.0
        return np.array(
            [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]],
            dtype=np.float64,
        )

    def estimate(self, face: FaceObservation) -> Optional[FacePose]:
        image_points = np.array(
            [face.px(idx) for idx in self._landmark_ids], dtype=np.float64
        )
        camera_matrix = self.build_camera_matrix(face.image_width, face.image_height)
        dist_coeffs = np.zeros((4, 1))  # assume negligible lens distortion

        success, rvec, tvec = cv2.solvePnP(
            self._model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rvec)

        projected, _ = cv2.projectPoints(
            self._model_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        reprojection_error = float(
            np.mean(np.linalg.norm(projected - image_points, axis=1))
        )

        return FacePose(
            success=True,
            rotation_matrix=rotation_matrix,
            rotation_vector=rvec,
            translation_vector=tvec,
            reprojection_error_px=reprojection_error,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )

    def pixel_to_camera_ray(
        self, pose: FacePose, pixel_xy: np.ndarray
    ) -> np.ndarray:
        """Return a normalized 3D direction (camera coords) for a ray
        from the camera's optical center through the given pixel,
        using the same pinhole intrinsics used for solvePnP."""
        fx = pose.camera_matrix[0, 0]
        fy = pose.camera_matrix[1, 1]
        cx = pose.camera_matrix[0, 2]
        cy = pose.camera_matrix[1, 2]
        x = (pixel_xy[0] - cx) / fx
        y = (pixel_xy[1] - cy) / fy
        direction = np.array([x, y, 1.0])
        return direction / np.linalg.norm(direction)
