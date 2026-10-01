"""
gaze/eye_model.py

Estimates each 3D eyeball center in CAMERA coordinates.

Critical design point (see spec section 6): the eye center is NOT the
average of eyelid landmarks. It is defined as a FIXED offset inside
the rigid face-model frame (medial shift toward the nose, backward
into the socket, slight downward -- approximate anthropometry, see
config.EyeGeometryConfig), then transformed into camera coordinates
via the current frame's (R, t) from FaceGeometryEstimator.

Because the offset is fixed in the face's own frame, eye ROTATION
never moves this point -- only head pose (R, t) does, which is the
entire point of decoupling eye-center estimation from eyelid/iris
landmark noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import EyeGeometryConfig, FACE_MODEL_3D
from face.face_geometry import FacePose


@dataclass
class EyeCenters:
    left_eye_center_3d: np.ndarray   # camera coords
    right_eye_center_3d: np.ndarray  # camera coords


class EyeModel:
    def __init__(self, cfg: EyeGeometryConfig):
        self._cfg = cfg
        left_outer = np.array(FACE_MODEL_3D["left_eye_outer_corner"])
        right_outer = np.array(FACE_MODEL_3D["right_eye_outer_corner"])

        # Medial direction: left eye's nose-ward direction is +X
        # (toward center, since left_outer.x < 0); right eye's is -X.
        self._left_center_model = left_outer + np.array(
            [cfg.medial_shift_mm, -cfg.downward_shift_mm, -cfg.backward_shift_mm]
        )
        self._right_center_model = right_outer + np.array(
            [-cfg.medial_shift_mm, -cfg.downward_shift_mm, -cfg.backward_shift_mm]
        )

    def estimate(self, pose: FacePose) -> EyeCenters:
        left = pose.face_to_camera(self._left_center_model)
        right = pose.face_to_camera(self._right_center_model)
        return EyeCenters(left_eye_center_3d=left, right_eye_center_3d=right)

    @property
    def eyeball_radius_mm(self) -> float:
        return self._cfg.eyeball_radius_mm
