"""
face/head_pose.py

Derives yaw/pitch/roll from the rotation matrix produced by
FaceGeometryEstimator. Head pose is SUPPORTING evidence only -- see
gaze/gaze_classifier.py for how it is fused (never used as the
primary signal, and never via a bare "if yaw > X" rule).

Frame conventions: solvePnP's R maps face-model coords (X right, Y up,
Z out of the face toward the viewer) into OpenCV camera coords (X right,
Y down, Z forward away from the lens). A subject facing the camera
squarely therefore yields R = diag(1, -1, -1), a 180-degree flip about X,
not the identity. Decomposing that R directly (as an earlier version of
this module did) mislabels the axes: a left/right head turn shows up as
"pitch", an in-plane tilt as "yaw", and a nod as "roll" offset by 180
degrees. We first re-express R in a viewer frame (camera coords with Y and
Z flipped) so a frontal face is the identity, then decompose as
M = Ry(yaw) @ Rx(pitch) @ Rz(roll).

Resulting sign conventions (frontal face = 0, 0, 0):
    yaw   > 0 : face turned toward image-right (the subject's own left)
    pitch > 0 : head tilted down (chin toward chest)
    roll  > 0 : head tilted about the viewing axis
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from face.face_geometry import FacePose

# Camera coords (Y down, Z forward) -> viewer coords (Y up, Z toward camera).
_CAMERA_TO_VIEWER = np.diag([1.0, -1.0, -1.0])


@dataclass
class HeadPoseAngles:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


class HeadPoseEstimator:
    """Converts a rotation matrix (face -> camera) into Euler angles."""

    def estimate(self, pose: FacePose) -> HeadPoseAngles:
        return self.angles_from_rotation(pose.rotation_matrix)

    @staticmethod
    def angles_from_rotation(rotation_matrix: np.ndarray) -> HeadPoseAngles:
        m = _CAMERA_TO_VIEWER @ np.asarray(rotation_matrix, dtype=np.float64)

        # For M = Ry(yaw) Rx(pitch) Rz(roll):
        #   M[1,2] = -sin(pitch)
        #   M[0,2] = sin(yaw) cos(pitch),  M[2,2] = cos(yaw) cos(pitch)
        #   M[1,0] = cos(pitch) sin(roll), M[1,1] = cos(pitch) cos(roll)
        sin_pitch = float(np.clip(-m[1, 2], -1.0, 1.0))
        pitch = np.arcsin(sin_pitch)
        cos_pitch = np.sqrt(m[0, 2] ** 2 + m[2, 2] ** 2)

        if cos_pitch > 1e-6:
            yaw = np.arctan2(m[0, 2], m[2, 2])
            roll = np.arctan2(m[1, 0], m[1, 1])
        else:
            # Gimbal lock (pitch ~ +/-90 deg): yaw and roll are coupled;
            # attribute the whole remaining rotation to yaw.
            yaw = np.arctan2(-m[2, 0], m[0, 0])
            roll = 0.0

        return HeadPoseAngles(
            yaw_deg=float(np.degrees(yaw)),
            pitch_deg=float(np.degrees(pitch)),
            roll_deg=float(np.degrees(roll)),
        )
