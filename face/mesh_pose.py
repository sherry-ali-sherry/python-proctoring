"""
face/mesh_pose.py

Head rotation from MediaPipe's facial transformation matrix (the 4x4
transform of the canonical face mesh into camera space, fitted to all 478
landmarks). Steadier than the 6-point solvePnP pose in face/head_pose.py,
and with signs checked on real video (nose position in the image):

    yaw   > 0 : face turned toward the IMAGE right (the subject's own left)
    pitch > 0 : head tilted down (chin toward chest)
    roll  > 0 : head tilted toward the image right

These are the same conventions HeadPoseAngles documents, so the value can
be used anywhere a HeadPoseAngles is expected.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from face.head_pose import HeadPoseAngles


def head_pose_from_transform(matrix: Optional[np.ndarray]) -> Optional[HeadPoseAngles]:
    if matrix is None:
        return None
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        return None
    r = m[:3, :3]
    scale = np.linalg.norm(r, axis=0)
    if np.any(scale < 1e-9):
        return None
    r = r / scale
    # Canonical face: x to the subject's left (image right when facing the
    # camera), y up, z out of the face toward the camera. Camera space:
    # x image-right, y up, z toward the viewer.
    forward = r @ np.array([0.0, 0.0, 1.0])
    up = r @ np.array([0.0, 1.0, 0.0])
    yaw = np.degrees(np.arctan2(forward[0], forward[2]))
    pitch = np.degrees(np.arctan2(-forward[1], np.hypot(forward[0], forward[2])))
    roll = np.degrees(np.arctan2(up[0], up[1]))
    return HeadPoseAngles(yaw_deg=float(yaw), pitch_deg=float(pitch), roll_deg=float(roll))
