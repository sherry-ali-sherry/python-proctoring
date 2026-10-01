"""
gaze/gaze_ray.py

Explicit 3D gaze ray representation: R(t) = O + t*D, in camera
coordinates. Constructed strictly from 3D eye-center and 3D iris-center
estimates -- never from raw 2D pixel displacement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GazeRay:
    origin: np.ndarray       # O, camera coords
    direction: np.ndarray    # D, normalized, camera coords
    confidence: float        # inherited from the iris/eye estimate quality

    def point_at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction


def build_gaze_ray(
    eye_center_3d: np.ndarray,
    iris_center_3d: np.ndarray,
    confidence: float,
) -> Optional[GazeRay]:
    vec = iris_center_3d - eye_center_3d
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return None
    direction = vec / norm
    return GazeRay(origin=eye_center_3d, direction=direction, confidence=confidence)
