"""
gaze/binocular_gaze.py

Combines the left and right 3D gaze rays into a single representative
"combined gaze ray" using actual 3D ray geometry -- specifically the
closest-approach point between the two (generally skew) lines -- NOT
a 2D pixel average.

Closest approach between two lines:
    L1(t) = O1 + t*D1
    L2(s) = O2 + s*D2

Let w0 = O1 - O2. Solve the standard 2-line least-squares system:
    a = D1.D1 (=1, normalized)      b = D1.D2
    c = D2.D2 (=1, normalized)      d = D1.w0
    e = D2.w0
    denom = a*c - b*b
    t = (b*e - c*d) / denom
    s = (a*e - b*d) / denom
(When the rays are (near) parallel, denom ~ 0; fall back to the
midpoint of the two origins.)

The combined ray's origin is the midpoint of the two eye centers (the
"cyclopean eye"); its direction is the confidence-weighted average of
the two individual (normalized) directions, re-normalized. The
closest-approach separation is used as the eyes' disagreement penalty.

Why not originate the combined ray at the closest-approach midpoint (as
an earlier version did): that point is the estimated FIXATION point. For
the near-parallel rays of someone looking at a screen, small iris noise
slides it hundreds of units back and forth along the gaze line, and it
can land on the far side of the screen plane, where ray/plane
intersection rejects it as "screen behind the ray" (t < 0). A ray must
start at the eyes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gaze.gaze_ray import GazeRay


@dataclass
class ClosestApproachResult:
    point_on_left: np.ndarray
    point_on_right: np.ndarray
    midpoint: np.ndarray
    separation_distance: float   # how far apart the two closest points are
                                  # (a natural disagreement / noise metric)


def closest_approach(left: GazeRay, right: GazeRay) -> ClosestApproachResult:
    d1, d2 = left.direction, right.direction
    o1, o2 = left.origin, right.origin
    w0 = o1 - o2

    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    denom = a * c - b * b

    if abs(denom) < 1e-9:
        # Rays effectively parallel -- fall back to origin midpoint.
        midpoint = (o1 + o2) / 2.0
        return ClosestApproachResult(o1, o2, midpoint, float(np.linalg.norm(o1 - o2)))

    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom

    p1 = o1 + t * d1
    p2 = o2 + s * d2
    midpoint = (p1 + p2) / 2.0
    separation = float(np.linalg.norm(p1 - p2))
    return ClosestApproachResult(p1, p2, midpoint, separation)


def combine_gaze_rays(
    left: Optional[GazeRay], right: Optional[GazeRay], max_disagreement_deg: float = 10.0,
) -> Optional[GazeRay]:
    """Produce the binocular combined gaze ray. Handles the
    monocular-only case (one eye's iris estimate was invalid this
    frame) by simply returning the valid eye's ray with reduced
    confidence, since a single reliable eye still carries real
    directional information.

    `max_disagreement_deg`: the eyes' disagreement is the closest-approach
    separation seen from the camera, as an angle (separation / viewing
    distance), so it does not depend on the face model's unit scale or on
    how far the subject sits. At this angle confidence is halved (the
    maximum penalty). Normal horizontal vergence does not count: converging
    rays intersect, so their separation is ~0; only skew (mostly vertical)
    disagreement, which is measurement noise, is penalized."""
    if left is None and right is None:
        return None
    if left is None:
        return GazeRay(right.origin, right.direction, right.confidence * 0.7)
    if right is None:
        return GazeRay(left.origin, left.direction, left.confidence * 0.7)

    approach = closest_approach(left, right)

    weight_l = max(left.confidence, 1e-6)
    weight_r = max(right.confidence, 1e-6)
    combined_dir = weight_l * left.direction + weight_r * right.direction
    norm = np.linalg.norm(combined_dir)
    if norm < 1e-9:
        return None
    combined_dir /= norm

    # Disagreement between the two eyes' closest-approach points
    # degrades confidence -- large separation usually means noisy
    # iris estimates rather than a real anatomical effect.
    origin = (left.origin + right.origin) / 2.0
    viewing_distance = max(float(np.linalg.norm(origin)), 1e-6)
    disagreement_deg = float(np.degrees(np.arctan(approach.separation_distance / viewing_distance)))
    disagreement_penalty = float(np.clip(disagreement_deg / max_disagreement_deg, 0.0, 1.0))
    combined_confidence = (0.5 * (left.confidence + right.confidence)) * (1.0 - 0.5 * disagreement_penalty)

    return GazeRay(
        origin=origin,
        direction=combined_dir,
        confidence=float(np.clip(combined_confidence, 0.0, 1.0)),
    )
