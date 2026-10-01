"""
gaze/directions.py

Eight-way direction labels shared by every "not looking" signal (3D gaze,
eye direction, head turn), so the report, clips and web timeline all use
the same vocabulary:

    UP, DOWN, LEFT, RIGHT, UP-LEFT, UP-RIGHT, DOWN-LEFT, DOWN-RIGHT

Directions are as seen in the video (camera view): the candidate's own left
appears on the image right.

A deviation is described by two components already divided by their own
limits (so |component| > 1 means "beyond that limit"):
    rx > 0: toward the image right      ry > 0: up

It is labelled diagonal when the smaller component is at least
`diagonal_min_ratio` of the larger one. With 0.6 that is a direction more
than ~31 deg away from the nearest axis; pure sideways glances, which make
MediaPipe's "up" score twitch a little, stay LEFT / RIGHT.
"""
from __future__ import annotations

import math
from typing import Optional, Set

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "UP-LEFT", "UP-RIGHT", "DOWN-LEFT", "DOWN-RIGHT")


def direction_label(rx: float, ry: float, diagonal_min_ratio: float) -> Optional[str]:
    ax, ay = abs(rx), abs(ry)
    if ax == 0.0 and ay == 0.0:
        return None
    horizontal = "RIGHT" if rx > 0 else "LEFT"
    vertical = "UP" if ry > 0 else "DOWN"
    major, minor = max(ax, ay), min(ax, ay)
    if minor >= diagonal_min_ratio * major:
        return f"{vertical}-{horizontal}"
    return horizontal if ax >= ay else vertical


def outside_ellipse(rx: float, ry: float) -> bool:
    """True when the normalized deviation lies outside the unit ellipse.
    Unlike checking each axis separately (a box), this also catches
    diagonal looks that are moderately off on BOTH axes, e.g. 75% of the
    side limit and 75% of the down limit (radius 1.06)."""
    return math.hypot(rx, ry) > 1.0


def families(label: Optional[str]) -> Set[str]:
    """'DOWN-LEFT' -> {'DOWN', 'LEFT'}; None -> set()."""
    return set(label.split("-")) if label else set()
