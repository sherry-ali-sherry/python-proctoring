"""
gaze/eye_direction.py

Eye-in-head direction from MediaPipe's face blendshapes, and the
"eyes turned away from the screen" rule built on it.

FaceLandmarker's blendshape model outputs, per eye, how far the eye is
turned in / out / up / down (ARKit-style names, scores in [0, 1], ~0 when
the eye looks straight ahead). Unlike the geometric iris rays in
iris_tracker.py / gaze_ray.py, these scores need no screen calibration and
track sustained sideways and vertical gaze well.

Why the rule is RELATIVE to each candidate (v1.2)
-------------------------------------------------
A laptop or monitor webcam sits ABOVE the screen. Reading the screen
therefore means the eyes are rotated DOWN, not "straight ahead": on real
recordings the downward score while reading the screen ranged from ~0.2
to ~0.45 depending on the person and seating. "Eyes straight ahead" is
looking at the camera or ABOVE the screen.

Earlier versions compared the scores with fixed limits and located the
screen from "eyes centred" frames. On the 'sir' recording that picked the
8 s the candidate spent looking above the screen as "the screen", and a
fixed up-limit of 0.5 never fired (his up score only reached ~0.25).

Now:
  1. fit_eye_baseline() finds this candidate's screen-reading eye position.
     It finds the common eye positions (clusters) among roughly-ahead
     frames. Normally the largest one is the screen, but when another
     substantial cluster sits clearly BELOW it at about the same sideways
     position, that lower one is the screen: the screen is below the
     webcam, so the higher position is looking above the screen. (Absolute values differ a lot between
     people and setups - screen reading measured eye_v from -0.33 to
     +0.08 on the test recordings - so no fixed "screen height" works.)
  2. eyes_away_direction() measures each frame's deviation from that
     baseline with separate side / up / down limits (the screen's top edge
     is close to the camera, so "up" gets the smallest limit), tests it
     with an ELLIPSE so diagonal looks toward a corner are caught, and
     labels it with one of eight directions (gaze/directions.py).

Naming: blendshape "Left"/"Right" are the SUBJECT's eyes. With an
unmirrored webcam, the subject looking to their own left appears as
looking to the IMAGE right. Directions are expressed as seen in the video.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from gaze.directions import direction_label, outside_ellipse

_REQUIRED = ("eyeLookInLeft", "eyeLookOutLeft", "eyeLookInRight", "eyeLookOutRight",
             "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
             "eyeBlinkLeft", "eyeBlinkRight")


@dataclass(frozen=True)
class EyeDirection:
    x: float        # > 0: eyes turned toward the IMAGE right; < 0: image left (about -1..1)
    up: float       # 0..1
    down: float     # 0..1
    blink: float    # 0..1 (eyes closing; eye-direction scores are unreliable then)

    @property
    def v(self) -> float:
        """Vertical eye direction, -1 (fully down) .. +1 (fully up)."""
        return self.up - self.down


def eye_direction_from_blendshapes(bs: Dict[str, float]) -> Optional[EyeDirection]:
    if not bs or any(k not in bs for k in _REQUIRED):
        return None
    # Subject's left eye turning OUT and right eye turning IN both mean the
    # subject looks to their own left = the image right.
    image_right = (bs["eyeLookOutLeft"] + bs["eyeLookInRight"]) / 2.0
    image_left = (bs["eyeLookInLeft"] + bs["eyeLookOutRight"]) / 2.0
    return EyeDirection(
        x=image_right - image_left,
        up=(bs["eyeLookUpLeft"] + bs["eyeLookUpRight"]) / 2.0,
        down=(bs["eyeLookDownLeft"] + bs["eyeLookDownRight"]) / 2.0,
        blink=(bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2.0,
    )


# --------------------------------------------------------------------------- #
# Baseline: where this candidate's eyes point while reading the screen
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EyeCluster:
    x: float
    v: float
    frames: int


@dataclass(frozen=True)
class EyeBaseline:
    x: float                 # baseline eye_x
    v: float                 # baseline eye_v (= up - down)
    frames_used: int         # frames in the chosen cluster
    prior_fraction: float    # share of open-eye frames inside the roughly-ahead prior
    fallback: bool           # True: too few plausible frames, fitted from all frames
    message: str
    clusters: tuple = ()     # every substantial cluster considered (EyeCluster)

    def to_dict(self) -> dict:
        return {"x": round(self.x, 4), "v": round(self.v, 4), "frames_used": self.frames_used,
                "prior_fraction": round(self.prior_fraction, 4), "fallback": self.fallback,
                "message": self.message,
                "clusters": [{"x": round(c.x, 3), "v": round(c.v, 3), "frames": c.frames} for c in self.clusters]}


def _densest_point(points: np.ndarray, bandwidth: float) -> np.ndarray:
    """Mode of a 2D point cloud: peak of a Gaussian-smoothed histogram,
    refined as the median of the points near that peak. numpy only."""
    step = bandwidth / 3.0
    lo, hi = -1.2, 1.2
    edges = np.arange(lo, hi + step, step)
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[edges, edges])
    half = int(np.ceil(3 * bandwidth / step))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * step / bandwidth) ** 2)
    k /= k.sum()
    smooth = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 0, hist)
    smooth = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, smooth)
    i, j = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    peak = np.array([(edges[i] + edges[i + 1]) / 2, (edges[j] + edges[j + 1]) / 2])
    near = points[np.linalg.norm(points - peak, axis=1) <= 2 * bandwidth]
    return np.median(near, axis=0) if len(near) else peak


def find_eye_clusters(points: np.ndarray, bandwidth: float, radius: float, min_frames: int,
                      max_clusters: int = 5) -> List[EyeCluster]:
    """Repeatedly take the densest point and remove everything within
    `radius` of it. Returns clusters with at least `min_frames` frames,
    largest first."""
    remaining = points
    found: List[EyeCluster] = []
    for _ in range(max_clusters):
        if len(remaining) < max(min_frames, 1):
            break
        center = _densest_point(remaining, bandwidth)
        near = np.linalg.norm(remaining - center, axis=1) <= radius
        if near.sum() >= min_frames:
            found.append(EyeCluster(float(center[0]), float(center[1]), int(near.sum())))
        remaining = remaining[~near]
    return sorted(found, key=lambda c: -c.frames)


def fit_eye_baseline(eyes: Sequence[Optional[EyeDirection]], ac, ic) -> Optional[EyeBaseline]:
    """ac: AutoCalibrationConfig, ic: IncidentConfig. None without eye data."""
    pts = np.array([(e.x, e.v) for e in eyes if e is not None and e.blink < ic.eye_blink_ignore], dtype=float)
    if len(pts) == 0:
        return None
    prior = ((np.abs(pts[:, 0]) <= ac.screen_prior_max_side)
             & (pts[:, 1] <= ac.screen_prior_max_up)
             & (pts[:, 1] >= -ac.screen_prior_max_down))
    fraction = float(prior.mean())
    radius = ac.cluster_radius
    if prior.sum() >= ac.min_samples and fraction >= ac.min_screen_prior_fraction:
        clusters = find_eye_clusters(pts[prior], ac.baseline_bandwidth, radius, ac.min_samples)
        if clusters:
            big = [c for c in clusters if c.frames >= ac.cluster_min_share * clusters[0].frames]
            screen = clusters[0]
            # The screen is below the webcam: a substantial cluster clearly
            # below the largest one, at about the same sideways position,
            # is the screen and the largest one is looking above it.
            below = [c for c in big if c.v < screen.v - radius
                     and abs(c.x - screen.x) <= ac.cluster_max_side_offset]
            if below:
                screen = min(below, key=lambda c: c.v)
            above = [c for c in big if c is not screen and c.v > screen.v + radius
                     and abs(c.x - screen.x) <= ac.cluster_max_side_offset]
            note = (f" A second common eye position {above[0].v - screen.v:+.2f} higher "
                    f"({above[0].frames} frames) was treated as looking above the screen." if above else "")
            return EyeBaseline(screen.x, screen.v, screen.frames, fraction, False,
                               f"Screen-reading eye position found from {screen.frames} frames "
                               f"(of {int(prior.sum())} roughly-ahead frames)."
                               + note, tuple(clusters))
    x, v = _densest_point(pts, ac.baseline_bandwidth)
    return EyeBaseline(float(x), float(v), len(pts), fraction, True,
                       f"Only {fraction:.0%} of open-eye frames look roughly ahead, so the candidate's "
                       f"usual eye position was used as the screen. The camera may be off to the side, or "
                       f"the candidate rarely looked at the screen; review the clips.")


# --------------------------------------------------------------------------- #
# Per-frame "eyes away" decision
# --------------------------------------------------------------------------- #
def eye_deviation(ed: EyeDirection, baseline: EyeBaseline, ic) -> tuple:
    """(rx, ry) deviation from the baseline, each divided by its limit."""
    dx = ed.x - baseline.x
    dv = ed.v - baseline.v
    ry = dv / (ic.eye_limit_up if dv > 0 else ic.eye_limit_down)
    return dx / ic.eye_limit_side, ry


def eyes_away_direction(ed: Optional[EyeDirection], baseline: Optional[EyeBaseline], ic) -> Optional[str]:
    """One of the eight directions when the eyes are turned away from the
    candidate's screen position this frame, else None (also None while
    blinking or without eye data / baseline). ic: IncidentConfig."""
    if ed is None or baseline is None or ed.blink >= ic.eye_blink_ignore:
        return None
    rx, ry = eye_deviation(ed, baseline, ic)
    if not outside_ellipse(rx, ry):
        return None
    return direction_label(rx, ry, ic.eye_diagonal_min_ratio)


def near_baseline(ed: Optional[EyeDirection], baseline: Optional[EyeBaseline], ic, fraction: float) -> bool:
    """True when the eyes are well inside the on-screen ellipse (within
    `fraction` of it). Used to pick the frames the 3D screen position is
    fitted from."""
    if ed is None or baseline is None or ed.blink >= ic.eye_blink_ignore:
        return False
    rx, ry = eye_deviation(ed, baseline, ic)
    return float(np.hypot(rx, ry)) <= fraction


def _drop_flicker(times: Sequence[float], labels: List[Optional[str]], min_run_s: float) -> List[Optional[str]]:
    """Remove "away" runs shorter than min_run_s. Landmark noise near a
    limit makes single frames flicker across it; without this, the
    glance-tolerant merging joins those flickers to a real look-away and
    moves the incident's start seconds too early (seen on 'sherry30')."""
    if min_run_s <= 0:
        return labels
    out = list(labels)
    n, i = len(labels), 0
    while i < n:
        if labels[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < n and labels[j + 1] is not None:
            j += 1
        span = (times[j] - times[i]) + (times[1] - times[0] if n > 1 else 0.0)
        if span < min_run_s:
            for k in range(i, j + 1):
                out[k] = None
        i = j + 1
    return out


def label_eye_frames(times: Sequence[float], eyes: Sequence[Optional[EyeDirection]],
                     baseline: Optional[EyeBaseline], ic) -> List[Optional[str]]:
    """eyes_away label for every frame, including the closed-eyes rule:
    a blink (< eye_closed_as_away_s) is ignored, but lids held nearly
    closed for longer count as DOWN. Looking far down lowers the eyelids
    so much that MediaPipe reports a 'blink'; on the 'sir' recording the
    8 s of looking at the desk read as blink ~0.75 the whole time."""
    labels = [eyes_away_direction(e, baseline, ic) for e in eyes]
    if baseline is None:
        return labels
    labels = _drop_flicker(times, labels, ic.eye_min_run_s)
    if ic.eye_closed_as_away_s <= 0:
        return labels
    closed = [e is not None and e.blink >= ic.eye_blink_ignore for e in eyes]
    n, i = len(eyes), 0
    while i < n:
        if not closed[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and closed[j + 1]:
            j += 1
        if times[j] - times[i] >= ic.eye_closed_as_away_s:
            for k in range(i, j + 1):
                labels[k] = "DOWN"
        i = j + 1
    return labels
