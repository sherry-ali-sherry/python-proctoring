"""
gaze/eye_direction.py

Eye-in-head direction from MediaPipe's face blendshapes.

FaceLandmarker's blendshape model outputs, per eye, how far the eye is
turned in / out / up / down (ARKit-style names, scores in [0, 1], ~0 when
the eye looks straight ahead). Unlike the geometric iris rays in
iris_tracker.py / gaze_ray.py, these scores:
  * need no calibration (they are relative to the head, not the screen);
  * track sustained sideways and downward gaze such as reading notes next
    to or below the screen. Measured on real recordings, the geometric
    rays barely moved (~1.5 deg) between "looking at the screen" and
    "reading notes below it", while these scores separated the two
    clearly (~0.1 on screen vs ~0.5-0.9 reading off-screen).

Naming: blendshape "Left"/"Right" are the SUBJECT's eyes. With an
unmirrored webcam, the subject looking to their own left appears as
looking to the IMAGE right. Directions below are expressed as seen in
the video (image left/right), matching the rest of the report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

_REQUIRED = ("eyeLookInLeft", "eyeLookOutLeft", "eyeLookInRight", "eyeLookOutRight",
             "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
             "eyeBlinkLeft", "eyeBlinkRight")


@dataclass(frozen=True)
class EyeDirection:
    x: float        # > 0: eyes turned toward the IMAGE right; < 0: image left (about -1..1)
    up: float       # 0..1
    down: float     # 0..1
    blink: float    # 0..1 (eyes closing; eye-direction scores are unreliable then)


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


def eyes_away_direction(ed: Optional[EyeDirection], side_threshold: float, down_threshold: float,
                        up_threshold: float, blink_ignore: float) -> Optional[str]:
    """'LEFT' / 'RIGHT' / 'DOWN' / 'UP' when the eyes are turned away from
    the screen this frame, else None (also None while blinking or when
    no eye data exists)."""
    if ed is None or ed.blink >= blink_ignore:
        return None
    side = abs(ed.x) / side_threshold
    down = ed.down / down_threshold
    up = ed.up / up_threshold
    strongest = max(side, down, up)
    if strongest <= 1.0:
        return None
    if strongest == side:
        return "RIGHT" if ed.x > 0 else "LEFT"
    return "DOWN" if strongest == down else "UP"


def eyes_centered(ed: Optional[EyeDirection], fraction: float, side_threshold: float,
                  down_threshold: float, up_threshold: float, blink_ignore: float) -> bool:
    """True when the eyes are clearly near straight-ahead (within
    `fraction` of each away-threshold). Used to pick the frames the
    screen position is calibrated from."""
    if ed is None or ed.blink >= blink_ignore:
        return False
    return (abs(ed.x) <= side_threshold * fraction and ed.down <= down_threshold * fraction
            and ed.up <= up_threshold * fraction)
