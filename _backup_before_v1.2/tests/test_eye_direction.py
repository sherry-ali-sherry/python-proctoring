"""
tests/test_eye_direction.py

Blendshape eye direction and the eye-direction incident rule. The
numbers mirror real recordings: ~0.1 while looking at the screen, ~0.5-0.7
while reading notes beside the screen with brief looks back.
"""
import pytest

from config import AppConfig
from gaze.eye_direction import EyeDirection, eye_direction_from_blendshapes, eyes_away_direction, eyes_centered
from proctoring.state_manager import ProctorState as S
from video_analysis.incidents import (
    REASON_EYES, VERDICT_CHEATING, FrameRecord, compute_stats, compute_verdict, extract_incidents,
)

FPS = 10.0


def _bs(out_l=0.0, in_l=0.0, out_r=0.0, in_r=0.0, up=0.0, down=0.0, blink=0.0):
    return {"eyeLookOutLeft": out_l, "eyeLookInLeft": in_l, "eyeLookOutRight": out_r, "eyeLookInRight": in_r,
            "eyeLookUpLeft": up, "eyeLookUpRight": up, "eyeLookDownLeft": down, "eyeLookDownRight": down,
            "eyeBlinkLeft": blink, "eyeBlinkRight": blink}


def _away(ed):
    ic = AppConfig().video.incidents
    return eyes_away_direction(ed, ic.eye_side_threshold, ic.eye_down_threshold, ic.eye_up_threshold, ic.eye_blink_ignore)


def test_blendshape_mapping_to_image_directions():
    # subject looks to their own RIGHT: left eye turns in, right eye turns out -> image LEFT
    ed = eye_direction_from_blendshapes(_bs(in_l=0.6, out_r=0.7))
    assert ed.x == pytest.approx(-0.65)
    assert _away(ed) == "LEFT"
    ed = eye_direction_from_blendshapes(_bs(out_l=0.9, in_r=0.9))
    assert _away(ed) == "RIGHT"
    assert _away(eye_direction_from_blendshapes(_bs(down=0.7))) == "DOWN"
    assert _away(eye_direction_from_blendshapes(_bs(up=0.8))) == "UP"


def test_on_screen_blinks_and_missing_data_are_not_away():
    assert _away(eye_direction_from_blendshapes(_bs(out_l=0.12, in_r=0.1, up=0.1, down=0.1))) is None
    assert _away(eye_direction_from_blendshapes(_bs(down=0.8, blink=0.7))) is None      # blink, not reading
    assert eye_direction_from_blendshapes({}) is None
    assert eye_direction_from_blendshapes({"eyeLookInLeft": 0.5}) is None
    assert _away(None) is None


def test_eyes_centered_for_calibration():
    ic = AppConfig().video.incidents
    args = (0.6, ic.eye_side_threshold, ic.eye_down_threshold, ic.eye_up_threshold, ic.eye_blink_ignore)
    assert eyes_centered(EyeDirection(0.1, 0.1, 0.1, 0.1), *args)
    assert not eyes_centered(EyeDirection(0.3, 0.1, 0.1, 0.1), *args)    # under the away limit but not centred
    assert not eyes_centered(EyeDirection(0.0, 0.1, 0.1, 0.9), *args)    # blinking
    assert not eyes_centered(None, *args)


def _records(spec):
    """spec: (seconds, eyes_away direction or None)."""
    out, idx = [], 0
    for seconds, away in spec:
        for _ in range(int(round(seconds * FPS))):
            out.append(FrameRecord(index=idx, time_s=idx / FPS, num_faces=1, state=S.LOOKING_AT_SCREEN,
                                   eye_x=-0.6 if away else 0.1, eyes_away=away))
            idx += 1
    return out


def test_reading_with_brief_looks_back_is_one_incident():
    """Reading pattern from a real recording: off-screen 3 s, back 1 s,
    off 5 s, back 1 s, off 6 s. Each off-screen stretch alone is short,
    but together they are one sustained period of not looking."""
    cfg = AppConfig()
    recs = _records([(4, None), (3, "LEFT"), (1, None), (5, "LEFT"), (1, None), (6, "LEFT"), (2, None)])
    incs = extract_incidents(recs, cfg, FPS)
    assert len(incs) == 1
    inc = incs[0]
    assert inc.reasons == [REASON_EYES] and inc.direction == "LEFT"
    assert inc.start_s == pytest.approx(4.0) and inc.end_s == pytest.approx(20.0)
    v = compute_verdict(compute_stats(recs, incs, FPS), cfg, True)
    assert v.label == VERDICT_CHEATING


def test_short_isolated_glances_are_not_incidents():
    cfg = AppConfig()
    recs = _records([(5, None), (2, "RIGHT"), (5, None), (2, "LEFT"), (5, None)])
    assert extract_incidents(recs, cfg, FPS) == []


def test_eye_rule_can_be_switched_off():
    cfg = AppConfig()
    cfg.video.incidents.enable_eye_direction_rule = False
    recs = _records([(2, None), (10, "DOWN"), (2, None)])
    assert extract_incidents(recs, cfg, FPS) == []
