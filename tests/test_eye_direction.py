"""
tests/test_eye_direction.py

Blendshape eye direction, the per-candidate screen baseline, the eight-way
(diagonal-aware) eyes-away rule and its incidents. Numbers mirror real
recordings: reading a laptop screen puts the eyes ~0.2-0.45 "down"
(the webcam is above the screen); looking above the screen gives up ~0.25
with down ~0.02; looking at the desk gives down ~0.9 with the lids so low
that MediaPipe reports blink ~0.75.
"""
from pathlib import Path

import numpy as np
import pytest

from config import AppConfig
from gaze.directions import direction_label, families, outside_ellipse
from gaze.eye_direction import (
    EyeBaseline, EyeDirection, eye_direction_from_blendshapes, eyes_away_direction, fit_eye_baseline,
    label_eye_frames, near_baseline,
)
from proctoring.state_manager import ProctorState as S
from video_analysis.eye_replay import load_eye_log, replay_eye_rule
from video_analysis.incidents import (
    REASON_EYES, VERDICT_CHEATING, FrameRecord, compute_stats, compute_verdict, extract_incidents,
)

FPS = 10.0
DATA = Path(__file__).parent / "data"
SCREEN = EyeBaseline(x=0.1, v=-0.35, frames_used=100, prior_fraction=0.8, fallback=False, message="")


def _bs(out_l=0.0, in_l=0.0, out_r=0.0, in_r=0.0, up=0.0, down=0.0, blink=0.0):
    return {"eyeLookOutLeft": out_l, "eyeLookInLeft": in_l, "eyeLookOutRight": out_r, "eyeLookInRight": in_r,
            "eyeLookUpLeft": up, "eyeLookUpRight": up, "eyeLookDownLeft": down, "eyeLookDownRight": down,
            "eyeBlinkLeft": blink, "eyeBlinkRight": blink}


def _away(ed, baseline=SCREEN):
    return eyes_away_direction(ed, baseline, AppConfig().video.incidents)


# --------------------------------------------------------------------------- #
def test_blendshape_mapping_to_image_directions():
    # subject looks to their own RIGHT: left eye turns in, right eye turns out -> image LEFT
    ed = eye_direction_from_blendshapes(_bs(in_l=0.6, out_r=0.7, down=0.35))
    assert ed.x == pytest.approx(-0.65)
    assert ed.v == pytest.approx(-0.35)
    assert _away(ed) == "LEFT"
    assert _away(eye_direction_from_blendshapes(_bs(out_l=0.9, in_r=0.9, down=0.35))) == "RIGHT"
    assert _away(eye_direction_from_blendshapes(_bs(down=0.9))) == "DOWN"
    assert _away(eye_direction_from_blendshapes(_bs(up=0.25))) == "UP"


def test_reading_the_screen_blinks_and_missing_data_are_not_away():
    assert _away(eye_direction_from_blendshapes(_bs(out_l=0.12, in_r=0.1, down=0.4))) is None
    assert _away(eye_direction_from_blendshapes(_bs(down=0.9, blink=0.7))) is None      # single-frame blink
    assert eye_direction_from_blendshapes({}) is None
    assert eye_direction_from_blendshapes({"eyeLookInLeft": 0.5}) is None
    assert _away(None) is None
    assert _away(EyeDirection(0.9, 0.0, 0.0, 0.0), baseline=None) is None


def test_looking_above_the_screen_is_away_although_eyes_are_level():
    """The 'sir' failure: eyes level (down ~0.02, up ~0.25) used to count as
    the screen. Relative to his screen-reading position (v ~ -0.33) that is
    a +0.55 upward move, far past the 0.22 up limit."""
    assert _away(EyeDirection(0.07, 0.25, 0.02, 0.06)) == "UP"
    # Even level eyes with no "up" score are above the screen for this candidate.
    assert _away(EyeDirection(0.1, 0.0, 0.0, 0.0)) == "UP"


def test_diagonal_corner_look_is_caught_and_labelled():
    ic = AppConfig().video.incidents
    # 75% of the side limit and 75% of the down limit: each axis alone is
    # inside, but the look toward the lower-left corner is outside the ellipse.
    ed = EyeDirection(SCREEN.x - 0.75 * ic.eye_limit_side, 0.0, -SCREEN.v + 0.75 * ic.eye_limit_down, 0.0)
    assert _away(ed) == "DOWN-LEFT"
    # 80% of the side limit and 80% of the up limit -> upper-right corner.
    ed = EyeDirection(SCREEN.x + 0.8 * ic.eye_limit_side, 0.0, -(SCREEN.v + 0.8 * ic.eye_limit_up), 0.0)
    assert _away(ed) == "UP-RIGHT"
    assert outside_ellipse(0.8, 0.8) and not outside_ellipse(0.7, 0.7)
    # Well inside on both axes: not away.
    assert _away(EyeDirection(SCREEN.x + 0.4 * ic.eye_limit_side, 0.0, -SCREEN.v + 0.4 * ic.eye_limit_down, 0.0)) is None


def test_direction_labels():
    assert direction_label(2.0, 0.3, 0.6) == "RIGHT"        # pure sideways with a little up-twitch
    assert direction_label(-1.0, -0.9, 0.6) == "DOWN-LEFT"
    assert direction_label(0.7, 1.0, 0.6) == "UP-RIGHT"
    assert direction_label(0.1, -1.5, 0.6) == "DOWN"
    assert direction_label(0.0, 0.0, 0.6) is None
    assert families("DOWN-LEFT") == {"DOWN", "LEFT"}


# --------------------------------------------------------------------------- #
def _cloud(x, v, n, spread=0.03, seed=0, blink=0.1):
    rng = np.random.default_rng(seed)
    return [EyeDirection(float(a), 0.0, float(-b), blink)
            for a, b in zip(rng.normal(x, spread, n), rng.normal(v, spread, n))]


def test_baseline_ignores_looking_above_even_when_it_is_the_largest_cluster():
    ac, ic = AppConfig().video.auto_calibration, AppConfig().video.incidents
    eyes = (_cloud(0.1, -0.35, 300, seed=1)          # reading the screen
            + [EyeDirection(0.05, 0.25, 0.02, 0.05)] * 400   # above the screen, v = +0.23
            + _cloud(0.1, -0.9, 200, seed=2))        # desk, outside the prior
    b = fit_eye_baseline(eyes, ac, ic)
    assert not b.fallback
    assert b.x == pytest.approx(0.1, abs=0.03) and b.v == pytest.approx(-0.35, abs=0.03)


def test_baseline_prefers_the_largest_cluster_over_one_below_and_to_the_side():
    """Notes on the desk below-left are not 'the screen': only a cluster
    directly below the largest one replaces it."""
    ac, ic = AppConfig().video.auto_calibration, AppConfig().video.incidents
    eyes = _cloud(0.1, -0.30, 500, seed=3) + _cloud(-0.2, -0.55, 200, seed=4)
    b = fit_eye_baseline(eyes, ac, ic)
    assert b.v == pytest.approx(-0.30, abs=0.03)


def test_baseline_ignores_a_small_satellite_cluster_just_below():
    ac, ic = AppConfig().video.auto_calibration, AppConfig().video.incidents
    eyes = _cloud(0.13, -0.14, 400, seed=6) + _cloud(0.2, -0.20, 100, seed=7, spread=0.01)
    b = fit_eye_baseline(eyes, ac, ic)
    assert b.v == pytest.approx(-0.14, abs=0.03)


def test_baseline_falls_back_and_says_so_without_plausible_frames():
    ac, ic = AppConfig().video.auto_calibration, AppConfig().video.incidents
    b = fit_eye_baseline(_cloud(0.0, +0.6, 300, seed=5), ac, ic)
    assert b.fallback and "review the clips" in b.message
    assert fit_eye_baseline([None, None], ac, ic) is None


def test_near_baseline_selects_screen_frames_only():
    ic = AppConfig().video.incidents
    assert near_baseline(EyeDirection(0.1, 0.0, 0.35, 0.1), SCREEN, ic, 0.5)
    assert not near_baseline(EyeDirection(0.1, 0.25, 0.02, 0.1), SCREEN, ic, 0.5)     # above
    assert not near_baseline(EyeDirection(0.1, 0.0, 0.35, 0.9), SCREEN, ic, 0.5)      # blinking


def test_sustained_nearly_closed_eyes_count_as_down_but_blinks_do_not():
    ic = AppConfig().video.incidents
    times = [i / FPS for i in range(60)]
    open_ = EyeDirection(0.1, 0.0, 0.35, 0.1)
    shut = EyeDirection(0.1, 0.0, 0.9, 0.75)
    eyes = [open_] * 10 + [shut] * 3 + [open_] * 10 + [shut] * 30 + [open_] * 7
    labels = label_eye_frames(times, eyes, SCREEN, ic)
    assert labels[10:13] == [None] * 3                  # 0.3 s blink
    assert labels[23:53] == ["DOWN"] * 30               # 3 s with lids lowered
    ic.eye_closed_as_away_s = 0
    assert label_eye_frames(times, eyes, SCREEN, ic)[23:53] == [None] * 30


# --------------------------------------------------------------------------- #
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


def test_incident_lists_directions_in_order():
    cfg = AppConfig()
    recs = _records([(3, None), (3, "UP"), (1, None), (4, "DOWN-LEFT"), (3, None)])
    incs = extract_incidents(recs, cfg, FPS)
    assert len(incs) == 1 and incs[0].direction == "UP > DOWN-LEFT"


def test_short_isolated_glances_are_not_incidents():
    cfg = AppConfig()
    recs = _records([(5, None), (2, "RIGHT"), (5, None), (2, "UP-LEFT"), (5, None)])
    assert extract_incidents(recs, cfg, FPS) == []


def test_eye_rule_can_be_switched_off():
    cfg = AppConfig()
    cfg.video.incidents.enable_eye_direction_rule = False
    recs = _records([(2, None), (10, "DOWN"), (2, None)])
    assert extract_incidents(recs, cfg, FPS) == []


# --------------------------------------------------------------------------- #
def test_regression_sir_recording():
    """Eye scores from the real 'sir' recording (results/sir). The candidate
    reads the screen with glances left/right (0-11 s), looks ABOVE the
    screen (12-20 s), then down at the desk with lids lowered (21-28 s).
    Before v1.2 the 12-20 s period was taken as 'the screen'."""
    log = load_eye_log(DATA / "sir_eye_log.csv")
    rep = replay_eye_rule(log, AppConfig())
    assert rep.baseline.v == pytest.approx(-0.33, abs=0.05) and not rep.baseline.fallback

    def share(t0, t1, fam):
        sel = [lab for t, lab in zip(log.times, rep.labels) if t0 <= t < t1]
        return sum(1 for lab in sel if fam in families(lab)) / len(sel)

    assert share(12.5, 19.5, "UP") > 0.9
    assert share(21.0, 28.0, "DOWN") > 0.9
    assert share(5.0, 8.0, "UP") + share(5.0, 8.0, "DOWN") + share(5.0, 8.0, "LEFT") + share(5.0, 8.0, "RIGHT") == 0
    covered = lambda t: any(i.start_s <= t <= i.end_s for i in rep.incidents)
    assert all(covered(t) for t in (13.0, 16.0, 19.0, 22.0, 27.0))
    assert not covered(6.5)
    long_one = max(rep.incidents, key=lambda i: i.duration_s)
    assert "UP" in long_one.direction and "DOWN" in long_one.direction


def test_short_flicker_is_dropped_but_real_looks_are_kept():
    ic = AppConfig().video.incidents
    times = [i / 30 for i in range(120)]
    on, off = EyeDirection(0.1, 0.0, 0.35, 0.1), EyeDirection(0.9, 0.0, 0.35, 0.1)
    eyes = [on] * 20 + [off] * 3 + [on] * 30 + [off] * 30 + [on] * 37     # 0.1 s flicker, 1 s look
    labels = label_eye_frames(times, eyes, SCREEN, ic)
    assert labels[20:23] == [None] * 3
    assert labels[53:83] == ["RIGHT"] * 30
