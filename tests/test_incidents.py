"""
tests/test_incidents.py

Incident extraction, statistics and verdict over synthetic per-frame
records (no video / MediaPipe needed).
"""
import pytest

from config import AppConfig
from proctoring.state_manager import ProctorState as S
from video_analysis.incidents import (
    FrameRecord, REASON_FACE, REASON_GAZE, REASON_HEAD, VERDICT_CHEATING, VERDICT_INCONCLUSIVE,
    VERDICT_NO_CHEATING, compute_stats, compute_verdict, extract_incidents, timecode,
)

FPS = 10.0


def _records(spec):
    """spec: list of (seconds, state, num_faces, head_turned, direction)."""
    out, idx = [], 0
    for seconds, state, faces, turned, direction in spec:
        for _ in range(int(round(seconds * FPS))):
            out.append(FrameRecord(index=idx, time_s=idx / FPS, num_faces=faces, state=state,
                                   away_direction=direction, head_turned=turned,
                                   head_yaw_offset=40.0 if turned else 0.0, head_pitch_offset=0.0))
            idx += 1
    return out


def screen(sec):
    return (sec, S.LOOKING_AT_SCREEN, 1, False, None)


def test_timecode_format():
    assert timecode(0) == "00:00:00.000"
    assert timecode(3725.5) == "01:02:05.500"
    assert timecode(61.25, sep="-") == "00-01-01.250"


def test_brief_glance_is_not_an_incident():
    cfg = AppConfig()
    recs = _records([screen(10), (2.0, S.POSSIBLE_AWAY, 1, False, "LEFT"), screen(10)])
    assert extract_incidents(recs, cfg, FPS) == []


def test_confirmed_look_away_starts_when_gaze_left():
    cfg = AppConfig()
    recs = _records([
        screen(10),
        (3.5, S.POSSIBLE_AWAY, 1, False, "LEFT"),
        (4.0, S.LOOKING_AWAY, 1, False, "LEFT"),
        (0.5, S.POSSIBLE_SCREEN, 1, False, None),
        screen(10),
    ])
    incs = extract_incidents(recs, cfg, FPS)
    assert len(incs) == 1
    inc = incs[0]
    assert inc.start_s == pytest.approx(10.0)
    assert inc.end_s == pytest.approx(17.5)          # last LOOKING_AWAY frame + 1 frame
    assert inc.reasons == [REASON_GAZE]
    assert inc.direction == "LEFT"
    assert inc.start_frame == 100 and inc.end_frame == 174
    assert all(r.incident_number == 1 for r in recs[100:175])
    assert recs[99].incident_number is None and recs[175].incident_number is None


def test_face_lost_needs_persistence():
    cfg = AppConfig()   # seconds_to_confirm_face_lost = 1.5
    short = _records([screen(5), (1.0, S.UNCERTAIN, 0, False, None), screen(5)])
    assert extract_incidents(short, cfg, FPS) == []
    long = _records([screen(5), (3.0, S.UNCERTAIN, 0, False, None), screen(5)])
    incs = extract_incidents(long, cfg, FPS)
    assert len(incs) == 1 and incs[0].reasons == [REASON_FACE] and incs[0].direction is None
    assert incs[0].duration_s == pytest.approx(3.0)


def test_head_turn_rule_and_toggle():
    cfg = AppConfig()   # seconds_to_confirm_away = 3.5
    recs = _records([screen(5), (5.0, S.UNCERTAIN, 1, True, None), screen(5)])
    incs = extract_incidents(recs, cfg, FPS)
    assert len(incs) == 1 and incs[0].reasons == [REASON_HEAD] and incs[0].direction == "RIGHT"
    cfg.video.incidents.enable_head_turn_rule = False
    assert extract_incidents(recs, cfg, FPS) == []


def test_adjacent_intervals_merge_with_all_reasons():
    cfg = AppConfig()   # merge_gap_s = 1.5
    recs = _records([
        screen(5),
        (3.5, S.POSSIBLE_AWAY, 1, False, "DOWN"),
        (2.0, S.LOOKING_AWAY, 1, False, "DOWN"),
        (1.0, S.UNCERTAIN, 1, False, None),           # gap 1.0 s <= merge gap
        (2.0, S.UNCERTAIN, 0, False, None),           # face lost 2 s
        screen(5),
    ])
    incs = extract_incidents(recs, cfg, FPS)
    assert len(incs) == 1
    assert incs[0].reasons == [REASON_GAZE, REASON_FACE]
    assert incs[0].start_s == pytest.approx(5.0)
    assert incs[0].end_s == pytest.approx(13.5)


def test_gaze_rule_skipped_when_gaze_unavailable():
    cfg = AppConfig()
    recs = _records([screen(5), (3.5, S.POSSIBLE_AWAY, 1, False, "UP"), (3.0, S.LOOKING_AWAY, 1, False, "UP"), screen(5)])
    assert len(extract_incidents(recs, cfg, FPS, gaze_available=True)) == 1
    assert extract_incidents(recs, cfg, FPS, gaze_available=False) == []


def _away_block(seconds):
    return [(3.5, S.POSSIBLE_AWAY, 1, False, "LEFT"), (seconds - 3.5, S.LOOKING_AWAY, 1, False, "LEFT")]


def test_verdict_no_cheating_when_always_on_screen():
    cfg = AppConfig()
    recs = _records([screen(60)])
    incs = extract_incidents(recs, cfg, FPS)
    stats = compute_stats(recs, incs, FPS)
    assert stats.analyzed_duration_s == pytest.approx(60.0)
    assert stats.gaze_coverage == pytest.approx(1.0)
    assert compute_verdict(stats, cfg, True).label == VERDICT_NO_CHEATING


def test_verdict_cheating_on_long_single_incident():
    cfg = AppConfig()   # min_single_incident_s = 8
    recs = _records([screen(100), *_away_block(9.0), screen(100)])
    incs = extract_incidents(recs, cfg, FPS)
    v = compute_verdict(compute_stats(recs, incs, FPS), cfg, True)
    assert v.label == VERDICT_CHEATING and v.is_cheating


def test_verdict_cheating_on_repeated_incidents():
    cfg = AppConfig()   # min_incidents = 3
    spec = [screen(60)]
    for _ in range(3):
        spec += [*_away_block(5.0), screen(60)]
    recs = _records(spec)
    incs = extract_incidents(recs, cfg, FPS)
    assert len(incs) == 3
    assert compute_verdict(compute_stats(recs, incs, FPS), cfg, True).label == VERDICT_CHEATING


def test_verdict_short_incident_below_limits_is_no_cheating():
    cfg = AppConfig()
    recs = _records([screen(100), *_away_block(5.0), screen(100)])
    incs = extract_incidents(recs, cfg, FPS)
    assert len(incs) == 1
    assert compute_verdict(compute_stats(recs, incs, FPS), cfg, True).label == VERDICT_NO_CHEATING


def test_verdict_inconclusive_when_gaze_rarely_measurable():
    cfg = AppConfig()
    recs = _records([screen(10), (50, S.UNCERTAIN, 1, False, None)])
    incs = extract_incidents(recs, cfg, FPS)
    stats = compute_stats(recs, incs, FPS)
    assert stats.gaze_coverage == pytest.approx(10 / 60)
    assert compute_verdict(stats, cfg, True).label == VERDICT_INCONCLUSIVE
    assert compute_verdict(compute_stats(_records([screen(60)]), [], FPS), cfg, False).label == VERDICT_INCONCLUSIVE
