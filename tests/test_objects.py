"""
tests/test_objects.py

Phone / book / extra-person evidence, incidents and verdict. Uses made-up
detections, so no model file is needed.
"""
import pytest

from config import AppConfig
from detection.evidence import build_evidence
from detection.object_detector import BOOK, PERSON, PHONE, DetectedObject, count_people
from proctoring.state_manager import ProctorState as S
from video_analysis.incidents import (
    REASON_BOOK, REASON_PEOPLE, REASON_PHONE, VERDICT_CHEATING, VERDICT_NO_CHEATING, FrameRecord,
    compute_stats, compute_verdict, extract_incidents,
)

FPS = 10.0
CANDIDATE = DetectedObject(PERSON, 0.9, (0.3, 0.2, 0.4, 0.8))
PHONE_IN_HAND = DetectedObject(PHONE, 0.8, (0.6, 0.6, 0.08, 0.12))
SHELF_BOOK = DetectedObject(BOOK, 0.6, (0.05, 0.05, 0.1, 0.15))
HELD_BOOK = DetectedObject(BOOK, 0.6, (0.4, 0.55, 0.2, 0.2))
SECOND_PERSON = DetectedObject(PERSON, 0.8, (0.75, 0.2, 0.2, 0.7))


def _timeline(spec, every=5):
    """spec: list of (seconds, detections). Frames at FPS; every `every`-th
    frame is sampled (2 samples per second at 10 fps)."""
    times, sampled, t = [], [], 0.0
    idx = 0
    for seconds, dets in spec:
        for _ in range(int(round(seconds * FPS))):
            times.append(idx / FPS)
            sampled.append(list(dets) if idx % every == 0 else None)
            idx += 1
    return times, sampled


def _records(times, evidence, faces=1):
    return [FrameRecord(index=i, time_s=t, num_faces=faces, state=S.LOOKING_AT_SCREEN, eye_x=0.0,
                        phone_score=ev.phone, book_score=ev.book, people=ev.people, objects=ev.objects)
            for i, (t, ev) in enumerate(zip(times, evidence.frames))]


def _analyse(spec, faces=1):
    cfg = AppConfig()
    times, sampled = _timeline(spec)
    ev = build_evidence(times, sampled, cfg.video.objects)
    recs = _records(times, ev, faces)
    incs = extract_incidents(recs, cfg, FPS)
    stats = compute_stats(recs, incs, FPS)
    return ev, incs, stats, compute_verdict(stats, cfg, True)


def test_phone_in_hand_is_an_incident_and_cheating():
    ev, incs, stats, verdict = _analyse([(5, [CANDIDATE]), (3, [CANDIDATE, PHONE_IN_HAND]), (5, [CANDIDATE])])
    assert [i.reasons for i in incs] == [[REASON_PHONE]]
    assert incs[0].start_s == pytest.approx(5.0) and 7.5 <= incs[0].end_s <= 8.9
    assert stats.phone_s > 2.5 and stats.incident_count == 0 and stats.total_not_looking_s == 0
    assert verdict.label == VERDICT_CHEATING and any("phone" in r for r in verdict.reasons)


def test_a_single_sampled_frame_is_not_enough():
    _, incs, _, verdict = _analyse([(5, [CANDIDATE]), (0.5, [CANDIDATE, PHONE_IN_HAND]), (5, [CANDIDATE])])
    assert incs == [] and verdict.label == VERDICT_NO_CHEATING


def test_low_confidence_phone_is_ignored():
    weak = DetectedObject(PHONE, 0.3, PHONE_IN_HAND.box)
    _, incs, _, _ = _analyse([(5, [CANDIDATE]), (4, [CANDIDATE, weak]), (5, [CANDIDATE])])
    assert incs == []


def test_book_being_read_counts_but_a_shelf_book_does_not():
    ev, incs, stats, verdict = _analyse([(6, [CANDIDATE, SHELF_BOOK]), (3, [CANDIDATE, SHELF_BOOK, HELD_BOOK]),
                                         (6, [CANDIDATE, SHELF_BOOK])])
    assert [i.reasons for i in incs] == [[REASON_BOOK]]
    assert 5.5 <= incs[0].start_s <= 6.5
    assert ev.notes and "background" in ev.notes[0]
    assert verdict.label == VERDICT_CHEATING


def test_second_person_is_cheating_and_candidate_duplicates_are_not():
    dup = DetectedObject(PERSON, 0.7, (0.31, 0.22, 0.38, 0.78))       # same person, second box
    assert count_people([CANDIDATE, dup], 0.6, 0.02) == 1
    assert count_people([CANDIDATE, SECOND_PERSON], 0.6, 0.02) == 2
    tiny = DetectedObject(PERSON, 0.9, (0.9, 0.1, 0.02, 0.05))
    assert count_people([CANDIDATE, tiny], 0.6, 0.02) == 1
    _, incs, stats, verdict = _analyse([(4, [CANDIDATE]), (4, [CANDIDATE, SECOND_PERSON]), (4, [CANDIDATE])])
    assert [i.reasons for i in incs] == [[REASON_PEOPLE]]
    assert stats.multiple_people_s >= 2.0 and verdict.label == VERDICT_CHEATING


def test_two_faces_count_as_two_people_even_without_person_boxes():
    _, incs, _, verdict = _analyse([(6, [CANDIDATE])], faces=2)
    assert [i.reasons for i in incs] == [[REASON_PEOPLE]] and verdict.label == VERDICT_CHEATING


def test_object_rules_can_be_switched_off():
    cfg = AppConfig()
    cfg.video.objects.phone_is_cheating = False
    times, sampled = _timeline([(5, [CANDIDATE]), (3, [CANDIDATE, PHONE_IN_HAND]), (5, [CANDIDATE])])
    recs = _records(times, build_evidence(times, sampled, cfg.video.objects))
    incs = extract_incidents(recs, cfg, FPS)
    stats = compute_stats(recs, incs, FPS)
    assert incs and compute_verdict(stats, cfg, True).label == VERDICT_NO_CHEATING   # still shown, not in verdict
    cfg.video.objects.enabled = False
    assert extract_incidents(recs, cfg, FPS) == []


def test_evidence_is_not_held_across_a_long_sampling_gap():
    cfg = AppConfig()
    times = [i / FPS for i in range(40)]
    sampled = [[CANDIDATE, PHONE_IN_HAND]] + [None] * 39      # one sample, then nothing
    ev = build_evidence(times, sampled, cfg.video.objects)
    held = [f.phone is not None for f in ev.frames]
    assert held[0] and not held[-1] and sum(held) <= 1.5 / cfg.video.objects.samples_per_second * FPS + 1
