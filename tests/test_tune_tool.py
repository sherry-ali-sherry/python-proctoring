"""
tests/test_tune_tool.py

tools/tune_eye_limits.py on the real 'sir' eye log with the example labels.
"""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import tune_eye_limits as tool  # noqa: E402

from config import AppConfig  # noqa: E402


@pytest.fixture
def sir_folder(tmp_path):
    folder = tmp_path / "sir"
    (folder / "data").mkdir(parents=True)
    shutil.copy(ROOT / "tests" / "data" / "sir_eye_log.csv", folder / "data" / "frame_log.csv")
    shutil.copy(ROOT / "tools" / "examples" / "sir_labels.csv", folder / "labels.csv")
    return folder


def test_labels_are_read_per_frame(sir_folder):
    s = tool.load_session(sir_folder)
    t = s.log.times
    at = lambda sec: s.truth[min(range(len(t)), key=lambda i: abs(t[i] - sec))]
    assert at(15.0) == 1 and at(6.0) == 0 and at(3.5) == -1


def test_current_limits_score_well_on_sir(sir_folder):
    s = tool.load_session(sir_folder)
    cfg = AppConfig()
    from gaze.eye_direction import fit_eye_baseline
    baselines = {s.name: fit_eye_baseline(s.log.eyes, cfg.video.auto_calibration, cfg.video.incidents)}
    frame, period = tool.evaluate([s], tool._with_limits(cfg, tool._current_limits(cfg)), baselines)
    assert frame.f1 > 0.95 and period.recall > 0.95


def test_template_and_tune_commands_run(sir_folder, tmp_path, capsys):
    other = tmp_path / "new"
    (other / "data").mkdir(parents=True)
    shutil.copy(sir_folder / "data" / "frame_log.csv", other / "data" / "frame_log.csv")
    assert tool.main(["template", str(other)]) == 0
    assert (other / "labels.csv").read_text().splitlines()[3] == "start_s,end_s,label"
    assert tool.main(["tune", str(sir_folder)]) == 0          # no --apply: nothing written
    out = capsys.readouterr().out
    assert "Best" in out and "Nothing saved" in out


def test_bad_labels_are_reported(tmp_path, sir_folder):
    (sir_folder / "labels.csv").write_text("start_s,end_s,label\n1,2,MAYBE\n")
    with pytest.raises(ValueError, match="SCREEN or AWAY"):
        tool.load_session(sir_folder)


def test_regression_sherry30_user_labelled(tmp_path):
    """The user's own flagged test video (2026-09-30): screen reading at two
    heights, above-screen 12-15 s, side glances, down 38-42 s, a head turn
    51-58 s and looking at the screen's corner 63-66 s (not cheating)."""
    folder = tmp_path / "sherry30"
    (folder / "data").mkdir(parents=True)
    shutil.copy(ROOT / "tests" / "data" / "sherry30_eye_log.csv", folder / "data" / "frame_log.csv")
    shutil.copy(ROOT / "tests" / "data" / "sherry30_labels.csv", folder / "labels.csv")
    s = tool.load_session(folder)
    cfg = AppConfig()
    from gaze.eye_direction import fit_eye_baseline
    b = fit_eye_baseline(s.log.eyes, cfg.video.auto_calibration, cfg.video.incidents)
    frame, period = tool.evaluate([s], cfg_limits := tool._with_limits(cfg, tool._current_limits(cfg)), {s.name: b})
    assert period.f1 >= 0.8 and period.precision >= 0.85
    _, in_period = tool.predict(s, cfg_limits, b)
    t = s.log.times
    share = lambda a, z: sum(1 for i, x in enumerate(t) if a <= x < z and in_period[i]) / sum(1 for x in t if a <= x < z)
    assert share(12.5, 15.5) > 0.9 and share(25, 29) > 0.9 and share(37, 41) > 0.9
    assert share(0, 7) == 0 and share(43, 50) == 0 and share(63.5, 65.5) < 0.2
