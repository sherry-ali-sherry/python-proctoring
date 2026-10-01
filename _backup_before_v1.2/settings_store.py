"""
settings_store.py

User-editable detection settings, persisted in settings.json next to
this file and applied on top of the defaults in config.py. Only the
fields listed in FIELDS can be changed this way (the web Settings page,
and every entry point that builds an AppConfig through load_config()).
Values are type-checked and range-checked; unknown keys are ignored.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import AppConfig

PROJECT_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = PROJECT_DIR / "settings.json"


@dataclass(frozen=True)
class Field:
    key: str                # dotted path inside AppConfig
    kind: type              # float | int | bool
    label: str
    help: str
    group: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    percent: bool = False   # stored as a fraction, shown as a percentage


FIELDS: List[Field] = [
    Field("video.verdict.min_incidents", int, "Periods of not looking", "CHEATING when there are at least this many separate periods.", "Verdict", 1, 1000),
    Field("video.verdict.min_single_incident_s", float, "Longest single period (s)", "CHEATING when any one period lasts at least this long.", "Verdict", 1, 3600),
    Field("video.verdict.min_not_looking_fraction", float, "Share of the video not looking (%)", "CHEATING when the periods add up to at least this share of the video.", "Verdict", 0.01, 1.0, percent=True),
    Field("video.verdict.min_gaze_coverage", float, "Minimum measurable gaze (%)", "Below this, a result that would be NO CHEATING becomes INCONCLUSIVE.", "Verdict", 0.0, 1.0, percent=True),
    Field("temporal.seconds_to_confirm_away", float, "Look-away time before it counts (s)", "Eyes must stay off the screen this long. Also the head-turn persistence.", "Detection", 0.5, 30),
    Field("temporal.seconds_to_confirm_face_lost", float, "Face-missing time before it counts (s)", "No face in the frame for this long counts as not looking.", "Detection", 0.2, 30),
    Field("video.incidents.merge_gap_s", float, "Merge periods closer than (s)", "Two periods separated by a shorter gap become one.", "Detection", 0, 30),
    Field("video.incidents.min_incident_duration_s", float, "Ignore periods shorter than (s)", "Merged periods shorter than this are dropped as noise.", "Detection", 0, 60),
    Field("video.incidents.enable_eye_direction_rule", bool, "Count eyes held away as not looking", "Catches reading notes beside or below the screen, even with the head facing forward.", "Eyes"),
    Field("video.incidents.eye_side_threshold", float, "Eyes sideways limit (0-1)", "How far the eyes may turn left/right before it counts. On screen ~0.1; reading off-screen ~0.5+. Lower = stricter.", "Eyes", 0.1, 1.0),
    Field("video.incidents.eye_down_threshold", float, "Eyes down limit (0-1)", "How far the eyes may look down before it counts. Lower = stricter.", "Eyes", 0.1, 1.0),
    Field("video.incidents.eye_up_threshold", float, "Eyes up limit (0-1)", "How far the eyes may look up before it counts. Lower = stricter.", "Eyes", 0.1, 1.0),
    Field("video.incidents.count_face_lost_as_not_looking", bool, "Count a missing face as not looking", "", "Detection"),
    Field("video.incidents.enable_head_turn_rule", bool, "Count a large head turn as not looking", "", "Detection"),
    Field("video.incidents.head_turn_yaw_deg", float, "Head turn left/right limit (deg)", "Relative to the candidate's own usual head position.", "Detection", 10, 90),
    Field("video.incidents.head_turn_pitch_deg", float, "Head turn up/down limit (deg)", "Relative to the candidate's own usual head position.", "Detection", 10, 90),
    Field("video.auto_calibration.screen_half_angle_x_deg", float, "Screen half-width (deg)", "How far left/right of the screen centre still counts as on screen.", "Screen", 5, 60),
    Field("video.auto_calibration.screen_half_angle_y_deg", float, "Screen half-height (deg)", "How far up/down of the screen centre still counts as on screen.", "Screen", 3, 45),
    Field("video.clip_padding_s", float, "Clip context before/after (s)", "Extra video kept around each incident clip.", "Output", 0, 30),
    Field("video.export_annotated_video", bool, "Also export the full annotated video", "Slower; writes the whole video with a status banner.", "Output"),
]
_BY_KEY = {f.key: f for f in FIELDS}


def _get(cfg: AppConfig, key: str) -> Any:
    obj = cfg
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


def _set(cfg: AppConfig, key: str, value: Any) -> None:
    *parents, leaf = key.split(".")
    obj = cfg
    for part in parents:
        obj = getattr(obj, part)
    setattr(obj, leaf, value)


def _coerce(field: Field, raw: Any) -> Tuple[Any, Optional[str]]:
    if field.kind is bool:
        if isinstance(raw, bool):
            return raw, None
        return None, f"{field.label}: must be true or false."
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None, f"{field.label}: must be a number."
    if field.kind is int:
        if float(raw) != int(raw):
            return None, f"{field.label}: must be a whole number."
        value: Any = int(raw)
    else:
        value = float(raw)
    if field.minimum is not None and value < field.minimum:
        return None, f"{field.label}: must be at least {_display(field, field.minimum)}."
    if field.maximum is not None and value > field.maximum:
        return None, f"{field.label}: must be at most {_display(field, field.maximum)}."
    return value, None


def _display(field: Field, value: float) -> str:
    return f"{value * 100:g}%" if field.percent else f"{value:g}"


def validate(values: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Returns (clean values for known keys, error messages)."""
    clean, errors = {}, []
    for key, raw in (values or {}).items():
        field = _BY_KEY.get(key)
        if field is None:
            continue
        value, error = _coerce(field, raw)
        if error:
            errors.append(error)
        else:
            clean[key] = value
    return clean, errors


def load_overrides(path: Optional[Path] = None) -> Dict[str, Any]:
    try:
        with open(path or SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    clean, _ = validate(data if isinstance(data, dict) else {})
    return clean


def save_overrides(values: Dict[str, Any], path: Optional[Path] = None) -> None:
    """Atomic write, so a crash never leaves a half-written file."""
    path = Path(path or SETTINGS_FILE)
    fd, tmp = tempfile.mkstemp(prefix="settings_", suffix=".json", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def apply_overrides(cfg: AppConfig, values: Dict[str, Any]) -> AppConfig:
    for key, value in values.items():
        _set(cfg, key, value)
    return cfg


def load_config(path: Optional[Path] = None) -> AppConfig:
    """AppConfig defaults + the saved settings.json overrides."""
    return apply_overrides(AppConfig(), load_overrides(path))


def describe(path: Optional[Path] = None) -> List[dict]:
    """Every editable field with its current and default value (for the UI)."""
    current = load_config(path)
    defaults = AppConfig()
    return [{
        "key": f.key, "type": f.kind.__name__, "label": f.label, "help": f.help, "group": f.group,
        "min": f.minimum, "max": f.maximum, "percent": f.percent,
        "value": _get(current, f.key), "default": _get(defaults, f.key),
    } for f in FIELDS]
