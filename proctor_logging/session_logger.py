"""
logging/session_logger.py

Optional structured logging (spec sections 25-26). Two separate
outputs:
    - frame-level CSV (numeric time series -- state, angles, 3D gaze
      geometry, screen intersection, detection flags)
    - event-level JSON (discrete ProctorEvent records)

Deliberately does NOT save raw video (spec section 25).
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from proctoring.event_engine import ProctorEvent

FRAME_LOG_FIELDS = [
    "timestamp", "state", "confidence", "yaw", "pitch", "roll",
    "left_eye_x", "left_eye_y", "left_eye_z",
    "right_eye_x", "right_eye_y", "right_eye_z",
    "left_gaze_x", "left_gaze_y", "left_gaze_z",
    "right_gaze_x", "right_gaze_y", "right_gaze_z",
    "combined_gaze_x", "combined_gaze_y", "combined_gaze_z",
    "screen_intersection_x", "screen_intersection_y",
    "face_detected", "iris_detected", "number_of_faces",
]


class SessionLogger:
    def __init__(self, log_dir: str, session_name: Optional[str] = None):
        self._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        name = session_name or time.strftime("session_%Y%m%d_%H%M%S")
        self._csv_path = os.path.join(log_dir, f"{name}_frames.csv")
        self._json_path = os.path.join(log_dir, f"{name}_events.json")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=FRAME_LOG_FIELDS)
        self._csv_writer.writeheader()

    def log_frame(self, row: dict) -> None:
        safe_row = {k: row.get(k, "") for k in FRAME_LOG_FIELDS}
        self._csv_writer.writerow(safe_row)

    def flush(self) -> None:
        self._csv_file.flush()

    def save_events(self, events: List[ProctorEvent]) -> None:
        with open(self._json_path, "w") as f:
            json.dump([e.to_dict() for e in events], f, indent=2)

    def close(self, events: Optional[List[ProctorEvent]] = None) -> None:
        if events is not None:
            self.save_events(events)
        self._csv_file.close()

    @property
    def csv_path(self) -> str:
        return self._csv_path

    @property
    def json_path(self) -> str:
        return self._json_path
