"""
proctoring/event_engine.py

Turns state changes + face-count observations into discrete
proctoring EVENTS (spec section 20) with temporal-persistence
requirements (spec sections 21-22) -- a single noisy frame never
produces MULTIPLE_FACES or FACE_LOST.

This module deliberately does NOT decide "cheating" -- it only emits
structured observations for a downstream policy layer (out of scope
for phase 1, spec section 36).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from config import TemporalConfig
from proctoring.state_manager import ProctorState


@dataclass
class ProctorEvent:
    timestamp: float
    event_type: str
    direction: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    confidence: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "event": self.event_type,
            "direction": self.direction,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
            **self.metadata,
        }


class EventEngine:
    def __init__(self, cfg: TemporalConfig):
        self._cfg = cfg
        self.events: List[ProctorEvent] = []

        self._away_started_at: Optional[float] = None
        self._away_direction: Optional[str] = None
        self._away_confirmed = False

        self._face_lost_since: Optional[float] = None
        self._face_lost_confirmed = False
        self._face_present_since: Optional[float] = None

        self._multi_face_since: Optional[float] = None
        self._multi_face_confirmed = False

        self._low_conf_reported = False

    # -- gaze state -> events -------------------------------------------------
    def on_state(self, state: ProctorState, direction: Optional[str], confidence: float, now: float) -> None:
        if state == ProctorState.LOOKING_AWAY:
            if self._away_started_at is None:
                self._away_started_at = now
                self._away_direction = direction
                self._away_confirmed = False
            if not self._away_confirmed:
                self._away_confirmed = True
                self._emit("LOOKING_AWAY_STARTED", now, direction=direction, confidence=confidence,
                           start_time=self._away_started_at)
            else:
                self._emit("LOOKING_AWAY_CONTINUED", now, direction=direction, confidence=confidence,
                           start_time=self._away_started_at,
                           duration=now - self._away_started_at)
        else:
            if self._away_confirmed and self._away_started_at is not None:
                self._emit(
                    "LOOKING_AT_SCREEN_RESUMED", now, direction=self._away_direction,
                    confidence=confidence, start_time=self._away_started_at,
                    end_time=now, duration=now - self._away_started_at,
                )
            self._away_started_at = None
            self._away_direction = None
            self._away_confirmed = False

        if state == ProctorState.UNCERTAIN and confidence < 0.2:
            if not self._low_conf_reported:
                self._emit("LOW_GAZE_CONFIDENCE", now, confidence=confidence)
                self._low_conf_reported = True
        else:
            self._low_conf_reported = False

    # -- face presence/count -> events -----------------------------------------
    def on_face_count(self, num_faces: int, now: float) -> None:
        if num_faces == 0:
            if self._face_lost_since is None:
                self._face_lost_since = now
                self._face_lost_confirmed = False
            elapsed = now - self._face_lost_since
            if elapsed >= self._cfg.seconds_to_confirm_face_lost and not self._face_lost_confirmed:
                self._face_lost_confirmed = True
                self._emit("FACE_LOST", now, start_time=self._face_lost_since, duration=elapsed)
            self._face_present_since = None
        else:
            if self._face_lost_since is not None:
                if self._face_present_since is None:
                    self._face_present_since = now
                if now - self._face_present_since >= self._cfg.seconds_to_clear_face_lost:
                    self._face_lost_since = None
                    self._face_lost_confirmed = False
                    self._face_present_since = None
            else:
                self._face_present_since = now

        if num_faces > 1:
            if self._multi_face_since is None:
                self._multi_face_since = now
                self._multi_face_confirmed = False
            elapsed = now - self._multi_face_since
            if elapsed >= self._cfg.seconds_to_confirm_multi_face and not self._multi_face_confirmed:
                self._multi_face_confirmed = True
                self._emit("MULTIPLE_FACES_DETECTED", now, start_time=self._multi_face_since,
                           duration=elapsed, metadata={"num_faces": num_faces})
        else:
            self._multi_face_since = None
            self._multi_face_confirmed = False

    def on_calibration_failed(self, message: str, now: float) -> None:
        self._emit("CALIBRATION_FAILED", now, metadata={"message": message})

    def _emit(self, event_type: str, now: float, direction: Optional[str] = None,
              confidence: Optional[float] = None, start_time: Optional[float] = None,
              end_time: Optional[float] = None, duration: Optional[float] = None,
              metadata: Optional[dict] = None) -> None:
        self.events.append(
            ProctorEvent(
                timestamp=now, event_type=event_type, direction=direction,
                start_time=start_time, end_time=end_time, duration=duration,
                confidence=confidence, metadata=metadata or {},
            )
        )
