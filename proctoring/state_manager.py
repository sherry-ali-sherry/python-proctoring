"""
proctoring/state_manager.py

Hysteresis state machine for the final classification (spec section
19). Uses transitional states so a brief deviation cannot immediately
flip SCREEN -> AWAY, and symmetric transitional states for the return
path.

    LOOKING_AT_SCREEN --(deviation persists >= seconds_to_confirm_away)--> LOOKING_AWAY
    LOOKING_AWAY       --(back inside persists >= seconds_to_confirm_screen)--> LOOKING_AT_SCREEN
    (low confidence at any point) -> UNCERTAIN, with its own re-entry hysteresis
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import TemporalConfig
from gaze.gaze_classifier import RawGazeLabel


class ProctorState(str, Enum):
    LOOKING_AT_SCREEN = "LOOKING_AT_SCREEN"
    POSSIBLE_AWAY = "POSSIBLE_AWAY"
    LOOKING_AWAY = "LOOKING_AWAY"
    POSSIBLE_SCREEN = "POSSIBLE_SCREEN"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class StateTransition:
    changed: bool
    previous_state: ProctorState
    new_state: ProctorState
    state: ProctorState


class GazeStateManager:
    """Applies duration-based hysteresis on top of the per-frame raw
    label to produce a stable ProctorState."""

    def __init__(self, cfg: TemporalConfig):
        self._cfg = cfg
        self._state = ProctorState.UNCERTAIN
        self._pending_since: Optional[float] = None
        self._pending_label: Optional[RawGazeLabel] = None

    @property
    def state(self) -> ProctorState:
        return self._state

    def update(self, raw_label: RawGazeLabel, now: Optional[float] = None) -> StateTransition:
        now = now if now is not None else time.time()
        prev = self._state

        if raw_label == RawGazeLabel.UNCERTAIN:
            self._state = ProctorState.UNCERTAIN
            self._pending_since = None
            self._pending_label = None
            return StateTransition(prev != self._state, prev, self._state, self._state)

        if self._state in (ProctorState.LOOKING_AT_SCREEN, ProctorState.UNCERTAIN):
            if raw_label == RawGazeLabel.SCREEN:
                self._state = ProctorState.LOOKING_AT_SCREEN
                self._pending_since = None
            else:  # AWAY
                self._begin_or_continue_pending(RawGazeLabel.AWAY, now)
                if self._elapsed(now) >= self._cfg.seconds_to_confirm_away:
                    self._state = ProctorState.LOOKING_AWAY
                    self._pending_since = None
                else:
                    self._state = ProctorState.POSSIBLE_AWAY

        elif self._state == ProctorState.POSSIBLE_AWAY:
            if raw_label == RawGazeLabel.AWAY:
                self._begin_or_continue_pending(RawGazeLabel.AWAY, now)
                if self._elapsed(now) >= self._cfg.seconds_to_confirm_away:
                    self._state = ProctorState.LOOKING_AWAY
                    self._pending_since = None
            else:  # gaze returned to screen before confirmation
                self._state = ProctorState.LOOKING_AT_SCREEN
                self._pending_since = None

        elif self._state == ProctorState.LOOKING_AWAY:
            if raw_label == RawGazeLabel.SCREEN:
                self._begin_or_continue_pending(RawGazeLabel.SCREEN, now)
                if self._elapsed(now) >= self._cfg.seconds_to_confirm_screen:
                    self._state = ProctorState.LOOKING_AT_SCREEN
                    self._pending_since = None
                else:
                    self._state = ProctorState.POSSIBLE_SCREEN
            else:
                self._pending_since = None

        elif self._state == ProctorState.POSSIBLE_SCREEN:
            if raw_label == RawGazeLabel.SCREEN:
                self._begin_or_continue_pending(RawGazeLabel.SCREEN, now)
                if self._elapsed(now) >= self._cfg.seconds_to_confirm_screen:
                    self._state = ProctorState.LOOKING_AT_SCREEN
                    self._pending_since = None
            else:
                self._state = ProctorState.LOOKING_AWAY
                self._pending_since = None

        return StateTransition(prev != self._state, prev, self._state, self._state)

    def _begin_or_continue_pending(self, label: RawGazeLabel, now: float) -> None:
        if self._pending_label != label or self._pending_since is None:
            self._pending_label = label
            self._pending_since = now

    def _elapsed(self, now: float) -> float:
        if self._pending_since is None:
            return 0.0
        return now - self._pending_since
