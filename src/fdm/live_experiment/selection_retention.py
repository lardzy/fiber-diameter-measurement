"""Ephemeral, non-interactive geometry for successful canvas picks.

This owner never holds a frame, candidate batch, input gesture or project field.
Each departing field has its own monotonic deadline, unaffected by repainting.
"""
from __future__ import annotations

from enum import Enum
import math
import time

from PySide6.QtCore import QObject, QTimer, Qt, Signal

from .presentation import category_color


class FieldEndCause(str, Enum):
    MOTION = "motion"
    NEW_FIELD = "new_field"
    CONTEXT = "context"


class SelectionRetention(QObject):
    changed = Signal()

    def __init__(self, session, parent=None, *, clock=time.monotonic):
        super().__init__(parent)
        self.session = session
        self.clock = clock
        self.duration = 1.0
        self._picked = {}
        self._groups = []
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._expire)

    def set_duration(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 10:
            raise ValueError("标记保留时长须为 0～10 秒。")
        self.duration = round(value, 1)
        if self.duration == 0:
            self._groups.clear()
            self._timer.stop()
            self.changed.emit()

    def remember(self, record):
        # Called only after a successful canvas-pick transaction. All values
        # are immutable scalars/tuples; no reference to a frame is retained.
        if len(record.endpoints or ()) != 2:
            return
        self._picked[record.id] = dict(record_id=record.id,
            start=tuple(record.endpoints[0]), end=tuple(record.endpoints[1]),
            number=record.provenance.get("measurement_number"),
            color=category_color(self.session, record.operator_id, record.direction, record.material_id))

    def end_field(self, cause):
        self.prune()
        if cause in {FieldEndCause.MOTION, FieldEndCause.NEW_FIELD}:
            if self._picked and self.duration > 0:
                self._groups.append((self.clock() + self.duration, tuple(self._picked.values())))
            self._picked.clear()
            self._schedule()
            self.changed.emit()
        else:
            self.clear()

    def prune(self):
        """Deletion/undo removes marks; redo never resurrects expired hints."""
        records = self.session.measurements
        self._picked = {key: value for key, value in self._picked.items() if key in records}
        now = self.clock()
        groups = [(deadline, tuple(mark for mark in marks if mark["record_id"] in records))
                  for deadline, marks in self._groups if deadline > now]
        groups = [(deadline, marks) for deadline, marks in groups if marks]
        changed = groups != self._groups
        self._groups = groups
        if changed:
            self._schedule()
            self.changed.emit()

    def visible_marks(self):
        now = self.clock()
        return tuple(dict(mark) for deadline, marks in self._groups if deadline > now
                     for mark in marks if mark["record_id"] in self.session.measurements)

    def _schedule(self):
        self._timer.stop()
        if self._groups:
            remaining = min(deadline for deadline, _ in self._groups) - self.clock()
            self._timer.start(max(1, math.ceil(remaining * 1000)))

    def _expire(self):
        self.prune()
        self._schedule()

    def clear(self):
        self._picked.clear()
        self._groups.clear()
        self._timer.stop()
        self.changed.emit()
