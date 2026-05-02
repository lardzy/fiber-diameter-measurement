from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class UndoCommand:
    label: str
    before: dict[str, Any]
    after: dict[str, Any]

    def undo(self, document: Any) -> None:
        document.restore_snapshot(self.before)

    def redo(self, document: Any) -> None:
        document.restore_snapshot(self.after)


@dataclass(slots=True)
class AddMeasurementCommand:
    label: str
    measurement_payload: dict[str, Any]
    index: int
    previous_selected_measurement_id: str | None
    previous_selected_overlay_id: str | None

    def undo(self, document: Any) -> None:
        document.remove_measurement_incremental(
            str(self.measurement_payload["id"]),
            select_measurement_id=self.previous_selected_measurement_id,
            select_overlay_id=self.previous_selected_overlay_id,
            mark_dirty=False,
        )
        document.refresh_dirty_flags()

    def redo(self, document: Any) -> None:
        from fdm.models import Measurement

        measurement = Measurement.from_dict(self.measurement_payload)
        document.insert_measurement_incremental(
            measurement,
            index=self.index,
            select=True,
            mark_dirty=False,
        )
        document.refresh_dirty_flags()


class DocumentHistory:
    def __init__(self) -> None:
        self._undo_stack: list[UndoCommand | AddMeasurementCommand] = []
        self._redo_stack: list[UndoCommand | AddMeasurementCommand] = []

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    def push(self, label: str, before: dict[str, Any], after: dict[str, Any]) -> None:
        if before == after:
            return
        self._undo_stack.append(UndoCommand(label=label, before=before, after=after))
        self._redo_stack.clear()

    def push_add_measurement(
        self,
        label: str,
        *,
        measurement_payload: dict[str, Any],
        index: int,
        previous_selected_measurement_id: str | None,
        previous_selected_overlay_id: str | None,
    ) -> None:
        self._undo_stack.append(
            AddMeasurementCommand(
                label=label,
                measurement_payload=measurement_payload,
                index=index,
                previous_selected_measurement_id=previous_selected_measurement_id,
                previous_selected_overlay_id=previous_selected_overlay_id,
            )
        )
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self, document: Any) -> bool:
        if not self._undo_stack:
            return False
        command = self._undo_stack.pop()
        command.undo(document)
        self._redo_stack.append(command)
        return True

    def redo(self, document: Any) -> bool:
        if not self._redo_stack:
            return False
        command = self._redo_stack.pop()
        command.redo(document)
        self._undo_stack.append(command)
        return True
