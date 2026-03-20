from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class UndoCommand:
    label: str
    before: dict[str, Any]
    after: dict[str, Any]


class DocumentHistory:
    def __init__(self) -> None:
        self._undo_stack: list[UndoCommand] = []
        self._redo_stack: list[UndoCommand] = []

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    def push(self, label: str, before: dict[str, Any], after: dict[str, Any]) -> None:
        if before == after:
            return
        self._undo_stack.append(UndoCommand(label=label, before=before, after=after))
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self, document: Any) -> bool:
        if not self._undo_stack:
            return False
        command = self._undo_stack.pop()
        document.restore_snapshot(command.before)
        self._redo_stack.append(command)
        return True

    def redo(self, document: Any) -> bool:
        if not self._redo_stack:
            return False
        command = self._redo_stack.pop()
        document.restore_snapshot(command.after)
        self._undo_stack.append(command)
        return True
