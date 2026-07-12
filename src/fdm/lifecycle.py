from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from fdm.ui.thread_task_manager import TaskStopResult


class TransitionIntent(str, Enum):
    CLOSE_WINDOW = "close_window"
    OPEN_DOCUMENT = "open_document"
    OPEN_PROJECT = "open_project"
    RESET_WORKSPACE = "reset_workspace"
    SWITCH_DEVICE = "switch_device"


class AcquisitionDisposition(str, Enum):
    KEEP_PARTIAL = "keep_partial"
    DISCARD = "discard"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class TransitionResult:
    intent: TransitionIntent
    completed: bool
    cancelled: bool = False
    timed_out: bool = False
    reason: str = ""
    task_results: tuple[TaskStopResult, ...] = ()


@dataclass(slots=True)
class DigitalSlideAcquisitionSession:
    """Identity and acceptance gate for one acquisition generation."""

    generation: int
    request_id: str
    output_path: Path
    accepting_frames: bool = True
    accepted_tiles: int = 0
    status: str = "capturing"
    terminal_reason: str = ""

    def stop_accepting(self, *, status: str, reason: str = "") -> None:
        self.accepting_frames = False
        self.status = str(status)
        self.terminal_reason = str(reason)

    def accepts(self, generation: int, request_id: str) -> bool:
        return (
            self.accepting_frames
            and self.matches(generation, request_id)
        )

    def matches(self, generation: int, request_id: str) -> bool:
        return int(generation) == self.generation and str(request_id) == self.request_id
