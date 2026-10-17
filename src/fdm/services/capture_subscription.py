"""GUI-thread subscriptions to the application's single camera session."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import weakref


@dataclass(frozen=True, slots=True)
class CaptureState:
    state: str
    generation: int
    stage: str = ""
    message: str = ""


class CaptureSubscription:
    def __init__(self, manager, owner_id: str, on_frame: Callable, on_state: Callable):
        self._manager = weakref.ref(manager)
        self.owner_id, self.on_frame, self.on_state = owner_id, on_frame, on_state
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            manager = self._manager()
            if manager is not None:
                manager._release_subscription(self)
