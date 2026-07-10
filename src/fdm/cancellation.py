from __future__ import annotations

from threading import Event


class CancellationError(RuntimeError):
    """Raised when a cooperative operation observes cancellation."""


class CancellationToken:
    """Read-only cancellation state that is safe to inspect from any thread."""

    __slots__ = ("_event",)

    def __init__(self, event: Event) -> None:
        self._event = event

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise CancellationError("操作已取消。")


class CancellationTokenSource:
    """Owner of a cooperative cancellation signal and its read-only token."""

    __slots__ = ("_event", "_token")

    def __init__(self) -> None:
        self._event = Event()
        self._token = CancellationToken(self._event)

    @property
    def token(self) -> CancellationToken:
        return self._token

    def cancel(self) -> bool:
        """Request cancellation and return whether this was the first request."""

        was_cancelled = self._event.is_set()
        self._event.set()
        return not was_cancelled


# Backward-compatible short name used by the first integration pass.
CancellationSource = CancellationTokenSource
