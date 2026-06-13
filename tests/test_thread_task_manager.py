from __future__ import annotations

from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from PySide6.QtCore import QObject, Signal, Slot
    from PySide6.QtWidgets import QApplication

    from fdm.ui.thread_task_manager import ThreadTaskManager

    QT_TASK_MANAGER_AVAILABLE = True
except ModuleNotFoundError:
    QObject = object  # type: ignore[assignment]
    QApplication = None  # type: ignore[assignment]
    Signal = None  # type: ignore[assignment]
    Slot = lambda *args, **kwargs: (lambda fn: fn)  # type: ignore[assignment]
    ThreadTaskManager = None  # type: ignore[assignment]
    QT_TASK_MANAGER_AVAILABLE = False


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _spin_until(predicate, *, timeout_s: float = 1.5) -> None:
    app = _app()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


@unittest.skipUnless(QT_TASK_MANAGER_AVAILABLE, "requires PySide6")
class ThreadTaskManagerTests(unittest.TestCase):
    def test_one_shot_worker_starts_and_cleans_up_after_finished(self) -> None:
        events: list[str] = []

        class Worker(QObject):
            finished = Signal()

            @Slot()
            def run(self) -> None:
                events.append("run")
                self.finished.emit()

        manager = ThreadTaskManager(parent=_app())
        manager.start_one_shot("load", worker_factory=Worker)

        _spin_until(lambda: manager.worker("load") is None)

        self.assertEqual(events, ["run"])
        self.assertFalse(manager.is_running("load"))

    def test_stop_calls_cancel_and_waits_for_worker_thread(self) -> None:
        events: list[str] = []

        class Worker(QObject):
            def cancel(self) -> None:
                events.append("cancel")

        manager = ThreadTaskManager(parent=_app())
        manager.ensure_persistent("prompt", worker_factory=Worker)
        _spin_until(lambda: manager.is_running("prompt"))

        manager.stop("prompt", cancel=True)

        self.assertEqual(events, ["cancel"])
        self.assertFalse(manager.is_running("prompt"))
        self.assertIsNone(manager.worker("prompt"))

    def test_persistent_worker_is_reused_until_stopped(self) -> None:
        class Worker(QObject):
            pass

        manager = ThreadTaskManager(parent=_app())
        first = manager.ensure_persistent("geometry", worker_factory=Worker)
        _spin_until(lambda: manager.is_running("geometry"))
        second = manager.ensure_persistent("geometry", worker_factory=Worker)

        self.assertIs(first.worker, second.worker)

        manager.stop("geometry", cancel=False)
        rebuilt = manager.ensure_persistent("geometry", worker_factory=Worker)
        _spin_until(lambda: manager.is_running("geometry"))

        self.assertIsNot(first.worker, rebuilt.worker)
        manager.shutdown_all(cancel=False)


if __name__ == "__main__":
    unittest.main()
