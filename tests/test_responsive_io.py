"""File-operation completion must survive delayed timers and modal dispatch."""
from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess
import sys
from threading import Event

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtWidgets import QProgressBar, QWidget

from fdm.ui import responsive_io


def test_completion_and_errors_do_not_depend_on_progress_timer_or_console(tmp_path):
    # An old implementation hangs here despite completed I/O. Keep the
    # regression in a child so even a broken event loop has a bounded timeout.
    script = tmp_path / "completion.py"
    marker = tmp_path / "result.txt"
    script.write_text('''
import sys
from pathlib import Path
from threading import Event
from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog, QWidget
from fdm.ui import responsive_io
import fdm.runtime_logging as runtime_logging

sys.stdout = sys.stderr = None
runtime_logging.runtime_log_path = lambda: Path(sys.argv[1]).with_suffix(".log")
app = QApplication([])
parent = QWidget()
parent.show()
class Gate(QObject):
    entered = Signal()
class DelayedTimer(QTimer):
    def start(self, *args):
        pass
responsive_io.QTimer = DelayedTimer

def launch():
    try:
        for index in range(6):
            ready = Event()
            cancelled = Event()
            gate = Gate(parent)
            gate.entered.connect(ready.set, Qt.ConnectionType.QueuedConnection)
            def work(progress):
                assert ready.wait(2), "GUI did not dispatch the entry event"
                progress(1, 1)
                if index % 2:
                    raise OSError("expected failure")
                return index
            gate.entered.emit()
            try:
                value = responsive_io.run_responsive_io(
                    parent, title="test", label="local result", operation=work,
                    cancellation_event=cancelled,
                )
            except OSError as exc:
                assert index % 2 and str(exc) == "expected failure"
            else:
                assert not index % 2 and value == index
            assert not cancelled.is_set(), "finishing an operation must not cancel it"
            gate.deleteLater()
        # A later completion dialog must not revive already completed loaders
        # while their deferred deletion waits for the outer event loop.
        completion = QMessageBox(parent)
        completion.setText("capture complete")
        visible_loaders = []
        def close_completion():
            visible_loaders.extend(dialog.labelText() for dialog in parent.findChildren(QProgressDialog) if dialog.isVisible())
            completion.accept()
        QTimer.singleShot(400, completion, close_completion)
        completion.exec()
        assert not visible_loaders, visible_loaders
        Path(sys.argv[1]).write_text("completed", encoding="utf-8")
    except BaseException:
        import traceback
        Path(sys.argv[1]).write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        app.quit()

QTimer.singleShot(0, parent, launch)
app.exec()
''', encoding="utf-8")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env["PYTHONPATH"] = str(Path(responsive_io.__file__).resolve().parents[2])
    subprocess.run(
        [sys.executable, str(script), str(marker)],
        env=env, capture_output=True, text=True, timeout=8, check=True,
    )
    assert marker.read_text(encoding="utf-8") == "completed"


def test_visible_progress_update_does_not_dispatch_nested_gui_callbacks(monkeypatch):
    parent = QWidget()
    parent.show()
    started = Event()
    observed = Event()
    nested = []

    class Observer(QObject):
        notified = Signal()

    observer = Observer(parent)

    def record():
        stack = inspect.stack(context=0)
        nested.append(any(frame.function == "poll" and frame.filename == responsive_io.__file__ for frame in stack))
        observed.set()

    observer.notified.connect(record, Qt.ConnectionType.QueuedConnection)
    original_dialog = responsive_io._OwnedProgressDialog

    class VisibleDialog(original_dialog):
        def setMinimumDuration(self, _duration):
            super().setMinimumDuration(0)

    monkeypatch.setattr(responsive_io, "_OwnedProgressDialog", VisibleDialog)
    setup = QTimer(parent)

    def connect_bar():
        for dialog in parent.findChildren(VisibleDialog):
            if not dialog.isVisible():
                continue
            bar = dialog.findChild(QProgressBar)
            bar.valueChanged.connect(lambda value: observer.notified.emit() if value > 0 else None)
            setup.stop()
            started.set()

    setup.timeout.connect(connect_bar)
    setup.start(1)

    def work(progress):
        assert started.wait(2), "progress window was not shown"
        progress(50, 100)
        assert observed.wait(2), "queued callback never arrived"
        return 42

    try:
        assert responsive_io.run_responsive_io(parent, title="test", label="progress", operation=work) == 42
        assert nested == [False]
    finally:
        setup.stop()
        parent.close()
        parent.deleteLater()
