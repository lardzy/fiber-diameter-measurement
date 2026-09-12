from contextvars import copy_context
from threading import BoundedSemaphore, Event, Thread, get_ident

import pytest

from fdm import operation_diagnostics as diagnostics


def test_slow_operation_records_worker_phase_without_gui_dispatch(monkeypatch):
    records = []
    release = Event()
    captured = []
    monkeypatch.setattr(diagnostics, "_SLOW_AT_SECONDS", (0.01, 0.02, 0.03))

    def append(title, details):
        records.append((get_ident(), title, details))
        if title == "Slow digital slide operation":
            release.set()

    monkeypatch.setattr(diagnostics, "append_runtime_log", append)

    @diagnostics.diagnose_operation("test.slow")
    def operation():
        captured.append(diagnostics._CURRENT.get())
        context = copy_context()
        worker = Thread(target=lambda: context.run(diagnostics.operation_phase, "local.stat in worker"))
        worker.start()
        worker.join(1)
        assert not worker.is_alive()
        # No QTimer, event loop or console is available to wake the observer.
        assert release.wait(1)
        return 42

    assert operation() == 42
    assert diagnostics._CURRENT.get() is None
    captured[0].observer.join(1)
    assert not captured[0].observer.is_alive()
    assert all(ident != get_ident() for ident, _, _ in records)
    slow = [details for _, title, details in records if title == "Slow digital slide operation"]
    assert 1 <= len(slow) <= 3
    assert "local.stat in worker" in slow[0]
    assert "test_operation_diagnostics.py" in slow[0]
    assert "outcome=completed" in records[-1][2]


def test_diagnostics_preserve_exception_and_restore_nested_context(monkeypatch):
    records = []
    traces = []
    monkeypatch.setattr(diagnostics, "append_runtime_log", lambda title, details: records.append((title, details)))
    expected = OSError("expected local read failure")

    @diagnostics.diagnose_operation("test.inner")
    def inner():
        traces.append(diagnostics._CURRENT.get())
        raise expected

    @diagnostics.diagnose_operation("test.outer")
    def outer():
        trace = diagnostics._CURRENT.get()
        traces.append(trace)
        with pytest.raises(OSError) as exc:
            inner()
        assert exc.value is expected
        assert diagnostics._CURRENT.get() is trace

    outer()
    for trace in traces:
        trace.observer.join(1)
        assert not trace.observer.is_alive()
    assert traces[1].parent_id == traces[0].id
    assert diagnostics._CURRENT.get() is None
    assert any("OSError: expected local read failure" in details for _, details in records)


def test_blocked_log_writer_cannot_accumulate_unbounded_observers(monkeypatch):
    entered = Event()
    release = Event()
    monkeypatch.setattr(diagnostics, "_OBSERVER_SLOTS", BoundedSemaphore(1))

    def blocked_log(*args):
        entered.set()
        assert release.wait(1)

    monkeypatch.setattr(diagnostics, "append_runtime_log", blocked_log)
    first = diagnostics._OperationTrace("first")
    first.start()
    try:
        assert entered.wait(1)
        for index in range(100):
            first.phase(str(index))
            next_trace = diagnostics._OperationTrace("extra")
            next_trace.start()
            assert next_trace.observer is None
            next_trace.finish(None)
        assert len(first.phases) == 16
    finally:
        first.finish(None)
        release.set()
        first.observer.join(1)
    assert not first.observer.is_alive()
    assert diagnostics._OBSERVER_SLOTS.acquire(blocking=False)
    diagnostics._OBSERVER_SLOTS.release()
