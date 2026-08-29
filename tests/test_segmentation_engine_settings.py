from __future__ import annotations

from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fdm.services.segmentation_engines import EngineDiagnosticResult
from fdm.settings import OfflineSegmentationEnginePack
from fdm.ui.segmentation_engine_settings import OfflineSegmentationEngineDialog


def test_cpu_diagnostic_runs_without_blocking_preferences_ui(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    release = Event()

    class FakeService:
        def inspect(self, _path, *, managed=None):
            return SimpleNamespace(resource_count=1)

        def diagnose(self, _record):
            release.wait(timeout=2.0)
            return EngineDiagnosticResult(
                ok=True,
                message="CPU 诊断通过。",
                details={"device": "cpu"},
            )

    record = OfflineSegmentationEnginePack(
        engine_id="sam3",
        display_name="SAM3",
        version="1",
        path=str(tmp_path / "pack"),
        manifest_sha256="a" * 64,
    )
    dialog = OfflineSegmentationEngineDialog(
        [record],
        service=FakeService(),  # type: ignore[arg-type]
    )
    dialog._table.selectRow(0)  # noqa: SLF001

    started = monotonic()
    dialog._diagnose_selected()  # noqa: SLF001
    elapsed = monotonic() - started

    assert elapsed < 0.2
    assert dialog._diagnostic_task is not None  # noqa: SLF001
    assert "后台运行" in dialog._details.toPlainText()  # noqa: SLF001
    release.set()
    deadline = monotonic() + 2.0
    while dialog._diagnostic_task is not None and monotonic() < deadline:  # noqa: SLF001
        app.processEvents()
        QTest.qWait(10)
    assert dialog._diagnostic_task is None  # noqa: SLF001
    assert "CPU 诊断通过" in dialog._details.toPlainText()  # noqa: SLF001
    dialog.close()
