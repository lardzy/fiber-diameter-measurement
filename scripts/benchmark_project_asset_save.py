"""Measure real device imports and MainWindow project saves in a temporary profile.

uv run --no-sync python scripts/benchmark_project_asset_save.py --output report.json
--source-root can select an archived src tree for a same-machine comparison.
No source microscope files or user settings are modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class Timings:
    active = None
    background = False
    app = None

    def instrument(self, owner, name, label, stack):
        original = getattr(owner, name)

        @wraps(original)
        def call(*args, **kwargs):
            start = time.perf_counter()
            result = original(*args, **kwargs)
            if self.active is not None:
                self.active[label + "_ms"] += (time.perf_counter() - start) * 1000
                self.active[label + "_calls"] += 1
                if label == "write_verified_asset":
                    self.active["temporary_encoded_bytes"] += result.bytes_written
            return result

        stack.enter_context(patch.object(owner, name, call))

    def save(self, window, path):
        self.active = defaultdict(float)
        start = time.perf_counter()
        try:
            if self.background:
                from PySide6.QtCore import QTimer
                pulses = [time.perf_counter()]
                timer = QTimer()
                timer.setInterval(5)
                timer.timeout.connect(lambda: pulses.append(time.perf_counter()))
                timer.start()
                assert window.request_save_project(str(path))
                while window.project_save_coordinator.busy:
                    self.app.processEvents()
                    if time.perf_counter() - start > 60:
                        raise TimeoutError("background save")
                    time.sleep(0.0005)
                timer.stop()
                pulses.append(time.perf_counter())
                assert window.project_save_coordinator.status.phase in {"saved", "saved_newer"}, window.project_save_coordinator.status
                self.active.update(window.project_save_coordinator.last_metrics)
                gaps = [(b-a)*1000 for a, b in zip(pulses, pulses[1:])]
                self.active["max_event_gap_ms"] = max(gaps)
            else:
                result = window.save_project(str(path))
                assert result.success, result
            elapsed = (time.perf_counter() - start) * 1000
            return dict(self.active, save_total_ms=elapsed)
        finally:
            self.active = None


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    os.environ.setdefault("QT_QPA_PLATFORM", "windows" if sys.platform == "win32" else "offscreen")
    sys.path.insert(0, str(args.source_root / "src"))
    import numpy as np
    from PySide6 import __version__ as qt_version
    from PySide6.QtWidgets import QApplication
    from fdm import runtime_logging, screenshot_settings, settings
    from fdm.models import project_assets_root
    from fdm.project_io import ProjectIO
    from fdm.raster import RasterPlane
    from fdm.services import raster_io
    from fdm.services.device_image_io import default_selection, inspect_source
    from fdm.ui import project_session_controller as persistence
    from fdm.ui.device_import_dialog import DeviceReadWorker
    from fdm.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    timings = Timings()
    timings.background, timings.app = args.background, app
    report = {
        "platform": platform.platform(), "python": platform.python_version(),
        "pyside6": qt_version, "label": args.label, "iterations": args.iterations,
        "scope": "Actual MainWindow background save including Qt dispatch" if args.background else "Actual MainWindow synchronous save, no on-screen repaint timing",
        "stage_times_are_nested": True, "cases": [],
    }
    with TemporaryDirectory(prefix="fdm-asset-save-bench-") as temporary, ExitStack() as stack:
        temporary = Path(temporary)
        stack.enter_context(patch.object(settings, "settings_file_path", lambda: temporary / "settings.json"))
        stack.enter_context(patch.object(screenshot_settings, "screenshot_settings_file_path", lambda: temporary / "screenshot.json"))
        stack.enter_context(patch.object(runtime_logging, "runtime_log_path", lambda: temporary / "startup.log"))
        window = MainWindow()
        window._confirm_close_documents = lambda _, **kwargs: True

        def warning(title, message):
            raise RuntimeError(f"{title}: {message}")

        window._show_project_warning = warning
        controller = window.project_session_controller
        for owner, name, label in (
            (controller, "_build_project_save_plan", "save_plan"),
            (persistence.ProjectSessionController, "_stage_project_assets", "asset_stage"),
            (controller, "_cleanup_unreferenced_revision_assets_payload", "asset_cleanup"),
            (persistence, "write_native_raster_asset", "write_verified_asset"),
            (raster_io, "_write_png", "png_encode"),
            (raster_io, "_write_tiff", "tiff_encode"),
            (raster_io, "read_raster_file", "verify_decode"),
            (RasterPlane, "sha256", "pixel_hash"),
            (persistence, "_file_sha256", "file_hash"),
            (ProjectIO, "save_payload", "json_publish"),
            (window, "_mark_project_saved", "mark_saved"),
            (window, "_update_ui_for_current_document", "ui_update"),
        ):
            timings.instrument(owner, name, label, stack)
        if hasattr(persistence, "copy_verified_raster_asset"):
            from fdm.services import raster_asset_reuse
            timings.instrument(persistence, "copy_verified_raster_asset", "verified_copy", stack)
            timings.instrument(raster_asset_reuse, "file_sha256", "reuse_file_hash", stack)

        cases = (
            ("dsx_stitched_color", args.samples_dir / "DSX1000/拼接/merge_1_0001.dsx", ("color",)),
            ("poir_stitched_intensity", args.samples_dir / "OLS5000/拼接/反面_002_G001.poir", ("intensity",)),
            ("poir_stitched_color", args.samples_dir / "OLS5000/拼接/反面_002_G001.poir", ("color",)),
        )
        try:
            for name, source, preferred in cases:
                window._reset_workspace()
                window._session_processed_root = temporary / name / "session"
                before = digest(source)
                started = time.perf_counter()
                channels = default_selection(inspect_source(source), preferred)
                worker = DeviceReadWorker("import", channels, asset_root=window._session_processed_root)
                errors = []
                worker.itemReady.connect(window._mount_device_channel)
                worker.finished.connect(lambda cancelled, failures: errors.extend(failures))
                worker.run()
                import_ms = (time.perf_counter() - started) * 1000
                assert not errors, errors
                assert len(window.project.documents) == len(channels) == 1
                app.processEvents()
                document = window.project.documents[0]
                pixel_digest = window._rasters[document.id].sha256()
                output = temporary / name / "project.fdmproj"
                first = timings.save(window, output)
                asset = project_assets_root(output) / document.path
                initial_stat = asset.stat()
                initial_digest = digest(asset)
                repeated = []
                for _ in range(args.iterations):
                    repeated.append(timings.save(window, output))
                    app.processEvents()
                unchanged = asset.stat().st_mtime_ns == initial_stat.st_mtime_ns and digest(asset) == initial_digest
                document.mark_session_dirty()
                document.metadata["benchmark_note"] = "metadata only"
                edited_save = timings.save(window, output)
                save_as_path = temporary / name / "另存为" / "项目.fdmproj"
                save_as = timings.save(window, save_as_path)
                copy = project_assets_root(save_as_path) / document.path
                assert digest(copy) == initial_digest
                window._reset_workspace()
                started = time.perf_counter()
                assert controller.load_project_from_path(save_as_path)
                while window.is_image_loading():
                    app.processEvents()
                    if time.perf_counter() - started > 60:
                        raise TimeoutError("project reopen")
                    time.sleep(.01)
                reopen_ms = (time.perf_counter() - started) * 1000
                reopened_save = timings.save(window, save_as_path)
                loaded = window.project.documents[0]
                assert window._rasters[loaded.id].sha256() == pixel_digest
                samples = [record["save_total_ms"] for record in repeated]
                case = {
                    "name": name, "source_bytes": source.stat().st_size,
                    "dimensions": list(document.image_size), "pixel_type": document.raster_pixel_type.value,
                    "import_ms": import_ms, "first_save": first, "repeat_runs": repeated,
                    "repeat_p50_ms": float(np.percentile(samples, 50)),
                    "repeat_p95_ms": float(np.percentile(samples, 95)),
                    "metadata_edit_save": edited_save, "save_as": save_as,
                    "reopen_ms": reopen_ms, "reopened_save": reopened_save,
                    "final_asset_unchanged_on_repeat": unchanged,
                    "source_unmodified": digest(source) == before,
                    "asset_bytes": initial_stat.st_size,
                }
                report["cases"].append(case)
                print(json.dumps({k: case[k] for k in ("name", "repeat_p50_ms", "repeat_p95_ms")}, allow_nan=False), flush=True)
        finally:
            window._reset_workspace()
            window.close()
            app.processEvents()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"REPORT: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--samples-dir", type=Path, default=ROOT / ".tmp/DSX1000、OLS5000样张")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--label", default="working-tree")
    parser.add_argument("--background", action="store_true", help="Measure the asynchronous UI save command and event gaps")
    arguments = parser.parse_args()
    if arguments.iterations < 1:
        parser.error("--iterations must be positive")
    run(arguments)
