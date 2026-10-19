"""Capture real browse/settings/export widgets through live theme changes.

All profiles, slide data, render caches and images stay under --output. The
system-light/system-dark cases simulate the resolved OS palette; they are not
physical monitor or native Windows acceptance evidence.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", default="windows" if sys.platform == "win32" else "offscreen")
    args = parser.parse_args()
    os.environ["QT_QPA_PLATFORM"] = args.platform
    sys.path.insert(0, str(ROOT / "src"))
    from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QColor, QImage, QPalette, QWheelEvent
    from PySide6.QtWidgets import QApplication
    from fdm import runtime_logging, settings
    from fdm.models import ImageDocument
    from fdm.services.digital_slide_store import DigitalSlideManifest, DigitalSlideStore, DigitalSlideTile
    from fdm.services.export_service import ExportSelection
    from fdm.settings import AppSettings
    from fdm.ui import digital_slide_canvas, theme
    from fdm.ui.dialogs import ExportOptionsDialog, SettingsDialog
    from fdm.version import __version__

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    report = {"version": __version__, "platform": args.platform, "system_cases_simulated": True,
              "screenshots": [], "cases": []}

    def settle(seconds=.12):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.processEvents()
            # Release the GIL for the real Python tile renderer.
            time.sleep(.01)

    def capture(name, widget):
        settle()
        path = output / f"{name}.png"
        assert widget.grab().save(str(path))
        report["screenshots"].append({"name": name, "file": path.name,
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    with ExitStack() as stack:
        stack.enter_context(patch.object(settings, "settings_file_path", lambda: output / "settings.json"))
        stack.enter_context(patch.object(runtime_logging, "runtime_log_path", lambda: output / "runtime.log"))
        stack.enter_context(patch.object(digital_slide_canvas, "digital_slide_render_cache_directory", lambda: output / "cache"))
        source = output / "theme-source.fdmslide"
        if source.exists():
            parser.error("--output already contains theme-source.fdmslide; use a fresh output directory")
        manifest = DigitalSlideManifest(1, 800, 640, 200, 160, [0])
        store = DigitalSlideStore.create(source, manifest)
        for row in range(4):
            for column in range(4):
                image = QImage(200, 160, QImage.Format.Format_RGB32)
                image.fill(QColor(("#FFFFFF", "#000000", "#B6AC97", "#547A95")[(row + column) % 4]))
                store.write_tile(DigitalSlideTile(0, column * 200, row * 160, 200, 160), image,
                                 update_manifest=False)
        store.write_manifest(manifest)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        canvas = digital_slide_canvas.DigitalSlideCanvas()
        canvas.resize(1200, 760)
        canvas.set_settings(AppSettings(digital_slide_render_cache_gib=0))
        canvas.set_slide_document(ImageDocument(id="theme-audit", path=str(source), image_size=(800, 640),
                                                document_kind="digital_slide"), store)
        preferences = SettingsDialog(AppSettings(), document=None)
        export = ExportOptionsDialog(ExportSelection.all_enabled(), allow_all_scope=True)
        try:
            canvas.show()
            settle(.5)
            center = QPointF(canvas.rect().center())
            for _ in range(12):
                app.sendEvent(canvas, QWheelEvent(center, center, QPoint(), QPoint(0, -120),
                    Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.NoScrollPhase, False))
            settle(.5)
            assert canvas.large_area_browse_active()
            for widget in (preferences, export):
                widget.show()
                widget.resize(900, 640)
            cases = [(mode, mode) for mode in ("light", "warm", "dark", "navy")]
            cases += [("system", "light"), ("system", "dark")]
            for mode, resolved in cases:
                name = resolved if mode != "system" else f"system-{resolved}"
                with patch.object(theme, "_system_color_mode", return_value=resolved):
                    theme.apply_application_theme(app, mode)
                    settle()
                capture(f"{name}-slide-browse", canvas)
                preferences._theme_mode_combo.setCurrentIndex(preferences._theme_mode_combo.findData(mode))
                preferences._settings_navigation.setCurrentRow(0)
                capture(f"{name}-settings", preferences)
                preferences._settings_navigation.setCurrentRow(1)
                capture(f"{name}-measurement-settings", preferences)
                export._export_navigation.setCurrentRow(0)
                capture(f"{name}-export-data", export)
                export._export_navigation.setCurrentRow(1)
                capture(f"{name}-export-image", export)
                expected = theme.build_theme_palette(resolved)
                case = {"mode": mode, "resolved": resolved, "description_colors": [], "selected_colors": []}
                for navigation, description in ((preferences._settings_navigation, preferences._settings_page_description),
                                                (export._export_navigation, export._export_page_description)):
                    rect = navigation.visualItemRect(navigation.currentItem())
                    background = navigation.viewport().grab().toImage().pixelColor(rect.right() - 12, rect.center().y())
                    assert background == expected.color(QPalette.ColorRole.Highlight)
                    foreground = description.palette().color(QPalette.ColorRole.WindowText)
                    assert foreground == QColor(theme.semantic_colors(expected).muted)
                    case["description_colors"].append(foreground.name())
                    case["selected_colors"].append(background.name())
                report["cases"].append(case)
            report["source_image_unchanged"] = hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
            assert report["source_image_unchanged"]
            report["passed"] = True
        finally:
            canvas.shutdown()
            canvas.clear_document()
            store.close()
            for widget in (canvas, preferences, export):
                widget.close()
                widget.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report.get("passed", False), "cases": len(report["cases"]),
                      "screenshots": len(report["screenshots"]), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
