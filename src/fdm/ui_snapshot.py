from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QPoint, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fdm.geometry import Line, Point  # noqa: E402
from fdm.models import (  # noqa: E402
    Calibration,
    ImageDocument,
    Measurement,
    OverlayAnnotation,
    OverlayAnnotationKind,
    OverlayTextAnchorAlignment,
    OverlayTextLayoutSpec,
    OverlayTextSizeSpace,
    new_id,
)
from fdm.settings import AppSettings  # noqa: E402
from fdm.ui.dialogs import SettingsDialog  # noqa: E402
from fdm.ui.image_loader import ImageLoadRequest  # noqa: E402
from fdm.ui.main_window import MainWindow  # noqa: E402
from fdm.ui.theme import apply_application_theme  # noqa: E402


UI_SNAPSHOT_SCENARIOS = (
    "empty",
    "measurement",
    "measurement-fullscreen",
    "measurement-zoomed",
    "measurement-object",
    "overlay-text-object",
    "measurement-calibration-collapsed",
    "measurement-records-collapsed",
    "measurement-results",
    "acquisition",
    "digital-slide",
    "settings",
)


def _settle_ui(milliseconds: int = 800) -> None:
    """Let deferred layout, timer and paint work finish before a review grab."""

    loop = QEventLoop()
    QTimer.singleShot(max(0, int(milliseconds)), loop.quit)
    loop.exec()


def _render_widget(widget, output: Path) -> None:
    """Synchronously render the full widget tree into a deterministic image."""

    device_ratio = widget.devicePixelRatioF()
    target = QImage(
        max(1, int(round(widget.width() * device_ratio))),
        max(1, int(round(widget.height() * device_ratio))),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    target.setDevicePixelRatio(device_ratio)
    target.fill(widget.palette().window().color())
    painter = QPainter(target)
    try:
        widget.render(painter, QPoint(0, 0))
    finally:
        painter.end()
    if not target.save(str(output), "PNG"):
        raise RuntimeError(f"Unable to save UI snapshot: {output}")


def _demo_document() -> tuple[ImageDocument, QImage]:
    image = QImage(1280, 820, QImage.Format.Format_RGB32)
    painter = QPainter(image)
    gradient = QLinearGradient(0, 0, image.width(), image.height())
    gradient.setColorAt(0.0, QColor("#E7E1D5"))
    gradient.setColorAt(0.55, QColor("#C9D7D1"))
    gradient.setColorAt(1.0, QColor("#9FAEAA"))
    painter.fillRect(image.rect(), gradient)
    painter.setPen(QColor(90, 103, 103, 70))
    for offset in range(-300, 1500, 95):
        painter.drawLine(offset, 0, offset + 500, image.height())
    painter.end()

    document = ImageDocument(
        id=new_id("image"),
        path="/tmp/ui-review-demo.png",
        image_size=(image.width(), image.height()),
        calibration=Calibration(
            mode="preset",
            pixels_per_unit=4.0,
            unit="um",
            source_label="20x 标定",
        ),
    )
    document.initialize_runtime_state()
    cotton = document.create_group(color="#2A9D8F", label="棉")
    viscose = document.create_group(color="#D79B45", label="粘纤")
    document.set_active_group(cotton.id)
    values = [72, 80, 88, 96, 105, 116, 126]
    for index, length in enumerate(values):
        measurement = Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=cotton.id if index < 5 else viscose.id,
            mode="manual" if index % 3 else "snap",
            measurement_kind="line",
            line_px=Line(
                Point(155 + index * 102, 190 + (index % 2) * 155),
                Point(155 + index * 102 + length, 190 + (index % 2) * 155 + 18),
            ),
            confidence=1.0 if index % 3 else 0.92,
            status="manual" if index % 3 else "snapped",
        )
        document.add_measurement(measurement)
    document.add_measurement(
        Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=cotton.id,
            mode="continuous_manual",
            measurement_kind="polyline",
            polyline_px=[Point(230, 570), Point(360, 520), Point(500, 590)],
            status="continuous_manual",
        )
    )
    document.add_measurement(
        Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=viscose.id,
            mode="polygon_area",
            measurement_kind="area",
            polygon_px=[Point(760, 470), Point(1020, 450), Point(1080, 650), Point(820, 680)],
            status="manual",
        )
    )
    document.add_measurement(
        Measurement(
            id=new_id("meas"),
            image_id=document.id,
            fiber_group_id=cotton.id,
            mode="count",
            measurement_kind="count",
            point_px=Point(650, 265),
            status="count",
        )
    )
    document.recalculate_measurements()
    # Keep the review scene focused on a representative length result so the
    # live panel demonstrates a meaningful multi-sample summary.
    document.view_state.selected_measurement_id = document.measurements[4].id
    return document, image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic UI review screenshots.")
    parser.add_argument(
        "--scenario",
        choices=UI_SNAPSHOT_SCENARIOS,
        default="measurement",
    )
    parser.add_argument("--theme", choices=("dark", "light", "system"), default="dark")
    parser.add_argument(
        "--tool-mode",
        choices=(
            "select",
            "manual",
            "continuous_manual",
            "snap",
            "polygon_area",
            "freehand_area",
            "count",
            "magic_segment",
            "reference_propagation",
            "fiber_quick",
            "calibration",
            "overlay",
        ),
        default="select",
    )
    parser.add_argument(
        "--settings-page",
        choices=("general", "measurement", "annotation", "analysis", "area", "acquisition", "export"),
        default="general",
    )
    parser.add_argument(
        "--results-tab",
        choices=("records", "statistics", "distribution"),
        default="records",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--scale", type=float, default=1.0, help="Qt UI scale factor, for example 1.25 or 1.5")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _apply_measurement_zoomed_scene(window: MainWindow) -> bool:
    """Apply the deterministic zoom used to review canvas navigation chrome."""

    canvas = window.current_canvas()
    if canvas is None:
        return False
    canvas.set_view_zoom(2.4)
    canvas.center_on_image_point(Point(760.0, 430.0))
    return True


def _apply_measurement_fullscreen_scene(window: MainWindow) -> bool:
    """Enter the production full-screen path used by the review scene."""

    window._toggle_fullscreen_measurement(True)
    controller = window._fullscreen_controller
    return bool(controller is not None and controller.is_active)


def main() -> int:
    args = _parse_args()
    if not 0.75 <= args.scale <= 3.0:
        raise SystemExit("--scale must be between 0.75 and 3.0")
    os.environ["QT_SCALE_FACTOR"] = f"{args.scale:g}"
    app = QApplication.instance() or QApplication([])
    settings = AppSettings(theme_mode=args.theme)
    apply_application_theme(app, args.theme)
    document, image = _demo_document()
    if args.scenario == "overlay-text-object":
        text_overlay = OverlayAnnotation(
            id=new_id("overlay"),
            image_id=document.id,
            kind=OverlayAnnotationKind.TEXT,
            content="高分辨率样品\n中心锚点",
            anchor_px=Point(640.0, 390.0),
            text_layout=OverlayTextLayoutSpec(
                anchor_alignment=OverlayTextAnchorAlignment.CENTER,
                size_space=OverlayTextSizeSpace.IMAGE_PX,
                image_font_size_px=72.0,
            ),
        )
        document.add_overlay_annotation(text_overlay)
        document.select_overlay_annotation(text_overlay.id)
    if args.scenario == "settings":
        scenario_name = f"settings-{args.settings_page}"
    elif args.scenario == "measurement-results":
        scenario_name = f"measurement-results-{args.results_tab}"
    else:
        scenario_name = args.scenario
    output = args.output or (
        Path("build/ui-review")
        / f"{scenario_name}-{args.theme}-{args.width}x{args.height}@{args.scale:g}x.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with (
        patch("fdm.ui.main_window.AppSettingsIO.load", return_value=settings),
        patch("fdm.ui.main_window.AppSettingsIO.save", return_value=None),
    ):
        if args.scenario == "settings":
            widget = SettingsDialog(settings, document=document)
            page_order = ("general", "measurement", "annotation", "analysis", "area", "acquisition", "export")
            widget._settings_navigation.setCurrentRow(page_order.index(args.settings_page))
        else:
            widget = MainWindow()
            if args.scenario != "empty":
                widget._add_loaded_document(
                    ImageLoadRequest(path=document.path, document=document),
                    image,
                )
        widget.resize(max(640, args.width), max(480, args.height))
        widget.show()
        for _ in range(5):
            app.processEvents()
        if isinstance(widget, SettingsDialog):
            # Offscreen Qt exposes a synthetic screen unrelated to the review
            # matrix.  Re-apply the requested deterministic capture size after
            # the dialog has exercised its real show-time clamping logic.
            widget.resize(max(640, args.width), max(480, args.height))
            app.processEvents()
        if isinstance(widget, MainWindow) and args.scenario == "measurement-results":
            widget._toggle_results_panel()
            widget._results_tabs.setCurrentIndex(
                {"records": 0, "statistics": 1, "distribution": 2}[args.results_tab]
            )
        elif isinstance(widget, MainWindow) and args.scenario == "measurement-object":
            widget._object_properties_section.setExpanded(True)
            widget._refresh_object_inspector()
        elif isinstance(widget, MainWindow) and args.scenario == "overlay-text-object":
            widget._calibration_section.setExpanded(False)
            widget._records_section.setExpanded(False)
            widget._area_recognition_section.setExpanded(False)
            widget._object_properties_section.setExpanded(True)
            widget._refresh_object_inspector()
            widget._inspector_scroll.ensureWidgetVisible(
                widget._object_properties_section,
                0,
                0,
            )
        elif isinstance(widget, MainWindow) and args.scenario == "measurement-calibration-collapsed":
            widget._calibration_section.setExpanded(False)
        elif isinstance(widget, MainWindow) and args.scenario == "measurement-records-collapsed":
            widget._records_section.setExpanded(False)
        elif isinstance(widget, MainWindow) and args.scenario == "measurement-fullscreen":
            _apply_measurement_fullscreen_scene(widget)
        elif isinstance(widget, MainWindow) and args.scenario == "measurement-zoomed":
            _apply_measurement_zoomed_scene(widget)
        elif isinstance(widget, MainWindow) and args.scenario in {"acquisition", "digital-slide"}:
            widget._preview_active = True
            widget._digital_slide_mode = args.scenario == "digital-slide"
            widget._center_stack.setCurrentWidget(widget._preview_page)
            widget._on_live_preview_frame_ready(image)
            widget._sync_digital_slide_mode_ui()
            widget._update_preview_analysis_controls()
        if isinstance(widget, MainWindow):
            if args.scenario not in {"acquisition", "digital-slide"}:
                widget.set_tool_mode(args.tool_mode)
            # The production UI uses a short debounce.  Snapshot scenes force
            # one deterministic refresh so captures never depend on wall time.
            widget._refresh_statistics_ui()
        for _ in range(3):
            app.processEvents()
        if isinstance(widget, MainWindow) and args.scenario == "measurement-fullscreen":
            # Recreate the real entry hint after the window-state transition,
            # then freeze its production fade timer for a deterministic review
            # capture.  The application itself still fades the hint normally.
            widget._show_fullscreen_hint()
            widget._fullscreen_hint_timer.stop()
        widget.ensurePolished()
        widget.update()
        _settle_ui()
        widget.repaint()
        app.processEvents()
        _render_widget(widget, output)

        payload = {
            "scenario": args.scenario,
            "theme": args.theme,
            "tool_mode": args.tool_mode if args.scenario != "settings" else None,
            "settings_page": args.settings_page if args.scenario == "settings" else None,
            "results_tab": args.results_tab if args.scenario == "measurement-results" else None,
            "scale": args.scale,
            "path": str(output.resolve()),
            "window": [widget.width(), widget.height()],
            "device_pixel_ratio": widget.devicePixelRatioF(),
        }
        if isinstance(widget, MainWindow):
            center = widget.centralWidget()
            payload.update(
                {
                    "central": [center.width(), center.height()] if center is not None else None,
                    "project_visible": bool(widget._project_dock and widget._project_dock.isVisible()),
                    "inspector_visible": bool(widget._inspector_dock and widget._inspector_dock.isVisible()),
                    "results_visible": bool(widget._results_dock and widget._results_dock.isVisible()),
                    "compact": bool(widget._adaptive_layout and widget._adaptive_layout.is_compact),
                    "fullscreen_active": bool(
                        widget._fullscreen_controller is not None
                        and widget._fullscreen_controller.is_active
                    ),
                }
            )
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        widget.close()
        app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
