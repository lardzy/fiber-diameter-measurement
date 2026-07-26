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

from fdm.analysis_artifacts import (  # noqa: E402
    AnalysisArtifact,
    AnalysisCurve,
    AnalysisObjectKind,
    AnalysisObjectReference,
    AnalysisTable,
)
from fdm.geometry import Line, Point  # noqa: E402
from fdm.image_processing_models import (  # noqa: E402
    DisplayTransform,
    ImageProcessingRecipe,
)
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
from fdm.project_roi import (  # noqa: E402
    EllipseRoiGeometry,
    PolygonRoiGeometry,
    ProjectRoi,
    RectangleRoiGeometry,
    RoiPoint,
)
from fdm.services.export_service import ExportSelection  # noqa: E402
from fdm.services.image_batch import (  # noqa: E402
    BatchItemResourceEstimate,
    BatchResourceEstimate,
)
from fdm.services.raster_io import qimage_to_raster_plane  # noqa: E402
from fdm.settings import AppSettings  # noqa: E402
from fdm.ui.dialogs import ExportOptionsDialog, SettingsDialog  # noqa: E402
from fdm.ui.display_adjustment_dialog import DisplayAdjustmentDialog  # noqa: E402
from fdm.ui.image_loader import ImageLoadRequest  # noqa: E402
from fdm.ui.image_processing_workbench import (  # noqa: E402
    ImageProcessingWorkbench,
    default_operation_spec,
)
from fdm.ui.image_batch_dialog import (  # noqa: E402
    BatchDocumentOption,
    ImageBatchProcessingDialog,
)
from fdm.ui.main_window import MainWindow  # noqa: E402
from fdm.ui.analysis_results_center import AnalysisResultsCenter  # noqa: E402
from fdm.ui.advanced_analysis_dialog import (  # noqa: E402
    AdvancedAnalysisParametersDialog,
)
from fdm.ui.image_analysis_controller import AnalysisTool  # noqa: E402
from fdm.ui.raster_export_dialog import CurrentImageExportDialog  # noqa: E402
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
    "current-image-export",
    "measurement-export",
    "display-adjustment",
    "image-processing",
    "image-batch",
    "roi-workspace",
    "analysis-results",
    "advanced-analysis",
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


def _demo_analysis_artifacts(
    document: ImageDocument,
) -> tuple[AnalysisArtifact, ...]:
    measurement = next(
        item
        for item in document.measurements
        if item.measurement_kind == "area"
    )
    shape = AnalysisArtifact(
        id="analysis_review_shape",
        source_document_id=document.id,
        source_pixel_revision=0,
        source_reference=AnalysisObjectReference(
            kind=AnalysisObjectKind.MEASUREMENT,
            object_id=measurement.id,
            revision=measurement.geometry_revision,
        ),
        tool_id="fdm.shape",
        tool_version="1",
        parameters={"scope": "当前面积对象"},
        scalars={
            "net_area": 12634.25,
            "hole_area_px": 482.0,
            "hole_count": 2,
            "circularity": 0.83,
        },
        tables=(
            AnalysisTable(
                name="位置与边界",
                columns=("项目", "X", "Y", "宽", "高"),
                rows=(
                    ("质心", 913.4, 561.2, None, None),
                    ("边界框", 760.0, 450.0, 320.0, 230.0),
                ),
            ),
        ),
    )
    histogram = AnalysisArtifact(
        id="analysis_review_histogram",
        source_document_id=document.id,
        source_pixel_revision=0,
        tool_id="fdm.histogram",
        tool_version="1",
        parameters={"channel": "luminance", "bins": 16},
        scalars={
            "included_pixel_count": 98420,
            "non_finite_count": 0,
            "channel": "luminance",
        },
        curves=(
            AnalysisCurve(
                name="直方图",
                x=tuple(float(index * 16) for index in range(16)),
                y=(
                    320.0,
                    860.0,
                    1840.0,
                    4260.0,
                    7820.0,
                    10600.0,
                    13240.0,
                    14260.0,
                    13480.0,
                    10820.0,
                    7260.0,
                    4380.0,
                    2560.0,
                    1220.0,
                    540.0,
                    160.0,
                ),
                x_unit="强度",
                y_unit="频数",
            ),
        ),
    )
    return shape, histogram.mark_stale("ROI 几何已变化，建议重新计算")


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
        elif args.scenario == "current-image-export":
            widget = CurrentImageExportDialog(
                "/tmp/显微图像导出.png",
            )
        elif args.scenario == "measurement-export":
            widget = ExportOptionsDialog(
                ExportSelection.all_enabled(),
                allow_all_scope=True,
            )
        elif args.scenario == "display-adjustment":
            widget = DisplayAdjustmentDialog(
                qimage_to_raster_plane(image),
                DisplayTransform(
                    channel_ranges=(
                        (18.0, 232.0),
                        (20.0, 236.0),
                        (16.0, 228.0),
                    ),
                    gamma=1.15,
                ),
                source_name="激光共聚焦 RGB 示例",
            )
        elif args.scenario == "image-processing":
            widget = ImageProcessingWorkbench(
                qimage_to_raster_plane(image),
                source_document_id=document.id,
                source_name="显微图像处理示例",
                roi_summary="整张图片",
            )
            widget.set_operation_steps(
                (
                    default_operation_spec(
                        "gaussian_blur",
                        image.width(),
                        image.height(),
                        source_pixel_type=qimage_to_raster_plane(image).pixel_type,
                    ),
                )
            )
        elif args.scenario == "image-batch":
            raster = qimage_to_raster_plane(image)
            recipe = ImageProcessingRecipe.from_operations(
                (
                    default_operation_spec(
                        "gaussian_blur",
                        image.width(),
                        image.height(),
                        source_pixel_type=raster.pixel_type,
                    ),
                    default_operation_spec(
                        "clahe",
                        image.width(),
                        image.height(),
                        source_pixel_type=raster.pixel_type,
                    ),
                )
            )
            widget = ImageBatchProcessingDialog(
                recipe,
                (
                    BatchDocumentOption(
                        "batch-review-a",
                        "激光共聚焦样品 A",
                        "RGB8 · 1280×820",
                    ),
                    BatchDocumentOption(
                        "batch-review-b",
                        "激光共聚焦样品 B",
                        "RGB8 · 1920×1080",
                    ),
                    BatchDocumentOption(
                        "batch-review-c",
                        "批次 07 灰度样品",
                        "GRAY16 · 2048×1536",
                        selected=False,
                    ),
                    BatchDocumentOption(
                        "batch-review-slide",
                        "数字化切片（当前焦层）",
                        "数字化切片",
                        is_digital_slide=True,
                    ),
                ),
                recipe_name="纤维对比度增强",
            )
            widget.apply_preflight(
                BatchResourceEstimate(
                    items=(
                        BatchItemResourceEstimate(
                            document_id="batch-review-a",
                            source_bytes=4 << 20,
                            estimated_output_bytes=4 << 20,
                            estimated_peak_bytes=82 << 20,
                            allowed=True,
                        ),
                        BatchItemResourceEstimate(
                            document_id="batch-review-b",
                            source_bytes=7 << 20,
                            estimated_output_bytes=7 << 20,
                            estimated_peak_bytes=126 << 20,
                            allowed=True,
                        ),
                    ),
                    estimated_total_output_bytes=11 << 20,
                    available_disk_bytes=42 << 30,
                    reserve_disk_bytes=2 << 30,
                    disk_allowed=True,
                )
            )
        elif args.scenario == "analysis-results":
            artifacts = _demo_analysis_artifacts(document)
            widget = AnalysisResultsCenter(
                artifacts,
                document_names={document.id: "激光共聚焦示例图像"},
                measurement_names={
                    measurement.id: f"面积对象 #{index + 1}"
                    for index, measurement in enumerate(
                        document.measurements
                    )
                },
            )
        elif args.scenario == "advanced-analysis":
            widget = AdvancedAnalysisParametersDialog(
                AnalysisTool.DIRECTIONALITY,
                pixel_type=qimage_to_raster_plane(image).pixel_type,
                has_analysis_mask=True,
                active_group_label="玻璃纤维 · 批次 07",
            )
        else:
            widget = MainWindow()
            if args.scenario == "roi-workspace":
                widget.project.project_rois.extend(
                    (
                        ProjectRoi(
                            id="roi_review_rect",
                            document_id=document.id,
                            name="纤维密集区域",
                            geometry=RectangleRoiGeometry(
                                120.0,
                                140.0,
                                360.0,
                                250.0,
                            ),
                        ),
                        ProjectRoi(
                            id="roi_review_ellipse",
                            document_id=document.id,
                            name="孔洞复核区域",
                            geometry=EllipseRoiGeometry(
                                610.0,
                                260.0,
                                240.0,
                                190.0,
                            ),
                            color="#F4D35E",
                        ),
                        ProjectRoi(
                            id="roi_review_polygon",
                            document_id=document.id,
                            name="批次 07 自由区域",
                            geometry=PolygonRoiGeometry(
                                rings=(
                                    (
                                        RoiPoint(780.0, 470.0),
                                        RoiPoint(1080.0, 450.0),
                                        RoiPoint(1120.0, 700.0),
                                        RoiPoint(820.0, 690.0),
                                    ),
                                )
                            ),
                            color="#1C9ECB",
                        ),
                    )
                )
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
