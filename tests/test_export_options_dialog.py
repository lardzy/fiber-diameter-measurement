from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialogButtonBox,
    QSpinBox,
)
import pytest

from fdm.services.export_service import (
    ExportImageRenderMode,
    ExportScope,
    ExportSelection,
)
from fdm.services.raster_export import (
    RasterEncodingOptions,
    RasterExportFormat,
    TiffCompression,
)
from fdm.settings import RawRecordTemplate
from fdm.ui.dialogs import (
    ExportOptionsDialog,
    NoWheelComboBox,
    NoWheelSpinBox,
)
from fdm.ui.theme import apply_application_theme


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _FakeWheelEvent:
    def __init__(self, angle_y: int = -120) -> None:
        self._angle_y = angle_y
        self.accepted = False
        self.ignored = False

    def pixelDelta(self) -> QPoint:
        return QPoint()

    def angleDelta(self) -> QPoint:
        return QPoint(0, self._angle_y)

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True


def _show_dialog(
    app: QApplication,
    selection: ExportSelection,
    *,
    raw_record_templates: list[RawRecordTemplate] | None = None,
    width: int = 1093,
    height: int = 576,
) -> ExportOptionsDialog:
    dialog = ExportOptionsDialog(
        selection,
        allow_all_scope=True,
        raw_record_templates=raw_record_templates,
    )
    dialog.show()
    app.processEvents()
    dialog.resize(width, height)
    app.processEvents()
    return dialog


def test_frequent_template_export_is_the_default_page(
    app: QApplication,
) -> None:
    template = RawRecordTemplate(
        name="激光共聚焦原始记录",
        path="runtime/content-templates/confocal.xlsx",
    )
    dialog = _show_dialog(
        app,
        ExportSelection(
            include_excel=True,
            raw_record_template_path=template.path,
        ),
        raw_record_templates=[template],
    )
    try:
        assert dialog._export_pages.currentIndex() == 0
        assert dialog._raw_record_group.isVisible()
        assert dialog._raw_record_template_combo.currentData() == template.path
        assert "原始记录模板" in dialog._export_summary_label.text()
    finally:
        dialog.close()
        app.processEvents()


def test_image_only_selection_opens_image_page(
    app: QApplication,
) -> None:
    dialog = _show_dialog(
        app,
        ExportSelection(include_measurement_overlay=True),
    )
    try:
        assert dialog._export_pages.currentIndex() == 1
        assert dialog._overlay_group.isVisible()
    finally:
        dialog.close()
        app.processEvents()


def test_long_template_name_does_not_force_scrollbars(
    app: QApplication,
) -> None:
    template = RawRecordTemplate(
        name=(
            "研发中心高倍率激光共聚焦纤维与孔洞联合测量"
            "原始记录模板名称用于验证不会撑宽导出窗口"
        ),
        path=(
            "runtime/content-templates/"
            "研发中心高倍率激光共聚焦纤维与孔洞联合测量原始记录.xlsx"
        ),
    )
    dialog = _show_dialog(
        app,
        ExportSelection(
            include_excel=True,
            raw_record_template_path=template.path,
        ),
        raw_record_templates=[template],
    )
    try:
        page = dialog._export_page_scrolls[0]
        assert page.horizontalScrollBar().maximum() == 0
        assert page.verticalScrollBar().maximum() == 0
        assert dialog._export_summary_label.height() >= (
            dialog._export_summary_label.minimumSizeHint().height()
        )
        assert template.name in dialog._raw_record_template_combo.toolTip()
        assert template.path in dialog._export_summary_label.toolTip()
    finally:
        dialog.close()
        app.processEvents()


@pytest.mark.parametrize("theme", ("dark", "light"))
def test_image_page_fits_supported_small_workspace_without_scrolling(
    app: QApplication,
    theme: str,
) -> None:
    # 1093×576 可用逻辑区域扣除窗口边距后的实际首选对话框尺寸。
    apply_application_theme(app, theme)
    dialog = _show_dialog(
        app,
        ExportSelection.all_enabled(),
        width=900,
        height=544,
    )
    try:
        dialog._export_navigation.setCurrentRow(1)
        app.processEvents()
        page = dialog._export_page_scrolls[1]
        assert page.horizontalScrollBar().maximum() == 0
        for export_format in RasterExportFormat:
            dialog._image_format_combo.setCurrentIndex(
                dialog._image_format_combo.findData(export_format)
            )
            app.processEvents()
            assert page.verticalScrollBar().maximum() == 0
        dialog._webp_lossless_checkbox.setChecked(True)
        app.processEvents()
        assert page.verticalScrollBar().maximum() == 0
    finally:
        dialog.close()
        app.processEvents()


def test_all_dropdown_and_numeric_editors_are_wheel_protected(
    app: QApplication,
) -> None:
    dialog = _show_dialog(
        app,
        ExportSelection.all_enabled(),
        width=640,
        height=460,
    )
    try:
        combos = dialog.findChildren(QComboBox)
        spins = dialog.findChildren(QSpinBox)
        assert combos
        assert spins
        assert all(isinstance(combo, NoWheelComboBox) for combo in combos)
        assert all(isinstance(spin, NoWheelSpinBox) for spin in spins)

        dialog._export_navigation.setCurrentRow(1)
        app.processEvents()
        page = dialog._export_page_scrolls[1]
        assert page.property("redirectEditorWheel") is True
        assert page.verticalScrollBar().maximum() > 0
        page.verticalScrollBar().setValue(0)

        before = dialog._png_compression_spin.value()
        event = _FakeWheelEvent()
        dialog._png_compression_spin.wheelEvent(event)

        assert dialog._png_compression_spin.value() == before
        assert page.verticalScrollBar().value() > 0
        assert event.accepted
    finally:
        dialog.close()
        app.processEvents()


@pytest.mark.parametrize(
    (
        "export_format",
        "quality",
        "progressive",
        "png",
        "tiff",
        "webp_lossless",
        "webp_method",
        "background",
    ),
    (
        (RasterExportFormat.PNG, False, False, True, False, False, False, False),
        (RasterExportFormat.JPEG, True, True, False, False, False, False, True),
        (RasterExportFormat.TIFF, False, False, False, True, False, False, False),
        (RasterExportFormat.BMP, False, False, False, False, False, False, True),
        (RasterExportFormat.WEBP, True, False, False, False, True, True, False),
    ),
)
def test_image_format_only_shows_relevant_parameters(
    app: QApplication,
    export_format: RasterExportFormat,
    quality: bool,
    progressive: bool,
    png: bool,
    tiff: bool,
    webp_lossless: bool,
    webp_method: bool,
    background: bool,
) -> None:
    dialog = _show_dialog(
        app,
        ExportSelection(include_measurement_overlay=True),
    )
    try:
        dialog._image_format_combo.setCurrentIndex(
            dialog._image_format_combo.findData(export_format)
        )
        layout = dialog._image_format_layout
        assert layout.isRowVisible(dialog._image_quality_row) is quality
        assert dialog._jpeg_progressive_checkbox.isHidden() is (
            not progressive
        )
        assert layout.isRowVisible(dialog._png_compression_spin) is png
        assert layout.isRowVisible(dialog._tiff_compression_combo) is tiff
        assert dialog._webp_lossless_checkbox.isHidden() is (
            not webp_lossless
        )
        assert dialog._webp_method_spin.isHidden() is (not webp_method)
        assert (
            layout.isRowVisible(dialog._flatten_background_button)
            is background
        )
    finally:
        dialog.close()
        app.processEvents()


def test_webp_lossless_hides_quality_editor(
    app: QApplication,
) -> None:
    dialog = _show_dialog(
        app,
        ExportSelection(include_measurement_overlay=True),
    )
    try:
        dialog._image_format_combo.setCurrentIndex(
            dialog._image_format_combo.findData(RasterExportFormat.WEBP)
        )
        dialog._webp_lossless_checkbox.setChecked(True)
        assert not dialog._image_format_layout.isRowVisible(
            dialog._image_quality_row
        )
        assert not dialog._webp_lossless_checkbox.isHidden()
    finally:
        dialog.close()
        app.processEvents()


def test_continue_button_tracks_whether_any_output_is_selected(
    app: QApplication,
) -> None:
    dialog = _show_dialog(app, ExportSelection())
    try:
        continue_button = dialog._button_box.button(
            QDialogButtonBox.StandardButton.Ok
        )
        assert continue_button.text() == "保存"
        assert continue_button.objectName() == "exportOptionsSaveButton"
        assert not continue_button.isEnabled()
        dialog._csv.setChecked(True)
        assert continue_button.isEnabled()
    finally:
        dialog.close()
        app.processEvents()


def test_selection_round_trip_preserves_export_contract(
    app: QApplication,
) -> None:
    template = RawRecordTemplate(
        name="孔洞记录模板",
        path="runtime/content-templates/pores.xlsm",
    )
    selection = ExportSelection(
        include_measurement_overlay=True,
        include_scale_overlay=False,
        include_combined_overlay=True,
        include_scale_json=True,
        include_excel=True,
        include_csv=False,
        scope=ExportScope.ALL_OPEN,
        render_mode=ExportImageRenderMode.CURRENT_VIEWPORT,
        raw_record_template_path=template.path,
        image_encoding=RasterEncodingOptions(
            format=RasterExportFormat.JPEG,
            quality=87,
            jpeg_progressive=False,
            png_compression=4,
            tiff_compression=TiffCompression.LZW,
            webp_lossless=False,
            webp_method=5,
            jpeg_background=(32, 48, 64),
        ),
    )
    dialog = _show_dialog(
        app,
        selection,
        raw_record_templates=[template],
    )
    try:
        assert dialog.selection() == selection
    finally:
        dialog.close()
        app.processEvents()


def test_unregistered_selected_template_remains_available_for_validation(
    app: QApplication,
) -> None:
    missing_path = "runtime/content-templates/missing-template.xlsm"
    dialog = _show_dialog(
        app,
        ExportSelection(
            include_excel=True,
            raw_record_template_path=missing_path,
        ),
    )
    try:
        assert dialog._raw_record_template_combo.currentData() == missing_path
        assert "模板不可用" in dialog._raw_record_template_combo.currentText()
        assert dialog.selection().raw_record_template_path == missing_path
        assert "重新校验" in dialog._raw_record_template_hint.text()
    finally:
        dialog.close()
        app.processEvents()
