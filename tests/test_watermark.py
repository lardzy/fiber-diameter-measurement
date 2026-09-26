from dataclasses import replace
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtCore import QDateTime, QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QDialog

from fdm.geometry import Point
from fdm.history import DocumentHistoryState
from fdm.models import ImageDocument, ProjectState, project_assets_root
from fdm.project_io import ProjectIO
from fdm.services.export_service import ExportImageRenderMode, ExportSelection, ExportService
from fdm.ui.canvas import DocumentCanvas
from fdm.ui.watermark_dialog import DATETIME_DISPLAY_FORMAT, WatermarkDialog
from fdm.ui.watermark_rendering import draw_watermark, import_logo, watermark_geometry, watermark_raster_cache
from fdm.watermark import WatermarkSpec
from test_canvas_progressive_overlay import scene as scene


@pytest.fixture(autouse=True)
def app():
    instance = QApplication.instance() or QApplication([])
    watermark_raster_cache.clear()
    yield instance


def document(spec=None):
    item = ImageDocument(id="watermark-image", path="sample.png", image_size=(320, 240), watermark=spec)
    item.initialize_runtime_state()
    return item


def logo_document(tmp_path, *, logo_size=(32, 16), **options):
    image = QImage(*logo_size, QImage.Format.Format_ARGB32)
    image.fill(QColor(255, 0, 0, 128))
    path = tmp_path / "标识.png"
    assert image.save(str(path))
    digest, data = import_logo(path)
    spec = WatermarkSpec(enabled=True, kind="logo", logo_sha256=digest, anchor="top_left", offset_x=0, offset_y=0, **options)
    item = document(spec)
    item.watermark_assets[digest] = data
    return item


def render(item, *, scale=1.0, dpr=1.0, origin=(0, 0), size=None, spec=None):
    logical_size = size or (round(item.image_size[0] * scale), round(item.image_size[1] * scale))
    target = QImage(round(logical_size[0] * dpr), round(logical_size[1] * dpr), QImage.Format.Format_ARGB32_Premultiplied)
    target.setDevicePixelRatio(dpr)
    target.fill(QColor("white"))
    painter = QPainter(target)
    try:
        kwargs = {"spec": spec} if spec is not None else {}
        draw_watermark(painter, item, lambda p: QPointF(origin[0] + p.x * scale, origin[1] + p.y * scale), strict=True, **kwargs)
    finally:
        painter.end()
    return target


@pytest.mark.parametrize("opacity, expected_green", [(0, 255), (0.25, 223), (1, 127)])
def test_logo_alpha_is_multiplied_once_and_clipped(tmp_path, opacity, expected_green):
    item = logo_document(tmp_path, opacity=opacity)
    image = render(item)
    color = image.pixelColor(10, 10)
    assert color.red() == 255 and abs(color.green() - expected_green) <= 1
    assert image.pixelColor(200, 200) == QColor("white")
    assert item.image_size == (320, 240)


@pytest.mark.parametrize("layout", ["single", "tile"])
@pytest.mark.parametrize("rotation", [0, -30, 90])
@pytest.mark.parametrize("scale,dpr", [(0.5, 1), (1, 1.5), (1.75, 2)])
def test_viewport_is_a_crop_of_full_image_at_same_scale(tmp_path, layout, rotation, scale, dpr):
    item = logo_document(tmp_path, layout=layout, rotation=rotation)
    full = render(item, scale=scale, dpr=dpr)
    viewport = render(item, scale=scale, dpr=dpr, origin=(-40, -20), size=(100, 80))
    expected = full.copy(round(40 * dpr), round(20 * dpr), round(100 * dpr), round(80 * dpr))
    # Qt's transformed texture sampler may round a boundary channel by one
    # when the clip origin changes. Any phase/position error is much larger.
    np.testing.assert_allclose(
        np.frombuffer(viewport.constBits(), np.uint8),
        np.frombuffer(expected.constBits(), np.uint8), atol=1, rtol=0,
    )


def test_panning_reuses_logo_decode_and_stamp_cache(tmp_path):
    item = logo_document(tmp_path, layout="tile")
    render(item)
    misses = watermark_raster_cache.misses
    for offset in range(30):
        render(item, origin=(-offset, -offset), size=(200, 150))
    assert watermark_raster_cache.logo_decodes == 1
    assert watermark_raster_cache.misses == misses
    assert watermark_raster_cache.bytes <= watermark_raster_cache.budget


def test_oversized_logo_reuses_small_stamp_without_decoding_each_frame(tmp_path, monkeypatch):
    item = logo_document(tmp_path, logo_size=(1024, 512), layout="tile")
    monkeypatch.setattr(watermark_raster_cache, "budget", 65536)
    render(item)
    assert watermark_raster_cache.logo_decodes == 1
    for offset in range(30):
        render(item, origin=(-offset, -offset), size=(200, 150))
    assert watermark_raster_cache.logo_decodes == 1
    assert watermark_raster_cache.bytes <= 65536


def test_save_reopen_save_as_and_old_history_asset_survive(tmp_path):
    item = logo_document(tmp_path)
    spec = item.watermark
    project = ProjectState(version="test", documents=[item])
    target = tmp_path / "first.fdmproj"
    ProjectIO.save(project, target)
    assert "image-watermark/v1" in json.loads(target.read_text())["required_features"]
    (tmp_path / "标识.png").unlink()
    loaded = ProjectIO.load(target)
    reopened = loaded.documents[0]
    assert reopened.watermark == spec
    assert render(reopened) == render(item)
    before = DocumentHistoryState.capture(reopened)
    reopened.watermark = WatermarkSpec(enabled=True, text="新的文字")
    second = tmp_path / "moved" / "second.fdmproj"
    ProjectIO.save(loaded, second)
    before.restore(reopened)
    ProjectIO.save(loaded, second)
    assert (project_assets_root(second) / spec.asset_path).exists()
    assert render(ProjectIO.load(second).documents[0]) == render(item)
    moved = tmp_path / "relocated" / "project.fdmproj"
    moved.parent.mkdir()
    shutil.move(second, moved)
    shutil.move(project_assets_root(second), project_assets_root(moved))
    assert render(ProjectIO.load(moved).documents[0]) == render(item)


def test_shared_logo_is_saved_once_and_unsupported_reader_cannot_overwrite(tmp_path, monkeypatch):
    from fdm import models
    from fdm.project_io import ProjectCompatibilityError

    first = logo_document(tmp_path)
    second = replace(first, id="second-image")
    target = tmp_path / "shared.fdmproj"
    ProjectIO.save(ProjectState(version="test", documents=[first, second]), target)
    assert len(list((project_assets_root(target) / "watermarks").glob("*.png"))) == 1
    monkeypatch.setattr(models, "SUPPORTED_PROJECT_REQUIRED_FEATURES", models.SUPPORTED_PROJECT_REQUIRED_FEATURES - {"image-watermark/v1"})
    loaded = ProjectIO.load(target)
    assert loaded.is_read_only_compatible
    before = target.read_bytes()
    with pytest.raises(ProjectCompatibilityError):
        ProjectIO.save(loaded, target)
    assert target.read_bytes() == before


def test_failed_project_commit_rolls_back_only_new_logo_assets(tmp_path):
    item = logo_document(tmp_path)
    project = ProjectState(version="test", documents=[item])
    target = tmp_path / "atomic.fdmproj"
    with patch("fdm.project_io.atomic_write_json", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            ProjectIO.save(project, target)
    assert not target.exists()
    assert not list(project_assets_root(target).rglob("*.png"))
    assert item.watermark.logo_sha256 in item.watermark_assets
    ProjectIO.save(project, target)
    old = target.read_bytes()
    with patch("fdm.project_io.atomic_write_json", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            ProjectIO.save(project, target)
    assert target.read_bytes() == old
    assert (project_assets_root(target) / item.watermark.asset_path).exists()


def test_missing_logo_preserves_config_but_prevents_export(tmp_path):
    item = logo_document(tmp_path)
    path = tmp_path / "missing.fdmproj"
    ProjectIO.save(ProjectState(version="test", documents=[item]), path)
    (project_assets_root(path) / item.watermark.asset_path).unlink()
    reopened = ProjectIO.load(path).documents[0]
    assert reopened.watermark == item.watermark
    assert reopened.watermark_asset_error
    with pytest.raises(ValueError, match="Logo"):
        render(reopened)


def test_legacy_project_and_digital_slide_are_unmodified():
    item = document()
    assert "watermark" not in item.to_dict()
    assert "image-watermark/v1" not in ProjectState(version="test", documents=[item]).effective_required_features()
    blank = render(item)
    item.document_kind = "digital_slide"
    item.watermark = WatermarkSpec(
        enabled=True, text="不得出现", layout="tile",
        include_datetime=True, datetime_text="2026-09-21 14:35:26",
    )
    assert render(item) == blank


def test_missing_logo_can_be_replaced_by_text_and_saved(tmp_path):
    item = logo_document(tmp_path)
    item.watermark_assets.clear()
    dialog = WatermarkDialog(item, render(document()))
    dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("text"))
    dialog.text_edit.setPlainText("改用文字")
    dialog._accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    item.watermark = dialog.watermark()
    assert item.watermark.logo_sha256 == ""
    target = tmp_path / "text.fdmproj"
    ProjectIO.save(ProjectState(version="test", documents=[item]), target)
    assert ProjectIO.load(target).documents[0].watermark.text == "改用文字"
    dialog.deleteLater()


def test_text_dialog_cancel_and_remove_do_not_mutate_source(app):
    item = document(WatermarkSpec(enabled=True, text="原水印"))
    image = render(document())
    before = item.snapshot_state()
    dialog = WatermarkDialog(item, image)
    assert dialog.font_combo.currentText() == "系统默认"
    assert dialog.watermark().font_family == ""
    dialog.text_edit.setPlainText("新水印\n第二行")
    dialog.layout_combo.setCurrentIndex(1)
    dialog._refresh_preview()
    assert dialog.watermark().layout == "tile"
    dialog.reject()
    assert item.snapshot_state() == before
    dialog._remove()
    assert dialog.watermark() is None and dialog.result() == QDialog.DialogCode.Accepted
    assert item.snapshot_state() == before
    dialog.deleteLater()


def test_canvas_underlay_contains_watermark_and_source_stays_identical(tmp_path):
    item = logo_document(tmp_path)
    source = render(document())
    original = bytes(source.constBits())
    canvas = DocumentCanvas()
    canvas.resize(320, 240)
    canvas.set_document(item, source)
    canvas._zoom = 1
    canvas._pan = Point(0, 0)
    target = QImage(source.size(), source.format())
    painter = QPainter(target)
    canvas._draw_base_image(painter)
    painter.end()
    assert abs(target.pixelColor(10, 10).green() - 223) <= 1
    assert bytes(source.constBits()) == original
    signature = canvas._overlay_underlay_signature()
    item.watermark = replace(item.watermark, opacity=0.5)
    assert canvas._overlay_underlay_signature() != signature
    canvas.close()


def test_export_plan_freezes_watermark_when_live_document_changes():
    item = document(WatermarkSpec(enabled=True, text="导出快照"))
    service = ExportService()
    selection = ExportSelection(include_measurement_overlay=True, include_watermark=True)
    plan = service.build_plan([item], selection)
    original = item.watermark
    item.watermark = WatermarkSpec(enabled=True, text="下一次")
    service._validate_export_plan(plan, [item])
    assert plan.render_contexts[0].watermark == original
    assert plan.options.include_watermark


def test_result_and_raw_export_watermark_controls():
    from fdm.ui.dialogs import ExportOptionsDialog
    from fdm.ui.raster_export_dialog import CurrentImageExportDialog

    dialog = ExportOptionsDialog(ExportSelection(include_combined_overlay=True), allow_all_scope=False, watermark_count_current=1)
    assert dialog.selection().include_watermark
    dialog._watermark.setChecked(False)
    assert not dialog.selection().include_watermark
    raw = CurrentImageExportDialog("test.png", watermark_available=True)
    assert not raw.include_watermark()
    raw.display_radio.setChecked(True)
    assert raw.include_watermark()
    raw.raw_radio.setChecked(True)
    assert not raw.include_watermark()
    slide = CurrentImageExportDialog("test.png", watermark_available=True, digital_slide_viewport=True)
    slide.display_radio.setChecked(True)
    assert not slide.include_watermark()
    mixed = ExportOptionsDialog(ExportSelection(include_combined_overlay=True), allow_all_scope=True, watermark_count_current=0, watermark_count_all=1)
    assert not mixed.selection().include_watermark
    mixed._scope_all.setChecked(True)
    assert mixed.selection().include_watermark
    mixed._scope_current.setChecked(True)
    assert not mixed.selection().include_watermark


@pytest.mark.parametrize("values", [{"opacity": float("nan")}, {"width_ratio": 0}, {"gap_x": -1}, {"logo_sha256": "missing"}, {"color": "bad"}])
def test_invalid_watermark_settings_are_rejected(values):
    with pytest.raises(ValueError):
        WatermarkSpec(**values)


@pytest.fixture
def window(tmp_path):
    from fdm.ui.main_window import MainWindow
    from fdm.ui.image_loader import ImageLoadRequest

    host = MainWindow()
    for index in range(2):
        source = QImage(320, 240, QImage.Format.Format_RGB32)
        source.fill(QColor("white"))
        path = tmp_path / f"source-{index}.png"
        source.save(str(path))
        item = ImageDocument(id=f"image-{index}", path=str(path), image_size=(320, 240))
        host._add_loaded_document(ImageLoadRequest(path=str(path), document=item), source)
    host.tab_widget.setCurrentIndex(0)
    yield host
    host._confirm_close_documents = lambda documents: True
    host.close()


def test_batch_is_one_undo_and_does_not_touch_slide_or_pixel_sources(window):
    ordinary = list(window.project.documents)
    slide = ImageDocument(id="slide", path="test.fdmslide", image_size=(10000, 10000), document_kind="digital_slide")
    window.project.documents.append(slide)
    image_bytes = {key: bytes(value.constBits()) for key, value in window._images.items()}
    revisions = [item.measurement_geometry_revision for item in ordinary]
    spec = WatermarkSpec(enabled=True, text="批量水印", layout="tile")
    assert window._apply_watermark(spec, all_open=True) == 2
    assert all(item.watermark == spec for item in ordinary)
    assert slide.watermark is None
    window.undo_current_document()
    assert all(item.watermark is None for item in ordinary)
    window.redo_current_document()
    assert all(item.watermark == spec for item in ordinary)
    assert revisions == [item.measurement_geometry_revision for item in ordinary]
    assert image_bytes == {key: bytes(value.constBits()) for key, value in window._images.items()}
    with patch.object(window, "current_document", return_value=slide):
        window._update_action_states()
        assert not window.watermark_action.isEnabled()
        assert window._apply_watermark(spec, all_open=True) == 0
    window.project.documents.remove(slide)


def test_native_export_uses_frozen_logo_and_raw_display_source_stays_clean(window, tmp_path):
    logo = logo_document(tmp_path)
    window._apply_watermark(logo.watermark, assets=logo.watermark_assets)
    item = window.current_document()
    plan = window.export_service.build_plan([item], ExportSelection(include_measurement_overlay=True, include_watermark=True))
    item.watermark = WatermarkSpec(enabled=True, text="changed after plan")
    output = tmp_path / "frozen.png"
    window._render_overlay_image(item, output, include_measurements=False, include_scale=False, include_watermark=True, render_mode=ExportImageRenderMode.FULL_RESOLUTION, render_context=plan.render_contexts[0])
    assert abs(QImage(str(output)).pixelColor(10, 10).green() - 223) <= 1
    assert window._images[item.id].pixelColor(10, 10) == QColor("white")
    plain = tmp_path / "plain.png"
    window._render_overlay_image(item, plain, include_measurements=False, include_scale=False, render_mode=ExportImageRenderMode.FULL_RESOLUTION)
    assert QImage(str(plain)).pixelColor(10, 10) == QColor("white")


def test_current_display_export_contains_watermark_and_raw_export_does_not(window, tmp_path, monkeypatch):
    from fdm.ui.raster_export_dialog import CurrentImageExportDialog

    logo = logo_document(tmp_path)
    window._apply_watermark(logo.watermark, assets=logo.watermark_assets)
    for display in (False, True):
        destination = tmp_path / f"output-{display}.png"
        dialog = CurrentImageExportDialog(destination, watermark_available=True)
        dialog.display_radio.setChecked(display)
        monkeypatch.setattr(dialog, "exec", lambda: QDialog.DialogCode.Accepted)
        monkeypatch.setattr("fdm.ui.main_window.CurrentImageExportDialog", lambda *args, **kwargs: dialog)
        window.export_current_image()
        assert destination.exists()
        color = QImage(str(destination)).pixelColor(10, 10)
        assert abs(color.green() - (223 if display else 255)) <= 1


@pytest.mark.parametrize("failed_case", ["font_database", "preferences_roundtrip", "text@1", "tile@2", "datetime_text@1.5", "datetime_logo@2"])
def test_watermark_release_probe_and_build_gate(tmp_path, failed_case):
    import subprocess
    import sys
    from fdm.ui.watermark_self_check import run_watermark_self_check

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_windows_onedir import run_packaged_self_check
    from fdm.ui.project_save_self_check import run_project_save_self_check

    from fdm.ui.scale_overlay_self_check import run_scale_overlay_self_check

    watermark = run_watermark_self_check()
    assert watermark["ok"]
    payload = {"ok": True, "errors": [], "functional_checks": {
        "project_save": run_project_save_self_check(),
        "overlay_renderer": {"ok": True, "worker_stdio_none": True, "scale_overlay": run_scale_overlay_self_check()},
        "fiber_quick_geometry": {"ok": True, "backend": "skimage_zhang", "geometry_revision": 3, "backend_version": "test", "compiled_extension": True},
        "watermark_renderer": watermark,
    }}
    for valid in (True, False):
        if not valid:
            watermark["cases"][failed_case] = False
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
        with patch("build_windows_onedir.subprocess.run", return_value=completed):
            errors = run_packaged_self_check(tmp_path)
        assert bool(errors) is not valid
        if not valid:
            assert any(failed_case in error for error in errors)


@pytest.mark.parametrize("dtype,pixel_type", [(np.uint16, "gray16"), (np.float32, "gray32_float")])
def test_native_precision_export_is_unchanged_with_watermark(window, tmp_path, monkeypatch, dtype, pixel_type):
    from fdm.raster import RasterPlane
    from fdm.services.raster_io import read_raster_file
    from fdm.ui.raster_export_dialog import CurrentImageExportDialog

    item = window.current_document()
    values = np.arange(320 * 240).reshape(240, 320).astype(dtype)
    if dtype == np.float32:
        values = values / 1000.0 - 2.5
    plane = RasterPlane(320, 240, pixel_type, values.tobytes())
    window._rasters[item.id] = plane
    window._apply_watermark(WatermarkSpec(enabled=True, text="不能写入原始像素"))
    destination = tmp_path / "native.tiff"
    dialog = CurrentImageExportDialog(destination, watermark_available=True)
    from fdm.services.raster_export import RasterExportFormat
    dialog.format_combo.setCurrentIndex(dialog.format_combo.findData(RasterExportFormat.TIFF))
    monkeypatch.setattr(dialog, "exec", lambda: QDialog.DialogCode.Accepted)
    monkeypatch.setattr("fdm.ui.main_window.CurrentImageExportDialog", lambda *args, **kwargs: dialog)
    window.export_current_image()
    loaded = read_raster_file(destination)
    assert loaded.success and loaded.plane.pixel_type == plane.pixel_type
    assert loaded.plane.data == plane.data


def test_magic_source_features_remain_reusable_after_watermark_change(window):
    from fdm.settings import MagicSegmentToolMode
    item = window.current_document()
    canvas = window.current_canvas()
    before = window._segmentation_source_for_request(item, canvas, MagicSegmentToolMode.STANDARD)
    pixels = bytes(before.image.constBits())
    window._apply_watermark(WatermarkSpec(enabled=True, text="仅显示层"))
    after = window._segmentation_source_for_request(item, canvas, MagicSegmentToolMode.STANDARD)
    assert before is after
    assert bytes(after.image.constBits()) == pixels


@pytest.mark.parametrize("option", ["include_measurement_overlay", "include_scale_overlay", "include_combined_overlay"])
def test_export_service_routes_watermark_to_each_image_output(window, tmp_path, option):
    logo = logo_document(tmp_path)
    window._apply_watermark(logo.watermark, assets=logo.watermark_assets)
    item = window.current_document()
    selection = ExportSelection(**{option: True}, include_watermark=True)
    target = tmp_path / option
    result = window.export_service.export_project(window.project, target, documents=[item], selection=selection, overlay_renderer=window._render_overlay_image)
    assert result.success
    outputs = list(target.glob("*.png"))
    assert len(outputs) == 1
    assert abs(QImage(str(outputs[0])).pixelColor(10, 10).green() - 223) <= 1


def test_drag_background_restores_watermark_and_invalidates_its_cache(scene, tmp_path):
    canvas, item, *_ = scene
    logo = logo_document(tmp_path)
    item.watermark = replace(logo.watermark, opacity=1, width_ratio=1)
    item.watermark_assets.update(logo.watermark_assets)
    item.select_measurement(item.measurements[0].id)
    canvas._dragging_area_handle = (item.measurements[0].id, 0, 0)

    def background():
        surface = QImage(canvas.size(), QImage.Format.Format_ARGB32_Premultiplied)
        surface.fill(QColor("white"))
        painter = QPainter(surface)
        try:
            canvas._redraw_selected_measurement_background(painter, canvas._paint_context())
        finally:
            painter.end()
        return surface

    first = background()
    assert first.pixelColor(65, 65).red() > first.pixelColor(65, 65).green()
    with patch.object(canvas, "_draw_base_image", wraps=canvas._draw_base_image) as draw_base:
        assert background() == first
        assert draw_base.call_count == 0
        item.watermark = replace(item.watermark, opacity=0)
        assert background() != first
        assert draw_base.call_count == 1
    canvas._dragging_area_handle = None


def test_stamp_lru_enforces_total_byte_budget():
    from fdm.ui.watermark_rendering import WatermarkRasterCache

    cache = WatermarkRasterCache(budget=2 * 1024 * 1024)
    image = QImage(512, 512, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor("red"))
    cache.put(("one",), image)
    cache.put(("two",), QImage(image))
    assert cache.get(("one",)) is not None
    cache.put(("three",), QImage(image))
    assert cache.get(("two",)) is None
    assert cache.bytes == cache.budget


def test_watermark_apply_undo_and_asset_repair_repaint_empty_canvas(tmp_path):
    item = document()
    canvas = DocumentCanvas()
    canvas.set_document(item, render(document()))
    canvas.notify_document_visual_changed()
    logo = logo_document(tmp_path)
    with patch.object(canvas, "update") as update:
        item.watermark = logo.watermark
        canvas.notify_document_visual_changed()
        update.assert_called_once_with()
        missing_signature = canvas._overlay_underlay_signature()
        update.reset_mock()
        item.watermark_assets.update(logo.watermark_assets)
        canvas.notify_document_visual_changed()
        update.assert_called_once_with()
        assert canvas._overlay_underlay_signature() != missing_signature
        update.reset_mock()
        item.watermark = None
        canvas.notify_document_visual_changed()
        update.assert_called_once_with()
        update.reset_mock()
        canvas.notify_document_visual_changed()
        update.assert_not_called()
    canvas.close()


@pytest.mark.parametrize("layout", ["single", "tile"])
def test_zoomed_watermark_uses_bounded_direct_drawing(tmp_path, monkeypatch, layout):
    item = logo_document(tmp_path, layout=layout)
    monkeypatch.setattr(watermark_raster_cache, "budget", 4096)
    # The output viewport is small although the image-relative watermark is
    # magnified beyond the cache budget. Its alpha must still be applied once.
    target = render(item, scale=20, size=(200, 100))
    assert abs(target.pixelColor(50, 50).green() - 223) <= 1
    assert watermark_raster_cache.bytes <= 4096


DATETIME_TEXT = "2026-09-21 14:35:26"


@pytest.mark.parametrize("kind", ["text", "logo"])
def test_datetime_is_a_separate_caption_below_unchanged_content(tmp_path, kind):
    item = logo_document(tmp_path, color="#000000")
    if kind == "text":
        item.watermark = replace(item.watermark, kind="text", text="实验室\nLaboratory")
    plain_geometry = watermark_geometry(item, item.watermark)
    plain = render(item)
    item.watermark = replace(item.watermark, include_datetime=True, datetime_text=DATETIME_TEXT)
    geometry = watermark_geometry(item, item.watermark)
    result = render(item)
    assert geometry.width == plain_geometry.width
    assert geometry.content_height == plain_geometry.height
    assert geometry.datetime_top > geometry.content_height
    assert geometry.height > plain_geometry.height
    body_height = int(plain_geometry.height)
    result_body = result.copy(0, 0, 60, body_height)
    plain_body = plain.copy(0, 0, 60, body_height)
    np.testing.assert_allclose(
        np.frombuffer(result_body.constBits(), np.uint8),
        np.frombuffer(plain_body.constBits(), np.uint8), atol=1, rtol=0,
    )
    caption = result.copy(0, int(geometry.datetime_top), 60, int(geometry.height - geometry.datetime_top) + 1)
    channels = np.frombuffer(caption.constBits(), np.uint8)
    assert channels.min() < 250
    assert channels.min() >= 190  # 25% black over white, applied only once.
    misses = watermark_raster_cache.misses
    assert render(item) == result
    assert watermark_raster_cache.misses == misses
    item.watermark = replace(item.watermark, datetime_text="2027-01-02 03:04:05")
    assert render(item) != result


@pytest.mark.parametrize("layout", ["single", "tile"])
@pytest.mark.parametrize("dpr", [1, 1.5, 2])
def test_datetime_group_keeps_viewport_phase_and_rotation(tmp_path, layout, dpr):
    item = logo_document(tmp_path, layout=layout, rotation=-30, include_datetime=True, datetime_text=DATETIME_TEXT)
    full = render(item, dpr=dpr)
    viewport = render(item, dpr=dpr, origin=(-40, -20), size=(100, 80))
    expected = full.copy(round(40 * dpr), round(20 * dpr), round(100 * dpr), round(80 * dpr))
    np.testing.assert_allclose(
        np.frombuffer(viewport.constBits(), np.uint8), np.frombuffer(expected.constBits(), np.uint8), atol=1, rtol=0,
    )


def test_datetime_edit_is_optional_fixed_and_draft_only(tmp_path):
    item = logo_document(tmp_path)
    before = item.snapshot_state()
    dialog = WatermarkDialog(item, render(document()))
    assert not dialog.datetime_check.isChecked()
    assert not dialog.datetime_edit.isEnabled()
    assert dialog.watermark().datetime_text == ""
    assert "datetime_text" not in dialog.watermark().to_dict()
    dialog._accept()
    assert dialog.watermark() == item.watermark
    dialog.datetime_check.setChecked(True)
    dialog.datetime_edit.setDateTime(QDateTime.fromString(DATETIME_TEXT, DATETIME_DISPLAY_FORMAT))
    dialog._refresh_preview()
    assert dialog.font_combo.isEnabled()  # Logo captions use the same text controls.
    spec = dialog.watermark()
    assert spec.include_datetime and spec.datetime_text == DATETIME_TEXT
    dialog.reject()
    assert item.snapshot_state() == before
    item.watermark = spec
    reopened = WatermarkDialog(item, render(document()))
    assert reopened.datetime_edit.dateTime().toString(DATETIME_DISPLAY_FORMAT) == DATETIME_TEXT
    reopened.datetime_check.setChecked(False)
    assert not reopened.watermark().include_datetime
    assert reopened.watermark().datetime_text == DATETIME_TEXT
    reopened.datetime_check.setChecked(True)
    assert reopened.watermark() == spec
    reopened.reject()
    dialog.deleteLater()
    reopened.deleteLater()


def test_datetime_save_history_and_frozen_export(window, tmp_path):
    spec = WatermarkSpec(enabled=True, text="Laboratory", include_datetime=True, datetime_text=DATETIME_TEXT)
    window._apply_watermark(spec, all_open=True)
    window.undo_current_document()
    assert all(item.watermark is None for item in window.project.documents)
    window.redo_current_document()
    assert all(item.watermark == spec for item in window.project.documents)
    path = tmp_path / "datetime.fdmproj"
    ProjectIO.save(window.project, path)
    reopened = ProjectIO.load(path)
    assert "image-watermark-datetime/v1" in reopened.required_features
    assert all(item.watermark == spec for item in reopened.documents)
    item = window.current_document()
    plan = window.export_service.build_plan([item], ExportSelection(include_combined_overlay=True, include_watermark=True))
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    kwargs = dict(include_measurements=False, include_scale=False, include_watermark=True, render_mode=ExportImageRenderMode.FULL_RESOLUTION)
    window._render_overlay_image(item, before, **kwargs)
    window._apply_watermark(replace(spec, datetime_text="2028-02-29 23:59:59"))
    window._render_overlay_image(item, after, render_context=plan.render_contexts[0], **kwargs)
    assert QImage(str(before)) == QImage(str(after))


def test_datetime_project_protects_against_older_watermark_readers(tmp_path, monkeypatch):
    from fdm import models
    from fdm.project_io import ProjectCompatibilityError

    legacy = WatermarkSpec.from_dict({"enabled": True, "text": "older watermark"})
    assert not legacy.include_datetime
    assert "image-watermark-datetime/v1" not in ProjectState(version="test", documents=[document(legacy)]).effective_required_features()
    item = document(replace(legacy, include_datetime=True, datetime_text=DATETIME_TEXT))
    target = tmp_path / "datetime.fdmproj"
    ProjectIO.save(ProjectState(version="test", documents=[item]), target)
    monkeypatch.setattr(models, "SUPPORTED_PROJECT_REQUIRED_FEATURES", models.SUPPORTED_PROJECT_REQUIRED_FEATURES - {"image-watermark-datetime/v1"})
    loaded = ProjectIO.load(target)
    assert loaded.is_read_only_compatible
    with pytest.raises(ProjectCompatibilityError):
        ProjectIO.save(loaded, target)


@pytest.mark.parametrize("value", ["2026-02-30 10:00:00", "2026-09-21 24:00:00", "2026/09/21", "2026-09-21 10:00:00Z", 123])
def test_invalid_datetime_cannot_be_saved_as_a_watermark(value):
    with pytest.raises(ValueError):
        WatermarkSpec(include_datetime=True, datetime_text=value)
