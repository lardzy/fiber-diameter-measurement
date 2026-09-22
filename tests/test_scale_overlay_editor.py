from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter
from PySide6.QtWidgets import QDialog

from fdm.models import Calibration, ImageDocument, Point
from fdm.scale_overlay import ScaleOverlaySpec
from fdm.settings import AppSettings, AppSettingsIO
from fdm.services.export_service import (
    ExportImageRenderMode as Mode,
    ExportScope,
    ExportSelection,
)
from fdm.ui.dialogs import ExportOptionsDialog, SettingsDialog
from fdm.ui.scale_overlay_rendering import (
    layout_scale_overlay,
    moved_spec,
    paint_scale_overlay,
)


@pytest.fixture
def window(tmp_path):
    from fdm.ui.image_loader import ImageLoadRequest
    from fdm.ui.main_window import MainWindow

    host = MainWindow()
    for index in range(2):
        image = QImage(1024, 768, QImage.Format.Format_RGB32)
        image.fill(QColor("#203040"))
        path = tmp_path / f"scale-{index}.png"
        image.save(str(path))
        document = ImageDocument(
            f"scale-{index}",
            str(path),
            (1024, 768),
            calibration=Calibration("preset", 4.0 if index == 0 else 8.0, "um", "test"),
        )
        host._add_loaded_document(ImageLoadRequest(str(path), document=document), image)
    host.tab_widget.setCurrentIndex(0)
    yield host
    host._confirm_close_documents = lambda _: True
    host.close()


@pytest.mark.parametrize(
    "unit,length",
    [("nm", 50000), ("um", 50), ("mm", 0.05), ("cm", 0.005), ("m", 0.00005)],
)
@pytest.mark.parametrize(
    "cal_unit,ppu",
    [("nm", 0.004), ("um", 4.0), ("mm", 4000.0), ("cm", 40000.0), ("m", 4000000.0)],
)
def test_fixed_physical_length_converts_across_all_calibrations(
    unit, length, cal_unit, ppu
):
    spec = ScaleOverlaySpec(length_mode="custom", length=length, unit=unit)
    layout = layout_scale_overlay(spec, (0.0, 0.0, 1024.0, 768.0), ppu, cal_unit)
    assert layout.end[0] - layout.start[0] == pytest.approx(200.0)
    assert spec.with_unit("um").length == pytest.approx(50.0)


@pytest.mark.parametrize(
    "position", ["top_left", "top_right", "bottom_left", "bottom_right", "manual"]
)
@pytest.mark.parametrize("text_position", ["above", "below"])
def test_complete_bounds_constrained_even_when_label_wider_than_bar(
    position, text_position
):
    spec = ScaleOverlaySpec(
        length_mode="custom",
        length=0.0005,
        unit="um",
        position=position,
        font_mode="custom",
        font_size=56,
        text_position=text_position,
    )
    target = (8192.0, 4096.0, 1200.0, 900.0)
    layout = layout_scale_overlay(spec, target, 4.0, "um")
    assert QRectF(*target).contains(QRectF(*layout.bounds))
    assert QRectF(*layout.bounds).contains(QRectF(*layout.text_rect))
    for dx, dy in ((-1e6, -1e6), (1e6, 1e6)):
        moved = moved_spec(layout, dx, dy)
        assert QRectF(*target).contains(
            QRectF(*layout_scale_overlay(moved, target, 4.0, "um").bounds)
        )


def test_auto_px_and_invalid_fixed_physical_scale():
    layout = layout_scale_overlay(ScaleOverlaySpec(), (0.0, 0.0, 1000.0, 800.0))
    assert layout.value == 200 and layout.unit == "px" and layout.font_size == 12
    with pytest.raises(ValueError, match="尚未标定"):
        layout_scale_overlay(
            ScaleOverlaySpec(length_mode="custom"), (0.0, 0.0, 1000.0, 800.0)
        )
    with pytest.raises(ValueError, match="超出"):
        layout_scale_overlay(
            ScaleOverlaySpec(length_mode="custom", length=1e6),
            (0.0, 0.0, 1000.0, 800.0),
            4.0,
            "um",
        )


@pytest.mark.parametrize("dpr", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("zoom", [0.5, 1.0, 2.0])
def test_canvas_and_export_same_geometry_at_every_zoom_and_dpr(dpr, zoom):
    spec = ScaleOverlaySpec(font_mode="custom", font_size=48)
    results = []
    for origin in ((0.0, 0.0), (8192.0, 4096.0), (230.0, 51.0)):
        layout = layout_scale_overlay(spec, (*origin, 800.0, 600.0), 4.0, "um")
        assert layout.font_size == 48
        image = QImage(
            round(800 * zoom * dpr),
            round(600 * zoom * dpr),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.setDevicePixelRatio(dpr)
        image.fill(QColor("#203040"))
        painter = QPainter(image)
        painter.scale(zoom, zoom)
        painter.translate(-origin[0], -origin[1])
        paint_scale_overlay(painter, layout)
        painter.end()
        results.append(image)
    assert results[0] == results[1] == results[2]


def test_editor_lifecycle_scope_and_independent_history(window):
    preview = window.scale_preview
    original = [
        (
            document.dirty_flags.to_dict()
            if hasattr(document.dirty_flags, "to_dict")
            else repr(document.dirty_flags),
            document.history.can_undo(),
        )
        for document in window.project.documents
    ]
    assert not preview.visible
    window.scale_preview_action.trigger()
    assert preview.visible and preview.editing and preview.scope == "all"
    assert all(preview.is_visible(document) for document in window.project.documents)
    before = preview.spec
    preview.change(color="#000000")
    window.undo_current_document()
    assert preview.spec == before
    window.redo_current_document()
    assert preview.spec.color == "#000000"
    preview.cancel()
    assert preview.spec == before
    preview.begin()
    preview.change(font_mode="custom", font_size=45)
    assert preview.finish() and preview.visible and not preview.editing
    assert window._app_settings.last_scale_overlay == preview.confirmed
    preview.set_scope("current")
    first, second = window.project.documents
    assert preview.is_visible(first) and not preview.is_visible(second)
    window.tab_widget.setCurrentIndex(1)
    assert not preview.is_visible(second)
    preview.set_scope("all")
    assert preview.is_visible(second)
    preview.hide()
    assert not preview.visible
    preview.begin()
    assert preview.spec.font_size == 45
    assert original == [
        (
            document.dirty_flags.to_dict()
            if hasattr(document.dirty_flags, "to_dict")
            else repr(document.dirty_flags),
            document.history.can_undo(),
        )
        for document in window.project.documents
    ]


def test_nonediting_preview_does_not_consume_measurement_mouse_events(window):
    preview = window.scale_preview
    preview.begin()
    preview.finish()
    canvas = window.current_canvas()
    point = QPointF(200, 200)
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        point,
        point,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    assert preview.eventFilter(canvas, event) is False


def test_canvas_scale_pixels_match_frozen_export_after_measurement_refresh(
    window, tmp_path
):
    import numpy as np
    from fdm.geometry import Line
    from fdm.models import Measurement
    from test_canvas_overlay_handoff import frame, pixels

    preview = window.scale_preview
    preview.begin()
    preview.change(
        color="#FF00FF", text_color="#FF00FF", font_mode="custom", font_size=42
    )
    preview.finish()
    doc, canvas = window.current_document(), window.current_canvas()
    canvas.setFixedSize(1024, 768)
    # QWidget delivers its initial resize lazily on the first offscreen render.
    frame(canvas)
    canvas._zoom = 1.0
    canvas._pan = Point(0, 0)
    selection = ExportSelection(
        include_scale_overlay=True, scale_overlay=preview.confirmed
    )
    context = window._prepare_scale_export(selection, [doc], {})[doc.id]
    path = tmp_path / "scale-pixel-reference.png"
    window._render_overlay_image(
        doc,
        path,
        include_measurements=False,
        include_scale=True,
        include_annotations=False,
        render_mode=selection.render_mode,
        render_context=context,
    )

    def magenta(image):
        array = pixels(image).astype(np.int16)
        return (array[:, :, 0] - array[:, :, 1] > 120) & (
            array[:, :, 2] - array[:, :, 1] > 120
        )

    reference = magenta(QImage(str(path)))
    assert reference.any()
    np.testing.assert_array_equal(magenta(frame(canvas)), reference)
    doc.measurements.append(
        Measurement(
            "moving",
            doc.id,
            None,
            "manual",
            line_px=Line(Point(100, 100), Point(350, 200)),
        )
    )
    doc.mark_measurement_geometry_changed()
    canvas._sync_overlay_visual_state()
    np.testing.assert_array_equal(magenta(frame(canvas)), reference)
    preview.hide()
    assert not magenta(frame(canvas)).any()
    preview.begin()
    preview.finish()
    np.testing.assert_array_equal(magenta(frame(canvas)), reference)


def test_length_handle_changes_label_and_keeps_physical_mapping(window):
    preview = window.scale_preview
    preview.begin()
    canvas = window.current_canvas()
    layout, _ = preview.current_layout()
    start = canvas.image_to_widget(Point(*layout.end))

    def send(kind, pos, button, buttons):
        return preview.eventFilter(
            canvas,
            QMouseEvent(
                kind, pos, pos, button, buttons, Qt.KeyboardModifier.NoModifier
            ),
        )

    assert send(
        QEvent.Type.MouseButtonPress,
        start,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    new_pos = start + QPointF(-40 * canvas.view_zoom(), 0)
    send(
        QEvent.Type.MouseMove,
        new_pos,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
    )
    send(
        QEvent.Type.MouseButtonRelease,
        new_pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
    )
    updated, _ = preview.current_layout()
    assert preview.spec.length_mode == "custom"
    assert updated.end[0] - updated.start[0] == pytest.approx(
        layout.end[0] - layout.start[0] - 40
    )
    assert updated.value * 4 == pytest.approx(updated.end[0] - updated.start[0])
    preview.undo()
    assert preview.spec == layout.spec


def test_settings_remember_only_shared_spec_and_preserve_in_general_settings(tmp_path):
    spec = ScaleOverlaySpec(font_mode="custom", font_size=72, unit="nm", length=50000)
    path = tmp_path / "scale-settings.json"
    AppSettingsIO.save(AppSettings(last_scale_overlay=spec), path)
    settings = AppSettingsIO.load(path)
    assert settings.last_scale_overlay == spec
    dialog = SettingsDialog(settings, document=None)
    try:
        assert dialog.app_settings().last_scale_overlay == spec
        assert not hasattr(dialog, "_scale_overlay_length_spin")
    finally:
        dialog.close()
    assert AppSettings.from_dict({}).last_scale_overlay is None
    assert AppSettings.from_dict({"last_scale_overlay": {"length": -1}}).load_issues


@pytest.mark.parametrize(
    "mode", [Mode.FULL_RESOLUTION, Mode.SCREEN_SCALE_FULL_IMAGE, Mode.CURRENT_VIEWPORT]
)
def test_export_freezes_scale_calibration_zoom_pan_and_size(window, tmp_path, mode):
    doc, canvas = window.current_document(), window.current_canvas()
    canvas.resize(720, 540)
    canvas._zoom = 0.75
    canvas._pan = Point(-50, -30)
    spec = ScaleOverlaySpec(font_mode="custom", font_size=40, color="#FF00FF")
    selection = ExportSelection(
        include_scale_overlay=True, scale_overlay=spec, render_mode=mode
    )
    contexts = window._prepare_scale_export(selection, [doc], {})
    context = contexts[doc.id]
    plan = window.export_service.build_plan([doc], selection, render_contexts=contexts)
    assert plan.render_contexts[0] == context
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"

    def render(path):
        window._render_overlay_image(
            doc,
            path,
            include_measurements=False,
            include_scale=True,
            include_annotations=False,
            render_mode=mode,
            render_context=context,
        )

    render(before)
    canvas._zoom = 4.0
    canvas._pan = Point(123, -450)
    canvas.resize(350, 250)
    doc.calibration = Calibration("preset", 400, "nm", "changed")
    window._app_settings.scale_overlay_length_value = 1e6
    window.scale_preview.change(color="#FFFFFF", font_size=96)
    render(after)
    assert QImage(str(before)) == QImage(str(after))
    assert context.scale_layout.font_size == 40


def test_batch_auto_per_image_and_fixed_preflight_lists_every_failure(window):
    docs = window.project.documents
    spec = ScaleOverlaySpec()
    selection = ExportSelection(include_scale_overlay=True, scale_overlay=spec)
    contexts = window._prepare_scale_export(selection, docs, {})
    assert (
        contexts[docs[0].id].scale_layout.value
        != contexts[docs[1].id].scale_layout.value
    )
    docs[1].calibration = None
    selection.scale_overlay = replace(spec, length_mode="custom", length=1e8)
    with pytest.raises(ValueError) as error:
        window._prepare_scale_export(selection, docs, {})
    assert all(doc.path.split("/")[-1] in str(error.value) for doc in docs)
    assert "尚未标定" in str(error.value) and "超出" in str(error.value)


def test_scale_selection_alone_creates_a_frozen_service_plan(window):
    spec = ScaleOverlaySpec(length_mode="custom", length=50000, unit="nm")
    selection = ExportSelection(include_scale_overlay=True, scale_overlay=spec)
    doc = window.current_document()
    plan = window.export_service.build_plan([doc], selection)
    assert plan.options.scale_overlay == spec
    frozen = plan.render_contexts[0]
    assert frozen.scale_layout.end[0] - frozen.scale_layout.start[0] == pytest.approx(
        200
    )
    assert frozen.output_size == doc.image_size
    assert frozen.calibration_snapshot == (4.0, "um")


def test_all_images_scope_includes_later_images_and_legacy_anchor_is_read_only(
    window, tmp_path
):
    from fdm.ui.image_loader import ImageLoadRequest

    current = window.current_document()
    current.scale_overlay_anchor = Point(125, 550)
    window._app_settings.scale_overlay_placement_mode = "manual"
    preview = window.scale_preview
    preview.spec = ScaleOverlaySpec.from_legacy_settings(window._app_settings)
    preview.begin()
    assert preview.spec.position == "manual"
    preview.change(relative_x=0.7)
    preview.finish()
    assert current.scale_overlay_anchor == Point(125, 550)
    image = QImage(300, 200, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    path = tmp_path / "later.png"
    image.save(str(path))
    doc = ImageDocument("later", str(path), (300, 200))
    window._add_loaded_document(ImageLoadRequest(str(path), document=doc), image)
    assert preview.is_visible(doc)
    assert getattr(window._canvases[doc.id], "_scale_overlay_preview") is preview
    assert doc.scale_overlay_anchor is None


def test_export_dialog_round_trip_keeps_output_selection(window, monkeypatch):
    selection = ExportSelection(
        include_csv=True,
        include_combined_overlay=True,
        scope=ExportScope.ALL_OPEN,
        include_construction_geometry=True,
        render_mode=Mode.CURRENT_VIEWPORT,
    )
    dialog = ExportOptionsDialog(selection, allow_all_scope=True)
    monkeypatch.setattr(
        dialog,
        "exec",
        lambda: dialog._request_scale_adjustment() or QDialog.DialogCode.Rejected,
    )
    monkeypatch.setattr(window, "_create_export_options_dialog", lambda _: dialog)
    window.export_results(selection)
    assert window.scale_preview.editing
    recorded = []
    monkeypatch.setattr(window, "export_results", recorded.append)
    window.scale_preview.export()
    assert len(recorded) == 1
    result = recorded[0]
    assert (
        result.include_csv
        and result.include_combined_overlay
        and result.include_construction_geometry
    )
    assert (
        result.scope == ExportScope.ALL_OPEN
        and result.render_mode == Mode.CURRENT_VIEWPORT
    )
    dialog.close()


@pytest.mark.parametrize("outcome", ["cancel", "fail", "success"])
def test_confirmed_settings_survive_export_outcomes_and_restart_is_hidden(
    window, tmp_path, monkeypatch, outcome
):
    preview = window.scale_preview
    preview.begin()
    preview.change(font_mode="custom", font_size=50)
    preview.finish()
    selection = ExportSelection(include_scale_overlay=True)
    dialog = SimpleNamespace(
        DialogCode=QDialog.DialogCode,
        exec=lambda: QDialog.DialogCode.Accepted,
        selection=lambda: selection,
    )
    monkeypatch.setattr(window, "_create_export_options_dialog", lambda _: dialog)
    monkeypatch.setattr(
        window,
        "_select_export_save_path",
        lambda *_: "" if outcome == "cancel" else str(tmp_path / "export.png"),
    )
    monkeypatch.setattr(window, "_show_export_information", lambda *_: None)
    monkeypatch.setattr(window, "_show_export_warning", lambda *_: None)
    if outcome == "fail":

        def fail(*_, **__):
            raise OSError("test write failed")

        monkeypatch.setattr(window, "_render_overlay_image", fail)
    window.export_results(selection)
    saved = AppSettingsIO.load().last_scale_overlay
    assert saved == preview.confirmed
    from fdm.ui.main_window import MainWindow

    restarted = MainWindow()
    try:
        assert (
            not restarted.scale_preview.visible and not restarted.scale_preview.editing
        )
        if saved:
            assert restarted.scale_preview.spec == saved
    finally:
        restarted.close()


@pytest.fixture
def digital_window(tmp_path):
    from fdm.services.digital_slide_store import (
        DigitalSlideManifest,
        DigitalSlideStore,
        DigitalSlideTile,
    )
    from fdm.ui.main_window import MainWindow

    path = tmp_path / "scale-slide.fdmslide"
    manifest = DigitalSlideManifest(
        version=1,
        width=16384,
        height=8192,
        viewport_width=240,
        viewport_height=180,
        focus_levels=[0, 1],
    )
    store = DigitalSlideStore.create(path, manifest)
    for focus, color in enumerate(("#203040", "#405060")):
        image = QImage(240, 180, QImage.Format.Format_RGB32)
        image.fill(QColor(color))
        store.write_tile(
            DigitalSlideTile(z_index=focus, x=8192, y=4096, width=240, height=180),
            image,
        )
    store.close()
    document = ImageDocument(
        "scale-slide",
        str(path),
        (16384, 8192),
        document_kind="digital_slide",
        calibration=Calibration("preset", 2.0, "um", "test"),
        metadata={"digital_slide": {"viewport_origin": [8192, 4096], "focus_index": 0}},
    )
    document.initialize_runtime_state()
    host = MainWindow()
    host._add_digital_slide_document_from_path(path, document=document)
    yield host
    host._reset_workspace()
    host.close()


@pytest.mark.parametrize(
    "style", ["line", "ticks", "bar", "ticks_up", "ticks_down", "divisions"]
)
def test_digital_freezes_native_origin_focus_calibration_and_output(
    digital_window, tmp_path, style
):
    window = digital_window
    doc, canvas = window.current_document(), window.current_canvas()
    spec = ScaleOverlaySpec(
        font_mode="custom",
        font_size=32,
        color="#FF00FF",
        text_color="#FF00FF",
        style=style,
    )
    selection = ExportSelection(
        include_scale_overlay=True,
        render_mode=Mode.CURRENT_VIEWPORT,
        scale_overlay=spec,
    )
    contexts = window._export_render_contexts([doc], selection.render_mode)
    contexts = window._prepare_scale_export(selection, [doc], contexts)
    frozen = contexts[doc.id]
    assert (frozen.origin_x, frozen.origin_y) == (8192, 4096)
    assert frozen.scale_layout.target == (8192, 4096, 240, 180)
    reference = None
    for zoom in (0.01, 0.5, 1.5, 8.0):
        canvas._focus_index = 1
        canvas._zoom = zoom
        canvas._pan = Point(-100, -20)
        doc.calibration = Calibration("preset", 1000, "nm", "mutated")
        window.scale_preview.change(font_size=80, length=1e6)
        target = tmp_path / f"frozen-{zoom}.png"
        result = window._render_overlay_image(
            doc,
            target,
            include_measurements=False,
            include_scale=True,
            render_mode=selection.render_mode,
            render_context=frozen,
        )
        assert (result.width, result.height) == (240, 180)
        image = QImage(str(target))
        assert image.pixelColor(0, 0).name() == "#203040"
        # Matching two frozen exports is insufficient if both lost the same
        # geometry. Check the entire interior baseline in the actual slide crop.
        bar = frozen.scale_layout
        row = math.floor(bar.start[1] - frozen.origin_y)
        first = math.ceil(bar.start[0] - frozen.origin_x) + 1
        last = math.floor(bar.end[0] - frozen.origin_x) - 1
        assert all(
            image.pixelColor(x, row).name() == "#ff00ff" for x in range(first, last)
        )
        if reference is None:
            reference = image
        else:
            assert image == reference


def test_digital_preview_masks_old_focus_until_exact_native_pixels_arrive(
    digital_window,
):
    window = digital_window
    preview = window.scale_preview
    canvas = window.current_canvas()
    canvas.resize(600, 480)
    canvas._zoom = 1.0
    canvas._pan = Point(80 - 8192, 80 - 4096)
    preview.begin()
    native = QImage(240, 180, QImage.Format.Format_RGB32)
    native.fill(QColor("#203040"))
    canvas._image = native
    canvas._native_frame_key = canvas._native_request_key()

    def draw_base():
        image = QImage(600, 480, QImage.Format.Format_RGB32)
        image.fill(QColor("black"))
        painter = QPainter(image)
        canvas._draw_base_image(painter)
        painter.end()
        return image

    assert preview.native_ready(canvas)
    point = canvas.image_to_widget(Point(8312, 4186)).toPoint()
    assert draw_base().pixelColor(point).name() == "#203040"
    canvas._focus_index = 1
    assert not preview.native_ready(canvas)
    assert draw_base().pixelColor(point).name() == "#29343c"
    canvas._image = native.copy()
    canvas._image.fill(QColor("#405060"))
    canvas._native_frame_key = canvas._native_request_key()
    assert draw_base().pixelColor(point).name() == "#405060"
    assert preview.current_layout()[0].target[:2] == (8192, 4096)
    assert (
        preview.panel.region.currentData() == "viewport"
        and not preview.panel.region.isEnabled()
    )


def test_digital_export_keeps_existing_full_slide_restriction(digital_window):
    with pytest.raises(ValueError, match="仅支持"):
        digital_window._export_render_contexts(
            digital_window.project.documents, Mode.FULL_RESOLUTION
        )


def test_digital_preview_layout_does_not_read_store_each_frame(
    digital_window, monkeypatch
):
    window = digital_window
    window.scale_preview.begin()
    store = window._slide_stores[window.current_document().id]

    def unexpected_read():
        raise AssertionError("per-frame disk read")

    monkeypatch.setattr(store, "read_manifest", unexpected_read)
    for _ in range(60):
        assert window.scale_preview.current_layout()[0] is not None
