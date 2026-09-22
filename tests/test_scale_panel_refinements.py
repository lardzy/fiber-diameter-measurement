"""Scale editor actions stay available across every modal export outcome."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QCheckBox, QDialog

from fdm.models import Calibration, ImageDocument
from fdm.services.export_service import ExportScope, ExportSelection
from fdm.settings import AppSettingsIO


@pytest.fixture
def scale_window(tmp_path):
    from fdm.ui.image_loader import ImageLoadRequest
    from fdm.ui.main_window import MainWindow

    host = MainWindow()
    image = QImage(1024, 768, QImage.Format.Format_RGB32)
    image.fill(QColor("#203040"))
    path = tmp_path / "scale-panel.png"
    image.save(str(path))
    doc = ImageDocument(
        "scale-panel",
        str(path),
        (1024, 768),
        calibration=Calibration("preset", 4.0, "um", "test"),
    )
    host._add_loaded_document(ImageLoadRequest(str(path), document=doc), image)
    yield host
    host._confirm_close_documents = lambda _: True
    host.close()


def test_frequent_controls_visible_and_output_content_only_in_export_dialog(
    scale_window,
):
    preview = scale_window.scale_preview
    preview.begin()
    panel = preview.panel
    assert not panel.more.isVisible()
    assert not panel.more.isAncestorOf(panel.style_combo)
    assert not panel.more.isAncestorOf(panel.stroke)
    assert {
        panel.style_combo.itemData(i) for i in range(panel.style_combo.count())
    } == {"line", "ticks", "bar", "ticks_up", "ticks_down", "divisions"}
    assert not any(
        checkbox.text().startswith("包含") for checkbox in panel.findChildren(QCheckBox)
    )
    top = panel.layout().itemAt(0).layout()
    assert top.itemAt(0).widget() is panel.export_button
    assert top.itemAt(1).widget() is panel.hide_button
    assert panel.export_button.property("primary") is True


@pytest.mark.parametrize(
    "outcome", ["cancel_options", "cancel_path", "fail", "success"]
)
def test_export_keeps_panel_open_and_latest_confirmed_settings(
    scale_window, tmp_path, monkeypatch, outcome
):
    host = scale_window
    preview = host.scale_preview
    preview.begin()
    preview.change(color="#345678", font_mode="custom", font_size=24)
    selected = []

    def dialog(preset):
        selected.append(preset)
        return SimpleNamespace(
            DialogCode=QDialog.DialogCode,
            exec=lambda: (
                QDialog.DialogCode.Rejected
                if outcome == "cancel_options"
                else QDialog.DialogCode.Accepted
            ),
            selection=lambda: preset,
        )

    monkeypatch.setattr(host, "_create_export_options_dialog", dialog)
    monkeypatch.setattr(
        host,
        "_select_export_save_path",
        lambda *_: "" if outcome == "cancel_path" else str(tmp_path / "output.png"),
    )
    monkeypatch.setattr(host, "_show_export_information", lambda *_: None)
    warnings = []
    monkeypatch.setattr(
        host, "_show_export_warning", lambda *args: warnings.append(args)
    )
    if outcome == "fail":

        def fail(*_, **__):
            raise OSError("simulated export failure")

        monkeypatch.setattr(host, "_render_overlay_image", fail)

    preview.export()

    assert len(selected) == 1
    assert preview.visible and preview.editing
    assert not preview.panel.isHidden()
    assert host.scale_preview_action.isChecked()
    assert AppSettingsIO.load().last_scale_overlay == preview.confirmed
    assert preview.confirmed.color == "#345678"
    assert bool(warnings) == (outcome == "fail")
    assert (tmp_path / "output.png").exists() == (outcome == "success")

    # Further unconfirmed changes can still be cancelled back to this export's
    # accepted configuration, without losing the latest preference on disk.
    accepted = preview.confirmed
    preview.change(color="#FEDCBA")
    preview.cancel()
    assert preview.spec == accepted
    assert AppSettingsIO.load().last_scale_overlay == accepted


@pytest.mark.parametrize("accept", ["finish", "hide"])
def test_finish_and_hide_remember_settings_without_export(scale_window, accept):
    preview = scale_window.scale_preview
    preview.begin()
    preview.change(color="#123456", line_width=4.0)
    getattr(preview, accept)()
    accepted = preview.confirmed
    assert AppSettingsIO.load().last_scale_overlay == accepted
    assert not preview.editing
    assert preview.visible == (accept == "finish")
    preview.begin()
    preview.change(line_width=9.0)
    preview.cancel()
    assert preview.spec == accepted
    assert AppSettingsIO.load().last_scale_overlay == accepted


def test_unconfirmed_cancel_does_not_create_preferences(scale_window):
    preview = scale_window.scale_preview
    preview.begin()
    before = preview.spec
    preview.change(line_width=7.0)
    preview.cancel()
    assert preview.spec == before
    assert AppSettingsIO.load().last_scale_overlay is None


def test_focus_existing_editor_keeps_export_options_and_round_trip(
    scale_window, monkeypatch
):
    preview = scale_window.scale_preview
    preset = ExportSelection(
        include_csv=True,
        include_combined_overlay=True,
        include_annotations=False,
        include_watermark=False,
        include_construction_geometry=True,
        scope=ExportScope.ALL_OPEN,
    )
    preview.begin(preset)
    preview.begin()  # Focusing the active editor must not reset a pending export.
    recorded = []
    monkeypatch.setattr(scale_window, "export_results", recorded.append)
    preview.export()
    assert len(recorded) == 1
    assert replace(recorded[0], scale_overlay=None) == preset
    assert preview.editing and not preview.panel.isHidden()


@pytest.mark.parametrize("end_edit", ["finish", "hide", "cancel"])
def test_new_editor_session_resets_previous_export_scope_and_content(
    scale_window, monkeypatch, end_edit
):
    preview = scale_window.scale_preview
    preview.begin(
        ExportSelection(
            include_csv=True, include_combined_overlay=True, scope=ExportScope.ALL_OPEN
        )
    )
    getattr(preview, end_edit)()
    preview.begin()
    recorded = []
    monkeypatch.setattr(scale_window, "export_results", recorded.append)
    preview.export()
    assert len(recorded) == 1
    result = recorded[0]
    assert result.scope == ExportScope.CURRENT
    assert result.include_scale_overlay
    assert not result.include_csv and not result.include_combined_overlay


def test_failed_preference_save_can_retry_without_losing_session_settings(
    scale_window, monkeypatch
):
    preview = scale_window.scale_preview
    preview.begin()
    preview.change(color="#ABCDEF")
    monkeypatch.setattr(scale_window, "_save_app_settings", lambda **_: False)
    assert preview.finish()
    assert preview.confirmed.color == "#ABCDEF"
    assert scale_window._app_settings.last_scale_overlay is None
    preview.begin()
    calls = []
    monkeypatch.setattr(
        scale_window, "_save_app_settings", lambda **args: calls.append(args) or True
    )
    assert preview.finish()
    assert len(calls) == 1
    assert scale_window._app_settings.last_scale_overlay == preview.confirmed
