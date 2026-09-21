from dataclasses import replace
from unittest.mock import patch

import pytest
from PySide6.QtCore import QDateTime
from PySide6.QtWidgets import QDialog

from fdm import settings
from fdm.services.watermark_preferences import load_watermark_default_assets, save_watermark_defaults
from fdm.settings import AppSettings, AppSettingsIO
from fdm.ui.dialogs import SettingsDialog
from fdm.ui.watermark_dialog import DATETIME_DISPLAY_FORMAT, WatermarkDialog
from fdm.watermark import WatermarkSpec
from test_watermark import document, logo_document, render, window as window


def test_first_watermark_matches_requested_initial_values():
    item = document()
    before = QDateTime.currentDateTime().toSecsSinceEpoch()
    dialog = WatermarkDialog(item, render(item))
    spec = dialog.watermark()
    assert spec == WatermarkSpec(
        enabled=True, kind="text", text="GTTC", font_family="", bold=True, italic=False,
        color="#666666", include_datetime=True, datetime_text=spec.datetime_text,
        layout="tile", anchor="bottom_right", opacity=0.75, width_ratio=0.25,
        rotation=45, offset_x=0.02, offset_y=0.02, gap_x=0.25, gap_y=0.25,
    )
    timestamp = QDateTime.fromString(spec.datetime_text, DATETIME_DISPLAY_FORMAT)
    assert before <= timestamp.toSecsSinceEpoch() <= QDateTime.currentDateTime().toSecsSinceEpoch()
    assert not dialog.apply_to_all()
    assert item.watermark is None  # Opening the form does not enable an image overlay.
    dialog.reject()
    dialog.deleteLater()


def test_remembered_spec_roundtrip_and_settings_dialog_preserve_all_fields(tmp_path):
    spec = WatermarkSpec(
        enabled=True, text="机构\nCopyright", bold=True, italic=True, color="#123456",
        font_family="Arial", layout="tile", anchor="top_left", opacity=0.48,
        width_ratio=0.32, rotation=-22.5, offset_x=0.13, offset_y=-0.04,
        gap_x=0.5, gap_y=1.2, include_datetime=True, datetime_text="2000-01-02 03:04:05",
    )
    target = tmp_path / "settings.json"
    AppSettingsIO.save(AppSettings(last_watermark=spec), target)
    restored = AppSettingsIO.load(target)
    assert restored.last_watermark == spec
    general = SettingsDialog(restored, document=None)
    assert general.app_settings().last_watermark == spec
    general.reject()
    general.deleteLater()
    assert AppSettings.from_dict({}).last_watermark is None


@pytest.mark.parametrize("invalid", ["invalid", {"opacity": float("nan")}, {"enabled": True, "text": ""}])
def test_invalid_remembered_watermark_does_not_reset_other_settings(invalid):
    restored = AppSettings.from_dict({"theme_mode": "light", "last_watermark": invalid})
    assert restored.theme_mode == "light"
    assert restored.last_watermark is None
    assert any(issue["kind"] == "last_watermark" for issue in restored.load_issues)


def test_new_image_refreshes_time_but_existing_watermark_remains_authoritative():
    saved = WatermarkSpec(enabled=True, text="上次水印", opacity=0.56, include_datetime=True, datetime_text="2000-01-02 03:04:05")
    blank = document()
    fresh = WatermarkDialog(blank, render(blank), default_spec=saved)
    spec = fresh.watermark()
    assert replace(spec, datetime_text=saved.datetime_text) == saved
    assert spec.datetime_text != saved.datetime_text
    assert not fresh.apply_to_all()
    fresh.reject()
    fresh.deleteLater()

    existing = document(replace(saved, text="文档自己的水印", enabled=False))
    dialog = WatermarkDialog(existing, render(blank), default_spec=replace(saved, opacity=0.8))
    assert dialog.watermark() == existing.watermark
    assert not dialog.enabled_check.isChecked()
    dialog.reject()
    dialog.deleteLater()


def test_logo_defaults_survive_reload_and_external_source_deletion(tmp_path):
    item = logo_document(tmp_path, layout="tile", rotation=45)
    target = tmp_path / "profile" / "settings.json"
    save_watermark_defaults(AppSettings(), item.watermark, item.watermark_assets, settings_path=target)
    (tmp_path / "标识.png").unlink()
    restored = AppSettingsIO.load(target)
    assets = load_watermark_default_assets(restored.last_watermark, settings_path=target)
    assert assets == item.watermark_assets
    blank = document()
    dialog = WatermarkDialog(blank, render(blank), default_spec=restored.last_watermark, default_assets=assets)
    assert dialog.watermark() == item.watermark
    assert render(dialog._draft) == render(item)
    assert blank.watermark is None and blank.watermark_assets == {}
    dialog._accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    dialog.deleteLater()


def test_failed_settings_write_keeps_previous_defaults_and_logo(tmp_path):
    first = logo_document(tmp_path)
    target = tmp_path / "profile" / "settings.json"
    save_watermark_defaults(AppSettings(), first.watermark, first.watermark_assets, settings_path=target)
    original = target.read_bytes()
    prior = AppSettingsIO.load(target)
    second = logo_document(tmp_path, logo_size=(20, 20))
    with patch("fdm.settings.atomic_write_json", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            save_watermark_defaults(prior, second.watermark, second.watermark_assets, settings_path=target)
    assert target.read_bytes() == original
    assert prior.last_watermark == first.watermark
    assert load_watermark_default_assets(prior.last_watermark, settings_path=target) == first.watermark_assets
    assert {p.name for p in (target.parent / "watermark-assets").iterdir()} == {f"{first.watermark.logo_sha256}.png"}


def test_replacing_remembered_logo_does_not_delete_document_assets(tmp_path):
    first = logo_document(tmp_path)
    target = tmp_path / "profile" / "settings.json"
    save_watermark_defaults(AppSettings(), first.watermark, first.watermark_assets, settings_path=target)
    prior = AppSettingsIO.load(target)
    second = logo_document(tmp_path, logo_size=(20, 20))
    expected = render(first)
    save_watermark_defaults(prior, second.watermark, second.watermark_assets, settings_path=target)
    assert {p.name for p in (target.parent / "watermark-assets").iterdir()} == {f"{second.watermark.logo_sha256}.png"}
    assert render(first) == expected
    assert load_watermark_default_assets(AppSettingsIO.load(target).last_watermark, settings_path=target) == second.watermark_assets


def test_missing_remembered_logo_preserves_parameters_and_requires_reselection(tmp_path):
    item = logo_document(tmp_path, opacity=0.7, layout="tile")
    target = tmp_path / "profile" / "settings.json"
    save_watermark_defaults(AppSettings(), item.watermark, item.watermark_assets, settings_path=target)
    (target.parent / "watermark-assets" / f"{item.watermark.logo_sha256}.png").unlink()
    restored = AppSettingsIO.load(target)
    with pytest.raises(OSError):
        load_watermark_default_assets(restored.last_watermark, settings_path=target)
    dialog = WatermarkDialog(document(), render(document()), default_spec=restored.last_watermark)
    assert dialog.watermark() == item.watermark
    assert "Logo" in dialog.hint.text()
    dialog._accept()
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("text"))
    dialog.text_edit.setPlainText("可改用文字")
    dialog._accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    dialog.deleteLater()


def test_apply_remembers_defaults_and_cancel_or_remove_keeps_them(window, monkeypatch):
    def accept(dialog):
        dialog.text_edit.setPlainText("本机记忆\n第二行")
        dialog.opacity_spin.setValue(62)
        dialog.all_check.setChecked(True)
        dialog._accept()
        return dialog.result()

    monkeypatch.setattr(WatermarkDialog, "exec", accept)
    window.edit_watermark()
    saved = window.current_document().watermark
    assert window._app_settings.last_watermark == saved
    assert AppSettingsIO.load().last_watermark == saved
    assert all(d.watermark == saved for d in window.project.documents)
    window.undo_current_document()
    assert all(d.watermark is None for d in window.project.documents)
    assert window._app_settings.last_watermark == saved
    on_disk = settings.settings_file_path().read_bytes()

    def cancel(dialog):
        assert dialog.watermark().text == saved.text
        assert dialog.watermark().opacity == saved.opacity
        assert not dialog.apply_to_all()
        dialog.text_edit.setPlainText("取消的编辑")
        dialog.reject()
        return dialog.result()

    monkeypatch.setattr(WatermarkDialog, "exec", cancel)
    window.edit_watermark()
    assert window.current_document().watermark is None
    assert window._app_settings.last_watermark == saved
    assert settings.settings_file_path().read_bytes() == on_disk
    window.redo_current_document()

    def remove(dialog):
        dialog._remove()
        return dialog.result()

    monkeypatch.setattr(WatermarkDialog, "exec", remove)
    window.edit_watermark()
    assert window.current_document().watermark is None
    assert window._app_settings.last_watermark == saved
    assert settings.settings_file_path().read_bytes() == on_disk


def test_failed_remember_does_not_undo_applied_watermark(window, monkeypatch):
    def accept(dialog):
        dialog._accept()
        return dialog.result()

    monkeypatch.setattr(WatermarkDialog, "exec", accept)
    with (
        patch("fdm.services.watermark_preferences.save_watermark_defaults", side_effect=OSError("disk full")),
        patch("fdm.ui.main_window.QMessageBox.warning") as warning,
    ):
        window.edit_watermark()
    assert window.current_document().watermark.text == "GTTC"
    assert window._app_settings.last_watermark is None
    assert AppSettingsIO.load().last_watermark is None
    assert "水印已应用" in warning.call_args.args[2]


def test_logo_is_restored_through_main_window_after_restart(window, tmp_path, monkeypatch):
    from fdm.ui.image_loader import ImageLoadRequest
    from fdm.ui.main_window import MainWindow

    logo = logo_document(tmp_path)
    source_path = tmp_path / "标识.png"

    def first_apply(dialog):
        dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("logo"))
        with patch("fdm.ui.watermark_dialog.QFileDialog.getOpenFileName", return_value=(str(source_path), "")):
            dialog._choose_logo()
        dialog._accept()
        return dialog.result()

    monkeypatch.setattr(WatermarkDialog, "exec", first_apply)
    window.edit_watermark()
    saved = window._app_settings.last_watermark
    assert saved.logo_sha256 == logo.watermark.logo_sha256
    source_path.unlink()
    window._confirm_close_documents = lambda documents: True
    window.close()

    restarted = MainWindow()
    try:
        blank = document()
        source = render(blank)
        restarted._add_loaded_document(ImageLoadRequest(path=blank.path, document=blank), source)
        assert blank.watermark is None

        def second_apply(dialog):
            spec = dialog.watermark()
            assert replace(spec, datetime_text=saved.datetime_text) == saved
            assert dialog.assets() == logo.watermark_assets
            assert dialog._draft.watermark_assets == logo.watermark_assets
            assert not dialog.apply_to_all()
            dialog._accept()
            return dialog.result()

        monkeypatch.setattr(WatermarkDialog, "exec", second_apply)
        restarted.edit_watermark()
        assert blank.watermark.logo_sha256 == saved.logo_sha256
        assert blank.watermark_assets == logo.watermark_assets
        assert restarted._app_settings.last_watermark == blank.watermark
    finally:
        restarted._confirm_close_documents = lambda documents: True
        restarted.close()


def test_digital_slide_does_not_open_or_replace_remembered_defaults(window):
    saved = WatermarkSpec(enabled=True, text="普通图片")
    window._app_settings.last_watermark = saved
    slide = document()
    slide.document_kind = "digital_slide"
    with patch.object(window, "current_document", return_value=slide), patch.object(WatermarkDialog, "exec") as show:
        window.edit_watermark()
    show.assert_not_called()
    assert window._app_settings.last_watermark == saved
    assert slide.watermark is None
