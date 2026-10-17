from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSize
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication, QPushButton

from fdm.settings import AppSettings, AppThemeMode, normalize_theme_mode
from fdm.ui import icons, theme


MODES = (AppThemeMode.LIGHT, AppThemeMode.DARK, AppThemeMode.NAVY, AppThemeMode.WARM)


def _contrast(a: str, b: str) -> float:
    def luminance(value):
        color = QColor(value)
        channels = (color.redF(), color.greenF(), color.blueF())
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in channels]
        return sum(v * weight for v, weight in zip(linear, (.2126, .7152, .0722)))
    light, dark = sorted((luminance(a), luminance(b)), reverse=True)
    return (light + .05) / (dark + .05)


@pytest.fixture
def restore_theme(desktop_application):
    original = desktop_application.property("fdmAppliedThemeMode") or AppThemeMode.DARK
    yield desktop_application
    theme.apply_application_theme(desktop_application, original)
    desktop_application.processEvents()


@pytest.mark.parametrize("mode", (*MODES, AppThemeMode.SYSTEM))
def test_theme_preference_survives_settings_serialization(mode):
    assert normalize_theme_mode(mode.upper()) == mode
    source = AppSettings(theme_mode=mode)
    assert AppSettings.from_dict(source.to_dict()).theme_mode == mode


@pytest.mark.parametrize("mode", MODES)
def test_theme_text_status_and_focus_colors_have_readable_contrast(mode):
    palette = theme.build_theme_palette(mode)
    colors = theme.semantic_colors(palette)
    assert _contrast(colors.text, palette.color(QPalette.ColorRole.Window).name()) >= 4.5
    assert _contrast(colors.muted, palette.color(QPalette.ColorRole.Window).name()) >= 4.5
    assert _contrast(palette.color(QPalette.ColorRole.HighlightedText).name(), colors.accent) >= 4.5
    assert _contrast(colors.accent, palette.color(QPalette.ColorRole.Window).name()) >= 3
    for name in ("info", "success", "warning", "error"):
        assert _contrast(getattr(colors, name), getattr(colors, name + "_background")) >= 4.5


def test_theme_switch_keeps_control_geometry_and_is_idempotent(restore_theme):
    app = restore_theme
    button = QPushButton("开始采集")
    try:
        sizes = []
        for mode in MODES:
            theme.apply_application_theme(app, mode)
            app.processEvents()
            button.ensurePolished()
            sizes.append(button.sizeHint())
            assert app.property("fdmResolvedThemeMode") == mode
            assert "@warning" not in app.styleSheet()
            with patch.object(app, "setPalette") as set_palette:
                theme.apply_application_theme(app, mode)
                set_palette.assert_not_called()
        assert all(size == sizes[0] for size in sizes)
    finally:
        button.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_settings_offers_all_themes_and_keeps_old_ids(restore_theme):
    from fdm.ui.dialogs import SettingsDialog
    dialog = SettingsDialog(AppSettings(theme_mode=AppThemeMode.DARK), document=None)
    try:
        choice = dialog._theme_mode_combo
        assert choice.currentData() == "dark"
        assert {choice.itemData(index) for index in range(choice.count())} == {*MODES, "system"}
        for mode in MODES:
            choice.setCurrentIndex(choice.findData(mode))
            assert dialog.app_settings().theme_mode == mode
    finally:
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_bundled_fluent_assets_match_pinned_upstream_manifest():
    root = Path(icons.__file__).parent / "assets" / "fluent"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["revision"]) == 40
    assert "Microsoft Corporation" in (root / "LICENSE").read_text(encoding="utf-8")
    files = {entry["file"] for entry in manifest["files"]}
    assert {f"{name}_20_regular.svg" for name in icons.FLUENT_ICON_NAMES.values()} <= files
    for entry in manifest["files"]:
        data = (root / entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert b"<svg" in data


@pytest.mark.parametrize("dpr", (1, 1.25, 1.5, 2, 3))
def test_all_primary_icons_render_at_device_resolution_without_icon_font(dpr, monkeypatch):
    monkeypatch.setattr(icons, "qta", None)
    for name in icons.FLUENT_ICON_NAMES:
        icon = icons.themed_icon(name, color="#336699")
        assert not icon.isNull(), name
        pixmap = icon.pixmap(QSize(20, 20), dpr, QIcon.Mode.Normal)
        assert pixmap.width() == round(20 * dpr), name
        assert pixmap.devicePixelRatio() == dpr, name
        image = pixmap.toImage()
        assert any(image.pixelColor(x, y).alpha() for x in range(image.width()) for y in range(image.height())), name


@pytest.mark.parametrize("mode", MODES)
def test_icon_disabled_and_selected_states_follow_theme(mode, restore_theme):
    theme.apply_application_theme(restore_theme, mode)
    icon = icons.themed_icon("save_project")
    normal = icon.pixmap(QSize(24, 24), 2, QIcon.Mode.Normal).toImage()
    disabled = icon.pixmap(QSize(24, 24), 2, QIcon.Mode.Disabled).toImage()
    selected = icon.pixmap(QSize(24, 24), 2, QIcon.Mode.Selected).toImage()
    assert normal != disabled
    assert normal != selected
    expected = restore_theme.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText)
    solid_pixels = [disabled.pixelColor(x, y) for x in range(disabled.width()) for y in range(disabled.height())
                    if disabled.pixelColor(x, y).alpha() == 255]
    assert solid_pixels and all(color == expected for color in solid_pixels)


def test_geometric_fallback_is_also_dpi_aware(monkeypatch):
    monkeypatch.setattr(icons, "qta", None)
    icon = icons.themed_icon("continuous_manual", color="#336699")
    pixmap = icon.pixmap(QSize(20, 20), 2)
    assert not pixmap.isNull() and pixmap.width() == 40 and pixmap.devicePixelRatio() == 2


def test_pixel_size_and_fullscreen_commands_have_distinct_icons():
    actual = icons.themed_icon("actual_size").pixmap(24, 24).toImage()
    fullscreen = icons.themed_icon("fullscreen").pixmap(24, 24).toImage()
    assert actual != fullscreen


def test_qt_67_physical_size_contract_does_not_double_scale(monkeypatch):
    monkeypatch.setattr(icons, "_SCALED_PIXMAP_SIZE_IS_PHYSICAL", True)
    engine = icons._VectorIconEngine("#336699", svg=icons._fluent_svg("save"))
    pixmap = engine.scaledPixmap(QSize(40, 40), QIcon.Mode.Normal, QIcon.State.Off, 2)
    assert pixmap.width() == 40 and pixmap.devicePixelRatio() == 2


def test_fluent_assets_are_collected_for_frozen_distribution():
    hooks = pytest.importorskip("PyInstaller.utils.hooks")
    entries = hooks.collect_data_files("fdm.ui", includes=["assets/fluent/*"])
    names = {Path(source).name for source, _target in entries}
    assert {"LICENSE", "NOTICE", "manifest.json"} <= names
    assert {f"{name}_20_regular.svg" for name in icons.FLUENT_ICON_NAMES.values()} <= names
