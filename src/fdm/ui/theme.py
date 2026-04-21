from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget

from fdm.settings import AppThemeMode, normalize_theme_mode


_SYSTEM_THEME_CACHE: dict[int, tuple[str, QPalette]] = {}


def _ensure_system_theme_snapshot(app: QApplication) -> tuple[str, QPalette]:
    cache_key = id(app)
    if cache_key not in _SYSTEM_THEME_CACHE:
        style_name = str(app.style().objectName() or "").strip()
        _SYSTEM_THEME_CACHE[cache_key] = (style_name, QPalette(app.palette()))
    return _SYSTEM_THEME_CACHE[cache_key]


def _set_style_by_name(app: QApplication, style_name: str) -> None:
    token = str(style_name or "").strip()
    if not token:
        return
    for candidate in QStyleFactory.keys():
        if candidate.casefold() == token.casefold():
            app.setStyle(candidate)
            return
    app.setStyle(token)


def _set_role_color(
    palette: QPalette,
    role: QPalette.ColorRole,
    active: str,
    *,
    disabled: str | None = None,
) -> None:
    active_color = QColor(active)
    palette.setColor(QPalette.ColorGroup.Active, role, active_color)
    palette.setColor(QPalette.ColorGroup.Inactive, role, active_color)
    palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(disabled or active))


def build_dark_palette() -> QPalette:
    palette = QPalette()
    _set_role_color(palette, QPalette.ColorRole.Window, "#252A31", disabled="#252A31")
    _set_role_color(palette, QPalette.ColorRole.WindowText, "#F3F4F6", disabled="#8B96A3")
    _set_role_color(palette, QPalette.ColorRole.Base, "#1B1F24", disabled="#171A1F")
    _set_role_color(palette, QPalette.ColorRole.AlternateBase, "#242A31", disabled="#20252B")
    _set_role_color(palette, QPalette.ColorRole.ToolTipBase, "#2C323A", disabled="#2C323A")
    _set_role_color(palette, QPalette.ColorRole.ToolTipText, "#F7F4EA", disabled="#B8C1CC")
    _set_role_color(palette, QPalette.ColorRole.Text, "#F3F4F6", disabled="#7F8A96")
    _set_role_color(palette, QPalette.ColorRole.Button, "#31363D", disabled="#2A2F35")
    _set_role_color(palette, QPalette.ColorRole.ButtonText, "#F3F4F6", disabled="#86919D")
    _set_role_color(palette, QPalette.ColorRole.BrightText, "#FF7B72", disabled="#FF7B72")
    _set_role_color(palette, QPalette.ColorRole.Highlight, "#2A9D8F", disabled="#476F69")
    _set_role_color(palette, QPalette.ColorRole.HighlightedText, "#08191C", disabled="#D3DBE3")
    _set_role_color(palette, QPalette.ColorRole.Link, "#79C0FF", disabled="#79C0FF")
    _set_role_color(palette, QPalette.ColorRole.LinkVisited, "#C9B3E5", disabled="#C9B3E5")
    _set_role_color(palette, QPalette.ColorRole.PlaceholderText, "#7B8794", disabled="#66707C")
    _set_role_color(palette, QPalette.ColorRole.Light, "#3A4148", disabled="#3A4148")
    _set_role_color(palette, QPalette.ColorRole.Midlight, "#343B43", disabled="#343B43")
    _set_role_color(palette, QPalette.ColorRole.Dark, "#13171C", disabled="#13171C")
    _set_role_color(palette, QPalette.ColorRole.Mid, "#252A31", disabled="#252A31")
    _set_role_color(palette, QPalette.ColorRole.Shadow, "#0B0E12", disabled="#0B0E12")
    return palette


def build_light_palette() -> QPalette:
    palette = QPalette()
    _set_role_color(palette, QPalette.ColorRole.Window, "#F5F7FA", disabled="#F5F7FA")
    _set_role_color(palette, QPalette.ColorRole.WindowText, "#1F2933", disabled="#7A8592")
    _set_role_color(palette, QPalette.ColorRole.Base, "#FFFFFF", disabled="#F1F4F8")
    _set_role_color(palette, QPalette.ColorRole.AlternateBase, "#EEF2F7", disabled="#E9EDF3")
    _set_role_color(palette, QPalette.ColorRole.ToolTipBase, "#FFFFFF", disabled="#FFFFFF")
    _set_role_color(palette, QPalette.ColorRole.ToolTipText, "#1F2933", disabled="#1F2933")
    _set_role_color(palette, QPalette.ColorRole.Text, "#182430", disabled="#8A94A1")
    _set_role_color(palette, QPalette.ColorRole.Button, "#F3F6FA", disabled="#ECEFF3")
    _set_role_color(palette, QPalette.ColorRole.ButtonText, "#1F2933", disabled="#8A94A1")
    _set_role_color(palette, QPalette.ColorRole.BrightText, "#C62828", disabled="#C62828")
    _set_role_color(palette, QPalette.ColorRole.Highlight, "#2A9D8F", disabled="#98D3CA")
    _set_role_color(palette, QPalette.ColorRole.HighlightedText, "#FFFFFF", disabled="#FFFFFF")
    _set_role_color(palette, QPalette.ColorRole.Link, "#1565C0", disabled="#1565C0")
    _set_role_color(palette, QPalette.ColorRole.LinkVisited, "#7A59A5", disabled="#7A59A5")
    _set_role_color(palette, QPalette.ColorRole.PlaceholderText, "#8A94A1", disabled="#A0A8B4")
    _set_role_color(palette, QPalette.ColorRole.Light, "#FFFFFF", disabled="#FFFFFF")
    _set_role_color(palette, QPalette.ColorRole.Midlight, "#E6EBF1", disabled="#E6EBF1")
    _set_role_color(palette, QPalette.ColorRole.Dark, "#CBD2D9", disabled="#CBD2D9")
    _set_role_color(palette, QPalette.ColorRole.Mid, "#D9E2EC", disabled="#D9E2EC")
    _set_role_color(palette, QPalette.ColorRole.Shadow, "#9AA5B1", disabled="#9AA5B1")
    return palette


def apply_application_theme(app: QApplication, theme_mode: str | None) -> str:
    normalized = normalize_theme_mode(theme_mode)
    system_style_name, system_palette = _ensure_system_theme_snapshot(app)

    if normalized == AppThemeMode.SYSTEM:
        _set_style_by_name(app, system_style_name)
        app.setPalette(QPalette(system_palette))
        return normalized

    _set_style_by_name(app, "Fusion")
    if normalized == AppThemeMode.LIGHT:
        app.setPalette(build_light_palette())
    else:
        app.setPalette(build_dark_palette())
    return normalized


def refresh_widget_theme(widget: QWidget | None) -> None:
    if widget is None:
        return
    widgets = [widget, *widget.findChildren(QWidget)]
    for current in widgets:
        style = current.style()
        style.unpolish(current)
        style.polish(current)
        current.update()
