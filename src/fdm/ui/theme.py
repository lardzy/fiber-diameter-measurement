from __future__ import annotations

from PySide6.QtGui import QColor, QFontDatabase, QFontInfo, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget

from fdm.settings import AppThemeMode, normalize_theme_mode


_SYSTEM_THEME_CACHE: dict[int, tuple[str, QPalette]] = {}
_THEME_CACHE_PROPERTY = "fdmAppliedThemeMode"
_THEME_STYLE_REVISION_PROPERTY = "fdmThemeStyleRevision"
_THEME_STYLE_REVISION = 4


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
    # `Mid` is the shared outline token for inputs, tables and workbench
    # cards.  It must remain visibly distinct from both Window and Base.
    _set_role_color(palette, QPalette.ColorRole.Mid, "#46515C", disabled="#343B43")
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
    _set_role_color(palette, QPalette.ColorRole.Highlight, "#197C70", disabled="#98D3CA")
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
    if (
        app.property(_THEME_CACHE_PROPERTY) == normalized
        and app.property(_THEME_STYLE_REVISION_PROPERTY) == _THEME_STYLE_REVISION
    ):
        # Reapplying an identical application palette broadcasts expensive
        # PaletteChange/StyleChange events to every open widget.  MainWindow
        # construction and repeated Apply presses must therefore be idempotent.
        return normalized
    system_style_name, system_palette = _ensure_system_theme_snapshot(app)

    if normalized == AppThemeMode.SYSTEM:
        _set_style_by_name(app, system_style_name)
        app.setPalette(QPalette(system_palette))
    else:
        _set_style_by_name(app, "Fusion")
        if normalized == AppThemeMode.LIGHT:
            app.setPalette(build_light_palette())
        else:
            app.setPalette(build_dark_palette())
    system_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    resolved_family = QFontInfo(system_font).family()
    if resolved_family:
        system_font.setFamily(resolved_family)
    app.setFont(system_font)
    app.setStyleSheet(build_application_stylesheet())
    app.setProperty(_THEME_CACHE_PROPERTY, normalized)
    app.setProperty(_THEME_STYLE_REVISION_PROPERTY, _THEME_STYLE_REVISION)
    return normalized


def build_application_stylesheet() -> str:
    """Stable, low-chrome control geometry shared by every color theme."""

    return """
        QMainWindow, QDialog { background: palette(window); }
        QFrame#measurementContextBar, QFrame#captureTaskBar {
            background: palette(alternate-base);
            border-bottom: 1px solid palette(mid);
        }
        QToolBar#measurementContextToolbar { padding: 0; spacing: 0; }
        QToolButton#persistentCalibrationButton[uncalibrated="true"] {
            background: #FFF0CD; color: #713F12;
            border: 2px solid #B45309; font-weight: 700;
            border-radius: 4px;
        }
        QToolButton#persistentCalibrationButton[uncalibrated="true"]:hover {
            background: #FFE4A6;
        }
        QToolButton#quickAreaOperationButton[magicPrompt="negative"] {
            background: #FEE2E2; color: #7F1D1D; border: 1px solid #DC2626;
        }
        QFrame#currentMeasurementSummary {
            background: palette(base); border-bottom: 1px solid palette(mid);
        }
        QLabel#currentMeasurementValue { font-size: 18px; font-weight: 600; }
        QLabel#welcomeTitle { font-size: 24px; font-weight: 600; padding: 12px; }
        QLabel#welcomeHint { color: palette(window-text); padding: 12px; }
        QToolBar {
            spacing: 4px;
            padding: 2px 6px;
            border: 0;
            border-bottom: 1px solid palette(mid);
            background: palette(window);
        }
        QToolButton {
            min-height: 30px;
            padding: 2px 8px;
            border: 1px solid transparent;
            border-radius: 6px;
        }
        QToolButton:hover { background: palette(midlight); }
        QToolButton:pressed { background: palette(mid); }
        QToolButton:checked {
            border-color: #2A9D8F;
            background: palette(highlight);
            color: palette(highlighted-text);
        }
        QToolButton[workspaceTab="true"]:checked {
            background: palette(alternate-base); color: palette(window-text);
            border: 0; border-bottom: 2px solid palette(highlight); border-radius: 2px;
        }
        QToolButton[panelToggle="true"]:checked {
            background: palette(alternate-base); color: palette(window-text);
            border: 1px solid palette(mid); border-radius: 4px;
        }
        QPushButton {
            min-height: 30px;
            padding: 2px 10px;
            border: 1px solid palette(mid);
            border-radius: 6px;
            background: palette(button);
        }
        QPushButton:hover { border-color: #2A9D8F; background: palette(midlight); }
        QPushButton:pressed { background: palette(mid); }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            min-height: 28px;
            padding: 1px 7px;
            border: 1px solid palette(mid);
            border-radius: 5px;
            background: palette(base);
            selection-background-color: palette(highlight);
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border-color: #2A9D8F;
        }
        QMenu {
            padding: 5px;
            border: 1px solid palette(mid);
            border-radius: 6px;
            background: palette(base);
        }
        QMenu::item { min-height: 28px; padding: 3px 24px 3px 10px; border-radius: 4px; }
        QMenu::item:selected { background: palette(highlight); color: palette(highlighted-text); }
        QGroupBox {
            margin-top: 14px;
            padding-top: 7px;
            font-weight: 600;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 7px; padding: 0 3px; }
        QDockWidget::title {
            min-height: 26px;
            padding-left: 8px;
            border-bottom: 1px solid palette(mid);
            background: palette(alternate-base);
            font-weight: 600;
        }
        QTabBar::tab { min-height: 27px; padding: 3px 10px; }
        QTabBar::tab:selected { border-bottom: 2px solid #2A9D8F; }
        QHeaderView::section {
            padding: 5px 7px;
            border: 0;
            border-right: 1px solid palette(mid);
            border-bottom: 1px solid palette(mid);
            background: palette(alternate-base);
            font-weight: 600;
        }
        QTableView, QTableWidget {
            border: 1px solid palette(mid);
            border-radius: 4px;
            background: palette(base);
            alternate-background-color: palette(alternate-base);
            gridline-color: palette(mid);
        }
    """


def refresh_widget_theme(widget: QWidget | None) -> None:
    if widget is None:
        return
    widgets = [widget, *widget.findChildren(QWidget)]
    for current in widgets:
        style = current.style()
        style.unpolish(current)
        style.polish(current)
        current.update()
