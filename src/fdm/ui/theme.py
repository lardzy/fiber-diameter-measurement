from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QColor, QFontDatabase, QFontInfo, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget

from fdm.settings import AppThemeMode, normalize_theme_mode


_SYSTEM_THEME_CACHE: dict[int, tuple[str, QPalette]] = {}
_THEME_CACHE_PROPERTY = "fdmAppliedThemeMode"
_THEME_STYLE_REVISION_PROPERTY = "fdmThemeStyleRevision"
_THEME_STYLE_REVISION = 6
_RESOLVED_THEME_PROPERTY = "fdmResolvedThemeMode"
_BASE_STYLE_PROPERTY = "fdmBaseWidgetStyle"


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
    if app.style().objectName().casefold() == token.casefold():
        app.setProperty(_BASE_STYLE_PROPERTY, token)
        return
    if (app.style().metaObject().className() == "QStyleSheetStyle"
            and str(app.property(_BASE_STYLE_PROPERTY) or "").casefold() == token.casefold()):
        return
    for candidate in QStyleFactory.keys():
        if candidate.casefold() == token.casefold():
            app.setStyle(candidate)
            app.setProperty(_BASE_STYLE_PROPERTY, candidate)
            return
    app.setStyle(token)
    app.setProperty(_BASE_STYLE_PROPERTY, token)


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


def build_navy_palette() -> QPalette:
    """A low-glare blue workspace; image pixels and category colors are untouched."""
    palette = build_dark_palette()
    overrides = {
        "Window": "#182638", "WindowText": "#E8F0F8", "Base": "#111D2D",
        "AlternateBase": "#203247", "Button": "#263B52", "ButtonText": "#E8F0F8",
        "Text": "#E8F0F8", "ToolTipBase": "#263B52", "ToolTipText": "#E8F0F8",
        "Highlight": "#80BAF0", "HighlightedText": "#10243A",
        "Link": "#91C9FF", "Light": "#425E79", "Midlight": "#2E4661",
        "Mid": "#506880", "Dark": "#0C1724", "Shadow": "#07101B",
        "PlaceholderText": "#A5B6C9",
    }
    for name, color in overrides.items():
        _set_role_color(palette, getattr(QPalette.ColorRole, name), color)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText, QPalette.ColorRole.PlaceholderText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#8194A9"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor("#405A74"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor("#CFDBE7"))
    return palette


def build_warm_palette() -> QPalette:
    palette = build_light_palette()
    overrides = {
        "Window": "#F3F0EA", "WindowText": "#342F29", "Base": "#FFFDFA",
        "AlternateBase": "#EAE5DC", "Button": "#F0EBE3", "ButtonText": "#342F29",
        "Text": "#342F29", "ToolTipBase": "#FFFDFA", "ToolTipText": "#342F29",
        "Highlight": "#805B36", "HighlightedText": "#FFFFFF",
        "Link": "#295D89", "Light": "#FFFDFA", "Midlight": "#E5DED3",
        "Mid": "#B4A99A", "Dark": "#C4B9AA", "Shadow": "#9A8E7F",
        "PlaceholderText": "#70665B",
    }
    for name, color in overrides.items():
        _set_role_color(palette, getattr(QPalette.ColorRole, name), color)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText, QPalette.ColorRole.PlaceholderText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#93897D"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor("#D4C3AF"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor("#514639"))
    return palette


def build_theme_palette(theme_mode: str) -> QPalette:
    """Build an explicit palette; resolve 'system' when applying to QApplication."""
    builders = {
        AppThemeMode.DARK: build_dark_palette,
        AppThemeMode.LIGHT: build_light_palette,
        AppThemeMode.NAVY: build_navy_palette,
        AppThemeMode.WARM: build_warm_palette,
    }
    return builders.get(normalize_theme_mode(theme_mode), build_dark_palette)()


@dataclass(frozen=True)
class ThemeColors:
    """UI semantic colors only. Never use these to recolor measurement pixels."""

    text: str
    muted: str
    disabled: str
    accent: str
    on_accent: str
    border: str
    info: str
    success: str
    warning: str
    error: str
    info_background: str
    success_background: str
    warning_background: str
    error_background: str
    canvas: str
    canvas_text: str


def semantic_colors(palette: QPalette | None = None) -> ThemeColors:
    """Resolve semantic tokens from a widget or application palette."""
    palette = palette or (QApplication.palette() if QApplication.instance() else build_dark_palette())
    dark = palette.color(QPalette.ColorRole.Window).lightnessF() < 0.5
    return ThemeColors(
        text=palette.color(QPalette.ColorRole.WindowText).name(),
        muted="#BAC6D2" if dark else "#52606D",
        disabled=palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText).name(),
        accent=palette.color(QPalette.ColorRole.Highlight).name(),
        on_accent=palette.color(QPalette.ColorRole.HighlightedText).name(),
        border=palette.color(QPalette.ColorRole.Mid).name(),
        info="#96CAFA" if dark else "#155B96",
        success="#8DDCB7" if dark else "#1C6A47",
        warning="#FFD185" if dark else "#825000",
        error="#FFAAA5" if dark else "#AB2925",
        info_background="#1B344A" if dark else "#E8F2FC",
        success_background="#193C30" if dark else "#E7F3EB",
        warning_background="#483A22" if dark else "#FFF0CD",
        error_background="#482B2D" if dark else "#FCE9E7",
        canvas="#101820" if dark else "#D6DEE7",
        canvas_text="#C3D0DE" if dark else "#35495C",
    )


def _system_color_mode(app: QApplication) -> str:
    scheme = app.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return AppThemeMode.DARK
    if scheme == Qt.ColorScheme.Light:
        return AppThemeMode.LIGHT
    _, palette = _ensure_system_theme_snapshot(app)
    return AppThemeMode.DARK if palette.color(QPalette.ColorRole.Window).lightness() < 128 else AppThemeMode.LIGHT


class _SystemThemeObserver(QObject):
    def __init__(self, app: QApplication):
        super().__init__(app)
        self._app = app
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._apply_system_colors)
        app.styleHints().colorSchemeChanged.connect(self._scheme_changed)

    def _scheme_changed(self, _scheme) -> None:
        if self._app.property(_THEME_CACHE_PROPERTY) == AppThemeMode.SYSTEM:
            # Qt emits colorSchemeChanged before installing the platform palette.
            # Apply our semantic colors after that transition has completed.
            self._timer.start(0)

    def _apply_system_colors(self) -> None:
        if self._app.property(_THEME_CACHE_PROPERTY) == AppThemeMode.SYSTEM:
            apply_application_theme(self._app, AppThemeMode.SYSTEM)


def apply_application_theme(app: QApplication, theme_mode: str | None) -> str:
    normalized = normalize_theme_mode(theme_mode)
    _ensure_system_theme_snapshot(app)
    resolved = _system_color_mode(app) if normalized == AppThemeMode.SYSTEM else normalized
    if not hasattr(app, "_fdm_system_theme_observer"):
        app._fdm_system_theme_observer = _SystemThemeObserver(app)
    if (
        app.property(_THEME_CACHE_PROPERTY) == normalized
        and app.property(_RESOLVED_THEME_PROPERTY) == resolved
        and app.property(_THEME_STYLE_REVISION_PROPERTY) == _THEME_STYLE_REVISION
    ):
        # Reapplying an identical application palette broadcasts expensive
        # PaletteChange/StyleChange events to every open widget.  MainWindow
        # construction and repeated Apply presses must therefore be idempotent.
        return normalized
    # Native controls can ignore palette roles and use different arrow metrics.
    # System mode follows the OS color scheme with the same geometry as our
    # explicit themes, including when the OS changes appearance while running.
    _set_style_by_name(app, "Fusion")
    app.setPalette(build_theme_palette(resolved))
    if not app.property("fdmThemeFontInitialized"):
        system_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
        resolved_family = QFontInfo(system_font).family()
        if resolved_family:
            system_font.setFamily(resolved_family)
        app.setFont(system_font)
        app.setProperty("fdmThemeFontInitialized", True)
    # Repeating QApplication.setFont clears widget-class fonts on macOS,
    # enlarging existing toolbar buttons even when the general font is equal.
    # A color-scheme change must preserve those established font metrics.
    app.setStyleSheet(build_application_stylesheet())
    app.setProperty(_THEME_CACHE_PROPERTY, normalized)
    app.setProperty(_RESOLVED_THEME_PROPERTY, resolved)
    app.setProperty(_THEME_STYLE_REVISION_PROPERTY, _THEME_STYLE_REVISION)
    return normalized


def build_application_stylesheet() -> str:
    """Stable, low-chrome control geometry shared by every color theme."""

    colors = semantic_colors()
    stylesheet = """
        QMainWindow, QDialog { background: palette(window); }
        QFrame#measurementContextBar, QFrame#captureTaskBar {
            background: palette(alternate-base);
            border-bottom: 1px solid palette(mid);
        }
        QFrame#imageNavigationGroup, QFrame#measurementCategoryGroup {
            background: palette(base);
            border: 1px solid palette(mid);
            border-radius: 6px;
        }
        QFrame#imageNavigationGroup QComboBox,
        QFrame#measurementCategoryGroup QComboBox {
            border: 0; background: transparent;
        }
        QLabel#measurementCategoryLabel { color: palette(placeholder-text); }
        QGroupBox#projectImages, QGroupBox#projectCategories {
            background: transparent; border: 0;
        }
        QTabWidget#projectNavigationTabs::pane {
            background: palette(window); border: 0;
        }
        QToolBar#measurementContextToolbar { padding: 0; spacing: 0; }
        QToolButton#persistentCalibrationButton[uncalibrated="true"] {
            background: @warning_background; color: @warning;
            border: 2px solid @warning; font-weight: 700;
            border-radius: 4px;
        }
        QToolButton#persistentCalibrationButton[uncalibrated="true"]:hover {
            background: @warning_background;
        }
        QToolButton#quickAreaOperationButton[magicPrompt="negative"] {
            background: @error_background; color: @error; border: 1px solid @error;
        }
        QFrame#currentMeasurementSummary {
            background: palette(base); border-bottom: 1px solid palette(mid);
        }
        QPushButton#currentObjectProperties:checked {
            background: palette(alternate-base); border-color: palette(highlight);
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
            border-color: palette(highlight);
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
        QPushButton:hover { border-color: palette(highlight); background: palette(midlight); }
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
            border-color: palette(highlight);
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
        QTabBar::tab {
            min-height: 27px; padding: 3px 10px;
            border: 0; border-bottom: 2px solid transparent;
            background: transparent; color: palette(window-text);
        }
        QTabBar::tab:hover { background: palette(midlight); }
        QTabBar::tab:selected {
            border-bottom: 2px solid palette(highlight); background: palette(alternate-base);
        }
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
        QPushButton:focus, QToolButton:focus {
            border: 1px solid palette(highlight);
        }
        QPushButton:checked {
            border-color: palette(highlight); background: palette(alternate-base);
        }
        QPushButton[emphasis="primary"], QToolButton[emphasis="primary"] {
            background: palette(highlight); color: palette(highlighted-text);
            border: 1px solid palette(highlight); font-weight: 600;
        }
        QPushButton:disabled, QToolButton:disabled {
            color: @disabled; background: palette(window); border-color: palette(mid);
        }
        QLabel[secondary="true"] { color: @muted; }
        QLabel[statusTone="info"] { color: @info; }
        QLabel[statusTone="success"] { color: @success; }
        QLabel[statusTone="warning"] { color: @warning; }
        QLabel[statusTone="error"] { color: @error; }
        QFrame[statusTone="info"] { background: @info_background; }
        QFrame[statusTone="success"] { background: @success_background; }
        QFrame[statusTone="warning"] { background: @warning_background; }
        QFrame[statusTone="error"] { background: @error_background; }
    """
    # Replace longer names first so @warning does not consume its background token.
    for name in sorted(ThemeColors.__dataclass_fields__, key=len, reverse=True):
        stylesheet = stylesheet.replace("@" + name, getattr(colors, name))
    return stylesheet


def refresh_widget_theme(widget: QWidget | None) -> None:
    if widget is None:
        return
    widgets = [widget, *widget.findChildren(QWidget)]
    for current in widgets:
        style = current.style()
        style.unpolish(current)
        style.polish(current)
        current.update()
