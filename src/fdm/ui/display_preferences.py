"""User-owned acquisition display preferences, separate from project records.

All acquisition pages share the display mode. View centers belong to a device
and frame geometry; a new camera must not inherit another camera's viewport.
The legacy measurement canvas's logical zoom and project view state are not
changed by this module. Its adapter is only for the temporary preview canvas.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager
import json
import math
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QPointF, QEvent, QTimer, Signal
from fdm.ui.acquisition_alignment import ALIGNMENTS, ALIGNMENT_FACTORS, validate_alignment


MODES = ("pixel", "fit", "percent", "magnification")
MAX_ENTERED_PERCENT = 1600.0


def _positive(value, label, maximum=100000.0):
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是正数")
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= maximum:
        raise ValueError(f"{label}必须大于零且不超过 {maximum:g}")
    return number


def _center(value, *, allow_outside_image=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("视图中心必须包含两个坐标")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("视图中心必须是有限数值")
    if allow_outside_image:
        # A small frame aligned to an edge has its viewport center outside the
        # image. Limit imported values to a sane Qt-sized coordinate namespace.
        if any(abs(item) > 16777215 for item in result):
            raise ValueError("视图中心超出有效范围")
        return result
    return tuple(max(0.0, min(1.0, item)) for item in result)


def viewport_key(source_id, frame_size):
    width, height = map(int, frame_size)
    if not source_id or min(width, height) <= 0:
        raise ValueError("保存视图需要设备标识和有效帧尺寸")
    return json.dumps([str(source_id), width, height], ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def preferences_path():
    from fdm.settings import settings_file_path
    return settings_file_path().parent / "acquisition-display.json"


@dataclass(frozen=True)
class AcquisitionDisplayPreferences:
    mode: str = "pixel"
    percent: float = 100.0
    magnification: float = 100.0
    alignment: str = "free"

    def aligning(self, alignment):
        return replace(self, alignment=validate_alignment(alignment))

    def at_canvas_percent(self, value):
        # A fitted small camera ROI can exceed the percentage entry dialog's
        # limit. A gesture adopts the displayed scale without rounding it.
        return replace(self, mode="percent", percent=_positive(value, "画面实际比例"))

    def selecting(self, mode, value=None):
        if mode not in MODES:
            raise ValueError("未知显示方式")
        values = {"mode": mode}
        if mode in ("percent", "magnification") and value is not None:
            values[mode] = _positive(value, "显示比例" if mode == "percent" else "物理倍率",
                                     MAX_ENTERED_PERCENT if mode == "percent" else 100000.0)
        return replace(self, **values)


class AcquisitionDisplayStore(QObject):
    changed = Signal()

    def __init__(self, path: Path | None = None, parent=None):
        super().__init__(parent)
        self.path = Path(path) if path is not None else preferences_path()
        self.preferences = AcquisitionDisplayPreferences()
        self._centers = {}
        self.load_warning = ""
        try:
            self._load()
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            # A damaged local preference must not block the device or project.
            self.preferences = AcquisitionDisplayPreferences()
            self._centers = {}
            self.load_warning = f"显示偏好未能读取，暂用实际像素：{exc}"

    def _load(self):
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("显示偏好格式不受支持")
        values = payload.get("display", {})
        if not isinstance(values, dict):
            raise ValueError("显示偏好格式无效")
        percent = _positive(values.get("percent", 100), "显示比例", MAX_ENTERED_PERCENT)
        if "canvas_percent" in values:
            percent = _positive(values["canvas_percent"], "画面实际比例")
        self.preferences = AcquisitionDisplayPreferences(
            percent=percent,
            magnification=_positive(values.get("magnification", 100), "物理倍率"),
        ).selecting(values.get("mode", "pixel")).aligning(values.get("alignment", "free"))
        centers = payload.get("centers", {})
        if not isinstance(centers, dict):
            raise ValueError("视图中心格式无效")
        for key, value in list(centers.items())[-128:]:
            try:
                source_id, width, height = json.loads(key)
                if viewport_key(source_id, (width, height)) == key:
                    self._centers[key] = _center(value,
                        allow_outside_image=payload.get("center_coordinates") == "viewport")
            except (ValueError, TypeError, OverflowError):
                continue

    def _save(self, preferences, centers):
        from fdm.atomic_io import atomic_write_text
        self.path.parent.mkdir(parents=True, exist_ok=True)
        display = {
            "mode": preferences.mode, "percent": min(preferences.percent, MAX_ENTERED_PERCENT),
            "magnification": preferences.magnification,
            "alignment": preferences.alignment,
        }
        if preferences.percent > MAX_ENTERED_PERCENT:
            # Keep the standard field readable by earlier versions.
            display["canvas_percent"] = preferences.percent
        atomic_write_text(self.path, json.dumps({"version": 1, "display": display,
            "center_coordinates": "viewport", "centers": centers}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def set_mode(self, mode, value=None):
        self.set_preferences(self.preferences.selecting(mode, value))

    def set_preferences(self, selected):
        if selected == self.preferences:
            return
        self._save(selected, self._centers)
        self.preferences = selected
        self.changed.emit()

    def center(self, source_id, frame_size):
        return self._centers.get(viewport_key(source_id, frame_size), (0.5, 0.5))

    def set_alignment(self, alignment):
        self.set_preferences(self.preferences.aligning(alignment))

    def set_center(self, source_id, frame_size, center, *, allow_outside_image=False):
        key, value = viewport_key(source_id, frame_size), _center(center, allow_outside_image=allow_outside_image)
        if self._centers.get(key) == value:
            return
        centers = dict(self._centers)
        centers.pop(key, None)
        centers[key] = value
        centers = dict(list(centers.items())[-128:])
        self._save(self.preferences, centers)
        self._centers = centers


def acquisition_display_store(host):
    store = getattr(host, "_acquisition_display_store", None)
    if store is None:
        store = AcquisitionDisplayStore(parent=host)
        host._acquisition_display_store = store
    return store


@dataclass(frozen=True)
class ResolvedDisplay:
    mode: str
    value: float
    scale: float | None
    screen_pixel_mm: float | None = None
    message: str = ""


def resolve_display(preferences, *, dpr=1.0, calibration=None, screen_pixel_mm=None):
    """Resolve requested mode without replacing a temporarily unavailable choice."""
    dpr = _positive(dpr, "显示器像素比")
    if preferences.mode == "fit":
        return ResolvedDisplay("fit", 100.0, None)
    if preferences.mode == "percent":
        return ResolvedDisplay("percent", preferences.percent, preferences.percent / 100 / dpr)
    if preferences.mode == "magnification":
        if calibration is None or not getattr(calibration, "isotropic", False):
            return ResolvedDisplay("pixel", 100.0, 1 / dpr, message="已记住物理倍率；选择等比例显微镜标尺后恢复，当前按实际像素显示。")
        if screen_pixel_mm is None or not math.isfinite(screen_pixel_mm) or screen_pixel_mm <= 0:
            return ResolvedDisplay("pixel", 100.0, 1 / dpr, message="已记住物理倍率；请校准当前显示器标线，当前按实际像素显示。")
        scale = calibration.x_um * preferences.magnification / (screen_pixel_mm * 1000 * dpr)
        return ResolvedDisplay("magnification", preferences.magnification, scale, screen_pixel_mm)
    return ResolvedDisplay("pixel", 100.0, 1 / dpr)


class AcquisitionDisplayBinding(QObject):
    """Bind a trial canvas or a temporary DocumentCanvas to the shared store.

    Call set_source after mounting a frame and close before disposing the page.
    For a trial canvas pass its QScrollArea; for DocumentCanvas omit it. The
    calibration provider returns a CalibrationSnapshot (or None).
    """

    applied = Signal(object)

    def __init__(self, canvas, *, store, scroll_area=None,
                 calibration_provider: Callable | None = None,
                 screen_pixel_mm_provider: Callable | None = None,
                 status_callback: Callable | None = None, parent=None):
        super().__init__(parent or canvas)
        self.canvas, self.store, self.scroll_area = canvas, store, scroll_area
        self._calibration_provider = calibration_provider or (lambda: None)
        self._screen_pixel_mm_provider = screen_pixel_mm_provider
        self._status_callback = status_callback or (lambda _message: None)
        self._source = None
        self._signature = None
        self._applied_signature = None
        self._applying = False
        self._closed = False
        self._pending_center = None
        self._screen_window = None
        self._last_message = ""
        self._mount_depth = 0
        self._user_scroll_pending = set()
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.timeout.connect(self.apply)
        self._center_timer = QTimer(self)
        self._center_timer.setSingleShot(True)
        self._center_timer.setInterval(200)
        self._center_timer.timeout.connect(self._flush_pending_center)
        self._restore_timer = QTimer(self)
        self._restore_timer.setSingleShot(True)
        self._restore_timer.timeout.connect(self._restore_center)
        self._restore_target = None
        canvas.installEventFilter(self)
        if scroll_area is not None:
            scroll_area.viewport().installEventFilter(self)
            for bar in (scroll_area.horizontalScrollBar(), scroll_area.verticalScrollBar()):
                bar.valueChanged.connect(lambda _value, bar=bar: self._scroll_changed(bar))
                bar.actionTriggered.connect(lambda _action, bar=bar: self._scroll_intent(bar))
                bar.sliderMoved.connect(lambda _value, bar=bar: self._scroll_intent(bar))
        else:
            canvas.viewTransformChanged.connect(self.capture_center)
            canvas.userPanned.connect(self._user_panned)
            canvas.set_acquisition_alignment(store.preferences.alignment)
        store.changed.connect(self._preferences_changed)

    def _preferences_changed(self):
        if not self._closed and self.canvas.isVisible():
            self._apply_timer.start(0)

    def _calibration_signature(self):
        calibration = self._calibration_provider()
        return tuple(getattr(calibration, key, None) for key in ("x_um", "y_um", "source"))

    def set_source(self, source_id, frame_size):
        source = (str(source_id), tuple(map(int, frame_size)))
        viewport_key(*source)
        signature = (source, self._calibration_signature())
        if signature == self._signature:
            return
        self._flush_pending_center()
        self._source, self._signature = source, signature
        self.apply()

    def clear_source(self):
        """Detach the current image before clearing or replacing its document.

        Preferences and the last viewport remain in the user store. A later
        frame with the same source and dimensions must still reapply them.
        """
        self.flush()
        self._apply_timer.stop()
        self._restore_timer.stop()
        self._source = None
        self._signature = None
        self._applied_signature = None
        self._restore_target = None
        self._applying = False

    reset_source = clear_source

    @contextmanager
    def mounting_frame(self):
        """Suppress default mount transforms while replacing a preview document.

        Use around DocumentCanvas.set_document()/initial fit, then call
        set_source(). Per-frame set_image() needs no guard.
        """
        self._mount_depth += 1
        try:
            yield
        finally:
            self._mount_depth -= 1

    def remember_document_zoom(self):
        """Explicitly adopt a user zoom gesture from the legacy preview canvas.

        Wire only the user's zoom command/gesture, not all transform signals:
        document mounting and resizing also emit them. Acquisition's 100%
        describes physical pixels, while DocumentCanvas stores logical zoom.
        """
        if self.scroll_area is not None or self._source is None or self._mount_depth:
            return
        from fdm.ui.view_transform import CanvasZoomMode
        if self.canvas.zoom_mode() is CanvasZoomMode.FIT:
            self.set_mode("fit")
        else:
            self._set_mode_preferences(self.store.preferences.at_canvas_percent(
                self.canvas.view_zoom() * self.canvas.devicePixelRatioF() * 100))

    def _resolve(self, preferences):
        from fdm.ui.live_display import screen_pixel_mm
        mm = (self._screen_pixel_mm_provider() if self._screen_pixel_mm_provider is not None
              else screen_pixel_mm(self.canvas.screen()))
        return resolve_display(preferences, dpr=self.canvas.devicePixelRatioF(),
            calibration=self._calibration_provider(), screen_pixel_mm=mm)

    def set_mode(self, mode, value=None):
        self._set_mode_preferences(self.store.preferences.selecting(mode, value))

    def _set_mode_preferences(self, preferences):
        self.capture_center()
        self._flush_pending_center()
        self._apply_preferences(preferences)
        try:
            self.store.set_preferences(preferences)
        except OSError:
            self._apply_preferences(self.store.preferences)
            raise

    def _validate_display_change(self):
        if self.scroll_area is not None:
            self.canvas.validate_display_change()

    def set_alignment(self, alignment):
        preferences = self.store.preferences.aligning(alignment)
        self._validate_display_change()
        # Save the free viewport before an anchor replaces it. Releasing via
        # the UI retains the currently visible position instead of jumping.
        self.capture_center(force=alignment == "free")
        self._flush_pending_center()
        self._apply_preferences(preferences)
        try:
            self.store.set_alignment(alignment)
        except OSError:
            self._apply_preferences(self.store.preferences)
            raise

    def _scroll_intent(self, bar):
        if self._applying or self._closed:
            return
        try:
            self._validate_display_change()
        except ValueError as exc:
            # actionTriggered runs before Qt commits sliderPosition to value.
            bar.setSliderPosition(bar.value())
            self._status_callback(str(exc))
            return
        self._user_scroll_pending.add(bar)
        QTimer.singleShot(0, self, lambda: self._user_scroll_pending.discard(bar))

    def _scroll_changed(self, bar):
        if self._closed or self._applying:
            return
        if bar in self._user_scroll_pending:
            self._user_scroll_pending.discard(bar)
            self._user_panned()
        self._applying = True
        try:
            # Cancelling a draft emits controller.changed synchronously. It
            # must not restore the old anchor halfway through this input.
            self.canvas.display_position_changed()
        finally:
            self._applying = False
        self.capture_center()

    def _user_panned(self):
        if self._closed or self._applying or self._mount_depth or self._source is None:
            return
        self._applying = True
        saved = False
        try:
            self._validate_display_change()
            # Commit before releasing the canvas anchor. On disk failure the
            # last confirmed preference remains the visible source of truth.
            preferences = self.store.preferences.aligning("free")
            if self.scroll_area is None and preferences.mode == "fit":
                # A fitted preview becomes a manually positioned view at the
                # current scale. Commit both changes together so a failed save
                # cannot leave free positioning paired with automatic fitting.
                preferences = preferences.at_canvas_percent(
                    self.canvas.view_zoom() * self.canvas.devicePixelRatioF() * 100)
            self.store.set_preferences(preferences)
            if self.scroll_area is None:
                self.canvas.set_acquisition_alignment("free")
            else:
                self.canvas.set_display_alignment("free")
            # The originating canvas is already in its user-chosen position;
            # only other pages need the shared preference's deferred apply.
            self._apply_timer.stop()
            self._applied_signature = None
            saved = True
        except (ValueError, OSError) as exc:
            self._status_callback(f"画面位置未能保存：{exc}")
            self._applied_signature = None
            try:
                self._apply_preferences(self.store.preferences)
            except ValueError:
                # A pending measurement may defer restoration until the
                # controller's next change; do not recurse into its guard.
                pass
        finally:
            self._applying = False
        if saved:
            self.capture_center(force=True)

    def apply(self):
        # Canvas setters cancel unfinished gestures and can synchronously emit
        # controller.changed. The workspace retries here on that signal.
        if self._closed or self._applying or not self.canvas.isVisible():
            return
        try:
            self._apply_preferences(self.store.preferences)
        except (ValueError, OSError) as exc:
            self._status_callback(str(exc))

    def _apply_preferences(self, preferences):
        if self._closed or self._source is None:
            return
        resolved = self._resolve(preferences)
        viewport = self.scroll_area.viewport() if self.scroll_area is not None else self.canvas
        center = self.store.center(*self._source)
        if self._pending_center is not None and self._pending_center[:2] == self._source:
            # Preserve the latest gesture while its debounced disk write is
            # pending, including synchronous controller refreshes.
            center = self._pending_center[2]
        signature = (preferences, self._source, resolved, self.canvas.devicePixelRatioF(),
                     viewport.size().toTuple(), center)
        if signature == self._applied_signature:
            return
        if resolved.scale is not None and max(self._source[1]) * resolved.scale > 16777215:
            raise ValueError("显示范围过大，请降低比例或倍率。")
        self._validate_display_change()
        self._applying = True
        try:
            if self.scroll_area is not None:
                self.canvas.set_display_alignment(preferences.alignment,
                    center=center if preferences.alignment == "free" else None)
                current = (self.canvas.display_mode, self.canvas.display_value, self.canvas.screen_pixel_mm)
                desired = (resolved.mode, resolved.value, resolved.screen_pixel_mm)
                if current != desired:
                    self.canvas.set_display_mode(*desired)
                else:
                    self.canvas._update_pixel_extent()
            else:
                self.canvas.set_acquisition_alignment(preferences.alignment)
                if resolved.mode == "fit":
                    self.canvas.fit_to_view()
                else:
                    # Only the acquisition preview uses physical-pixel semantics.
                    self.canvas.set_view_zoom(resolved.scale)
            self._restore_target = (preferences.mode, center, preferences.alignment)
            self._applied_signature = signature
            self._restore_timer.start(0)
        except Exception:
            self._applying = False
            raise
        if resolved.message != self._last_message:
            self._last_message = resolved.message
            self._status_callback(resolved.message)
        self.applied.emit(resolved)

    def _restore_center(self):
        if self._closed or self._source is None:
            self._applying = False
            return
        try:
            mode, center, alignment = self._restore_target
            if alignment != "free":
                if self.scroll_area is not None:
                    for bar, factor in zip((self.scroll_area.horizontalScrollBar(),
                            self.scroll_area.verticalScrollBar()), ALIGNMENT_FACTORS[alignment]):
                        bar.setValue(round(bar.minimum() + (bar.maximum() - bar.minimum()) * factor))
                else:
                    self.canvas.set_acquisition_alignment(alignment)
                return
            if mode == "fit":
                if self.scroll_area is not None:
                    self.scroll_area.horizontalScrollBar().setValue(0)
                    self.scroll_area.verticalScrollBar().setValue(0)
                return
            width, height = self._source[1]
            if self.scroll_area is not None:
                scale, offset = self.canvas._transform()
                viewport = self.scroll_area.viewport()
                for bar, value in zip((self.scroll_area.horizontalScrollBar(), self.scroll_area.verticalScrollBar()),
                        (center[0] * width * scale + offset[0] - viewport.width() / 2,
                         center[1] * height * scale + offset[1] - viewport.height() / 2)):
                    bar.setValue(round(max(bar.minimum(), min(bar.maximum(), value))))
            else:
                from fdm.geometry import Point
                self.canvas.center_on_image_point(Point(center[0] * width, center[1] * height))
        finally:
            self._applying = False

    def capture_center(self, *_, force=False):
        if (self._closed or self._applying or self._mount_depth or self._source is None
                or (not force and (self._apply_timer.isActive()
                    or self.store.preferences.mode == "fit"
                    or self.store.preferences.alignment != "free"))
                or not self.canvas.isVisible()):
            # A Resize/Show event schedules restoration before the canvas's
            # resize handler emits viewTransformChanged. Its temporary center
            # belongs to layout, not to a user's pan. In particular, navigation
            # hides docks before the old preview is detached; flush must not
            # persist that intermediate geometry as the next session's center.
            return
        width, height = self._source[1]
        if self.scroll_area is not None:
            frame = self.canvas.frame
            if frame is None or frame.source_id != self._source[0] or tuple(frame.size) != self._source[1]:
                return
            scale, offset = self.canvas._transform()
            if scale <= 0:
                return
            viewport = self.scroll_area.viewport()
            center = ((self.scroll_area.horizontalScrollBar().value() + viewport.width() / 2 - offset[0]) / scale / width,
                      (self.scroll_area.verticalScrollBar().value() + viewport.height() / 2 - offset[1]) / scale / height)
        else:
            snapshot = self.canvas.viewport_snapshot()
            if snapshot is None or (snapshot.full_image_rect.width(), snapshot.full_image_rect.height()) != (width, height):
                return
            point = self.canvas.widget_to_image(QPointF(self.canvas.width() / 2, self.canvas.height() / 2))
            center = (point.x / width, point.y / height)
        self._pending_center = (*self._source, _center(center, allow_outside_image=True))
        self._center_timer.start()

    def _flush_pending_center(self):
        self._center_timer.stop()
        if self._pending_center is not None:
            pending, self._pending_center = self._pending_center, None
            try:
                self.store.set_center(*pending, allow_outside_image=True)
            except OSError as exc:
                self._status_callback(f"显示位置未能保存：{exc}")

    def flush(self):
        self.capture_center()
        self._flush_pending_center()

    def eventFilter(self, watched, event):
        if self._closed:
            return False
        kind = event.type()
        if kind in (QEvent.Type.Show, QEvent.Type.Resize, QEvent.Type.DevicePixelRatioChange):
            if kind == QEvent.Type.Show:
                window = self.canvas.window().windowHandle()
                if window is not None and window is not self._screen_window:
                    self._screen_window = window
                    window.screenChanged.connect(self._preferences_changed)
            if not self._applying:
                # A hidden page may resize away and back to the same final
                # size. Its canvas transform still changed, so equality with
                # the previous size-based signature cannot skip restoration.
                self._applied_signature = None
                self._apply_timer.start(0)
        elif kind == QEvent.Type.Hide:
            self._flush_pending_center()
        return False

    def close(self):
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._apply_timer.stop()
        self._restore_timer.stop()
        self._center_timer.stop()
        self.canvas.removeEventFilter(self)
        if self.scroll_area is not None:
            self.scroll_area.viewport().removeEventFilter(self)
        else:
            self.canvas.userPanned.disconnect(self._user_panned)
            self.canvas.set_acquisition_alignment(None)
        self.store.changed.disconnect(self._preferences_changed)
