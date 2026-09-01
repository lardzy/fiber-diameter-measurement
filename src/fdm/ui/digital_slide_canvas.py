from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from time import perf_counter
from weakref import ref

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget
from shiboken6 import isValid as is_qobject_valid

from fdm.geometry import Point
from fdm.models import ImageDocument
from fdm.runtime_logging import append_runtime_log
from fdm.services.digital_slide_renderer import (
    DigitalSlideDerivedCache,
    DigitalSlideRenderFailure,
    DigitalSlideRenderFrame,
    DigitalSlideRenderRequest,
    DigitalSlideRenderer,
    DigitalSlideRendererStats,
)
from fdm.services.digital_slide_store import (
    DIGITAL_SLIDE_OVERVIEW_MAX_EDGE,
    DigitalSlideManifest,
    DigitalSlideStore,
)
from fdm.settings import AppSettings, digital_slide_render_cache_directory
from fdm.ui.canvas import (
    DocumentCanvas,
    canvas_image_border,
    canvas_workspace_foreground,
)
from fdm.ui.canvas_overlay_cache import CanvasOverlayTileKey
from fdm.ui.view_transform import CanvasZoomMode


_OVERVIEW_MAX_EDGE = DIGITAL_SLIDE_OVERVIEW_MAX_EDGE
_OVERVIEW_CACHE_LIMIT = 3
_OVERVIEW_FOCUS_DEBOUNCE_MS = 180
_BROWSE_VIEW_VERSION = 1
_VIEW_MARGIN = 20.0
_PIXEL_WORK_EPSILON = 1.0e-9
_NATIVE_VIEWPORT_INDICATOR_MS = 1200


@dataclass(frozen=True, slots=True)
class DigitalSlideBrowseView:
    center_px: Point
    zoom: float
    mode: CanvasZoomMode


class DigitalSlideCanvas(DocumentCanvas):
    viewportChanged = Signal(int, int, int)
    focusChanged = Signal(int)
    navigationModeChanged = Signal(str)
    shiftNavigationEnabledChanged = Signal(bool)
    viewportBufferFailed = Signal(str)
    overviewImageChanged = Signal(QImage)
    pixelWorkAvailabilityChanged = Signal(bool, str)
    browseNoticeRequested = Signal(str)
    _renderFrameReady = Signal(object)
    _renderFrameFailed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._slide_store: DigitalSlideStore | None = None
        self._slide_manifest: DigitalSlideManifest | None = None
        self._viewport_origin = Point(0.0, 0.0)
        self._browse_center = Point(0.0, 0.0)
        self._browse_view_restored = False
        self._render_frame: DigitalSlideRenderFrame | None = None
        self._previous_render_frame: DigitalSlideRenderFrame | None = None
        # Paint-only handoff during focus changes.  These frames never become
        # the authoritative ``_image`` consumed by pixel algorithms.
        self._focus_transition_frame: DigitalSlideRenderFrame | None = None
        self._focus_transition_image: QImage | None = None
        self._focus_transition_native_rect: QRectF | None = None
        self._renderer: DigitalSlideRenderer | None = None
        self._render_request_id = 0
        self._latest_display_request_id = 0
        self._latest_native_request_id = 0
        self._latest_overview_request_id = 0
        self._native_frame_key: tuple[int, int, int] | None = None
        self._native_frame_pending_key: tuple[int, int, int] | None = None
        self._pixel_work_enabled = True
        self._pixel_work_reason = ""
        self._navigation_velocity = Point(0.0, 0.0)
        self._last_navigation_origin = Point(0.0, 0.0)
        self._last_navigation_at = 0.0
        self._focus_index = 0
        self._initial_fit_pending = False
        self._initial_fit_done = False
        self._initial_fit_attempts = 0
        self._navigation_mode = "smooth"
        self._shift_navigation_enabled = False
        self._smooth_nav_keys: set[int] = set()
        self._smooth_nav_keyboard_shift = False
        self._smooth_nav_last_at = 0.0
        self._smooth_nav_timer = QTimer(self)
        self._smooth_nav_timer.setInterval(16)
        self._smooth_nav_timer.timeout.connect(self._apply_smooth_navigation)
        self._viewport_buffer_error_blocked = False
        self._viewport_buffer_last_error = ""
        self._viewport_last_publish_at = 0.0
        self._overview_image = QImage()
        self._overview_focus_index = -1
        self._overview_enabled = False
        self._dynamic_focus_overview_enabled = True
        self._overview_cache: OrderedDict[int, QImage] = OrderedDict()
        self._overview_failed_focuses: set[int] = set()
        self._overview_pending = False
        self._overview_debounce_timer = QTimer(self)
        self._overview_debounce_timer.setSingleShot(True)
        self._overview_debounce_timer.setInterval(_OVERVIEW_FOCUS_DEBOUNCE_MS)
        self._overview_debounce_timer.timeout.connect(self.request_overview)
        self._native_viewport_indicator_visible = False
        self._native_viewport_indicator_timer = QTimer(self)
        self._native_viewport_indicator_timer.setSingleShot(True)
        self._native_viewport_indicator_timer.setInterval(
            _NATIVE_VIEWPORT_INDICATOR_MS
        )
        self._native_viewport_indicator_timer.timeout.connect(
            self._hide_native_viewport_indicator
        )
        # Pointer drags are constrained to the raster that is currently
        # mounted in the canvas.  Keep this separate from the whole-slide
        # coordinate system used by navigation and programmatic locating.
        self._clamp_pointer_to_mounted_viewport = False
        self._renderFrameReady.connect(self._on_render_frame_ready)
        self._renderFrameFailed.connect(self._on_render_frame_failed)

    def set_slide_document(self, document: ImageDocument, store: DigitalSlideStore) -> None:
        renderer = self._renderer
        self._renderer = None
        if renderer is not None:
            renderer.close()
        self._native_viewport_indicator_timer.stop()
        self._native_viewport_indicator_visible = False
        self._overview_debounce_timer.stop()
        self._latest_overview_request_id += 1
        self._overview_pending = False
        self._overview_image = QImage()
        self._overview_focus_index = -1
        self._overview_cache.clear()
        self._overview_failed_focuses.clear()
        self.overviewImageChanged.emit(QImage())
        self._slide_store = store
        self._slide_manifest = store.read_manifest()
        document.image_size = (self._slide_manifest.width, self._slide_manifest.height)
        slide_meta = dict(document.metadata.get("digital_slide", {})) if isinstance(document.metadata.get("digital_slide"), dict) else {}
        origin = slide_meta.get("viewport_origin")
        if isinstance(origin, (list, tuple)) and len(origin) >= 2:
            try:
                origin_x = float(origin[0])
                origin_y = float(origin[1])
            except (TypeError, ValueError):
                origin_x = 0.0
                origin_y = 0.0
            if math.isfinite(origin_x) and math.isfinite(origin_y):
                self._viewport_origin = Point(origin_x, origin_y)
        if "focus_index" in slide_meta:
            try:
                focus_index = int(slide_meta.get("focus_index", 0) or 0)
            except (TypeError, ValueError):
                focus_index = 0
        else:
            focus_index = max(0, len(self._slide_manifest.focus_levels) // 2)
        self._focus_index = self._normalized_focus_index(focus_index)
        self._browse_view_restored = False
        browse_payload = slide_meta.get("browse_view")
        try:
            browse_version = (
                int(browse_payload.get("version", 0) or 0)
                if isinstance(browse_payload, dict)
                else 0
            )
        except (TypeError, ValueError):
            browse_version = 0
        if isinstance(browse_payload, dict) and browse_version == _BROWSE_VIEW_VERSION:
            center = browse_payload.get("center")
            try:
                zoom = float(browse_payload.get("zoom", 0.0) or 0.0)
                mode = CanvasZoomMode(str(browse_payload.get("mode", CanvasZoomMode.CUSTOM.value)))
                center_x = float(center[0])
                center_y = float(center[1])
            except (IndexError, TypeError, ValueError):
                zoom = 0.0
                mode = CanvasZoomMode.CUSTOM
                center_x = math.nan
                center_y = math.nan
            if (
                isinstance(center, (list, tuple))
                and len(center) >= 2
                and math.isfinite(zoom)
                and math.isfinite(center_x)
                and math.isfinite(center_y)
                and zoom > 0.0
            ):
                self._browse_center = Point(center_x, center_y)
                self._zoom = zoom
                self._zoom_mode = mode
                self._browse_view_restored = True
        if not self._browse_view_restored:
            self._browse_center = Point(
                self._viewport_origin.x + self._slide_manifest.viewport_width / 2.0,
                self._viewport_origin.y + self._slide_manifest.viewport_height / 2.0,
            )
        self._initial_fit_done = False
        self._invalidate_viewport_buffer()
        image = QImage(
            int(self._slide_manifest.viewport_width),
            int(self._slide_manifest.viewport_height),
            QImage.Format.Format_RGB32,
        )
        image.fill(QColor("#101820"))
        super().set_document(document, image)
        document.image_size = (self._slide_manifest.width, self._slide_manifest.height)
        if self._browse_view_restored:
            browse_payload = slide_meta.get("browse_view")
            assert isinstance(browse_payload, dict)
            self._zoom = max(
                self._whole_slide_fit_zoom(),
                min(40.0, float(browse_payload.get("zoom", 1.0) or 1.0)),
            )
            try:
                self._zoom_mode = CanvasZoomMode(
                    str(browse_payload.get("mode", CanvasZoomMode.CUSTOM.value))
                )
            except ValueError:
                self._zoom_mode = CanvasZoomMode.CUSTOM
        else:
            # ``DocumentCanvas.set_document()`` restores the legacy image zoom.
            # Establish a native-field camera scale before clamping the derived
            # legacy center, otherwise a small slide in a large widget can be
            # recentered to the whole-slide midpoint before the deferred fit.
            self._zoom = self._native_field_fit_zoom()
            self._zoom_mode = CanvasZoomMode.NATIVE_FIELD_FIT
        self._clamp_viewport()
        self._clamp_browse_center()
        self._sync_pan_from_browse_center()
        self._update_native_viewport_origin()
        self._native_frame_key = None
        self._native_frame_pending_key = None
        self._start_renderer()
        self._request_display_frame()
        self._update_pixel_work_state()
        if self._browse_view_restored:
            self._initial_fit_done = True
            self._publish_view_transform(zoom_changed=True)
        else:
            self.schedule_initial_fit()
        if self._overview_enabled and self.isVisible():
            QTimer.singleShot(0, self.request_overview)

    def shutdown(self) -> None:
        """Detach long-lived slide resources before the Qt widget is deleted."""
        self._smooth_nav_keys.clear()
        self._smooth_nav_keyboard_shift = False
        self._smooth_nav_timer.stop()
        self._overview_debounce_timer.stop()
        self._native_viewport_indicator_timer.stop()
        self._native_viewport_indicator_visible = False
        self._latest_overview_request_id += 1
        self._overview_pending = False
        self._overview_enabled = False
        self._cancel_overlay_requests()
        self._stop_renderer()
        self._invalidate_viewport_buffer()
        self._overview_image = QImage()
        self._overview_focus_index = -1
        self._overview_cache.clear()
        self._overview_failed_focuses.clear()
        self._render_frame = None
        self._previous_render_frame = None
        self._focus_transition_frame = None
        self._focus_transition_image = None
        self._focus_transition_native_rect = None
        self._native_frame_key = None
        self._native_frame_pending_key = None
        self._slide_store = None
        self._slide_manifest = None

    def hideEvent(self, event) -> None:
        """Cancel navigation/buffer work when another document tab takes over."""

        self._smooth_nav_keys.clear()
        self._smooth_nav_timer.stop()
        self._smooth_nav_last_at = 0.0
        self._overview_debounce_timer.stop()
        self._native_viewport_indicator_timer.stop()
        self._native_viewport_indicator_visible = False
        self._latest_overview_request_id += 1
        self._overview_pending = False
        self._stop_renderer()
        self._invalidate_viewport_buffer()
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._allow_viewport_buffer_retry()
        if self._initial_fit_pending:
            self._apply_initial_fit()
        self._start_renderer()
        self._request_display_frame()
        self._request_native_frame()
        self._update_pixel_work_state()
        self.request_overview()

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._publish_view_transform()
        self.update()

    def set_settings(self, settings: AppSettings) -> None:
        previous_cache_gib = int(
            getattr(self._settings, "digital_slide_render_cache_gib", 2) or 0
        )
        super().set_settings(settings)
        current_cache_gib = int(
            getattr(settings, "digital_slide_render_cache_gib", 2) or 0
        )
        if (
            self._slide_store is not None
            and current_cache_gib != previous_cache_gib
        ):
            self._stop_renderer()
            self._start_renderer()
            self._request_display_frame()
            self._request_native_frame()

    def browse_view(self) -> DigitalSlideBrowseView:
        return DigitalSlideBrowseView(
            center_px=Point(self._browse_center.x, self._browse_center.y),
            zoom=float(self._zoom),
            mode=self._zoom_mode,
        )

    def view_zoom(self) -> float:
        return max(1.0e-9, min(40.0, float(self._zoom)))

    def set_browse_view(self, view: DigitalSlideBrowseView) -> bool:
        return self._apply_browse_view(
            center=Point(float(view.center_px.x), float(view.center_px.y)),
            zoom=float(view.zoom),
            mode=view.mode,
            zoom_changed=not math.isclose(float(view.zoom), self._zoom),
        )

    def visible_slide_rect(self) -> QRectF:
        if self._slide_manifest is None:
            return QRectF()
        return self._source_view_rect().intersected(
            QRectF(
                0.0,
                0.0,
                float(self._slide_manifest.width),
                float(self._slide_manifest.height),
            )
        )

    def native_viewport_rect(self) -> QRectF:
        if self._slide_manifest is None:
            return QRectF()
        return QRectF(
            float(self._viewport_origin.x),
            float(self._viewport_origin.y),
            float(self._slide_manifest.viewport_width),
            float(self._slide_manifest.viewport_height),
        )

    def pixel_work_enabled(self) -> bool:
        return bool(self._pixel_work_enabled)

    def pixel_work_unavailable_reason(self) -> str:
        return self._pixel_work_reason

    def large_area_browse_active(self) -> bool:
        """Return whether the camera is below the native pixel-work scale."""

        return bool(
            self._slide_manifest is not None
            and self._zoom + _PIXEL_WORK_EPSILON < self._native_field_fit_zoom()
        )

    def native_viewport_indicator_visible(self) -> bool:
        return bool(self._native_viewport_indicator_visible)

    def renderer_stats(self) -> DigitalSlideRendererStats | None:
        renderer = self._renderer
        return renderer.stats() if renderer is not None else None

    def clear_render_cache(self) -> None:
        renderer = self._renderer
        self._stop_renderer()
        if renderer is not None:
            renderer.clear_derived_cache()
        elif self._slide_store is not None and self._slide_manifest is not None:
            cache = DigitalSlideDerivedCache(
                digital_slide_render_cache_directory(), byte_limit=0
            )
            cache.clear_fingerprint(
                cache.source_fingerprint(
                    self._slide_store.path,
                    self._slide_manifest,
                    source_identity=self._render_source_identity(),
                )
            )
        self._allow_viewport_buffer_retry()
        self._start_renderer()
        self._request_display_frame()
        self._request_native_frame()

    def viewport_snapshot(self):
        snapshot = super().viewport_snapshot()
        if snapshot is None:
            return None
        return type(snapshot)(
            document_id=snapshot.document_id,
            full_image_rect=snapshot.full_image_rect,
            mounted_image_rect=self.native_viewport_rect(),
            visible_image_rect=self.visible_slide_rect(),
            zoom=snapshot.zoom,
            mode=snapshot.mode,
            device_pixel_ratio=snapshot.device_pixel_ratio,
            focus_index=snapshot.focus_index,
            native_viewport_rect=self.native_viewport_rect(),
            pixel_work_enabled=self._pixel_work_enabled,
        )

    def _content_rect(self) -> QRectF:
        if self._slide_manifest is None:
            return QRectF()
        available = QRectF(
            _VIEW_MARGIN,
            _VIEW_MARGIN,
            max(1.0, float(self.width()) - (_VIEW_MARGIN * 2.0)),
            max(1.0, float(self.height()) - (_VIEW_MARGIN * 2.0)),
        )
        native_aspect = float(self._slide_manifest.viewport_width) / max(
            1.0, float(self._slide_manifest.viewport_height)
        )
        if available.width() / max(available.height(), 1.0) > native_aspect:
            width = available.height() * native_aspect
            return QRectF(
                available.center().x() - width / 2.0,
                available.top(),
                width,
                available.height(),
            )
        height = available.width() / max(native_aspect, 1.0e-12)
        return QRectF(
            available.left(),
            available.center().y() - height / 2.0,
            available.width(),
            height,
        )

    def _native_field_fit_zoom(self) -> float:
        if self._slide_manifest is None:
            return 1.0
        content = self._content_rect()
        return min(
            content.width() / max(1.0, float(self._slide_manifest.viewport_width)),
            content.height() / max(1.0, float(self._slide_manifest.viewport_height)),
            40.0,
        )

    def _whole_slide_fit_zoom(self) -> float:
        if self._slide_manifest is None:
            return 1.0
        content = self._content_rect()
        return min(
            40.0,
            max(
                1.0e-9,
                min(
                    content.width() / max(1.0, float(self._slide_manifest.width)),
                    content.height() / max(1.0, float(self._slide_manifest.height)),
                ),
            ),
        )

    def _source_view_rect(self) -> QRectF:
        content = self._content_rect()
        if content.isEmpty() or self._zoom <= 0.0:
            return QRectF()
        width = content.width() / self._zoom
        height = content.height() / self._zoom
        return QRectF(
            self._browse_center.x - width / 2.0,
            self._browse_center.y - height / 2.0,
            width,
            height,
        )

    def _clamp_browse_center(self) -> None:
        if self._slide_manifest is None:
            return
        source = self._source_view_rect()
        half_width = source.width() / 2.0
        half_height = source.height() / 2.0
        slide_width = float(self._slide_manifest.width)
        slide_height = float(self._slide_manifest.height)
        center_x = (
            slide_width / 2.0
            if source.width() >= slide_width
            else max(half_width, min(slide_width - half_width, self._browse_center.x))
        )
        center_y = (
            slide_height / 2.0
            if source.height() >= slide_height
            else max(half_height, min(slide_height - half_height, self._browse_center.y))
        )
        self._browse_center = Point(center_x, center_y)

    def _sync_pan_from_browse_center(self) -> None:
        content = self._content_rect()
        self._pan = Point(
            content.center().x() - self._browse_center.x * self._zoom,
            content.center().y() - self._browse_center.y * self._zoom,
        )

    def _update_native_viewport_origin(self) -> None:
        if self._slide_manifest is None:
            return
        self._viewport_origin = Point(
            round(
                self._browse_center.x
                - float(self._slide_manifest.viewport_width) / 2.0
            ),
            round(
                self._browse_center.y
                - float(self._slide_manifest.viewport_height) / 2.0
            ),
        )
        self._clamp_viewport()

    def _native_request_key(self) -> tuple[int, int, int]:
        return (
            int(round(self._viewport_origin.x)),
            int(round(self._viewport_origin.y)),
            int(self._focus_index),
        )

    def _apply_browse_view(
        self,
        *,
        center: Point,
        zoom: float,
        mode: CanvasZoomMode,
        zoom_changed: bool,
    ) -> bool:
        if self._slide_manifest is None:
            return False
        if not (
            math.isfinite(float(center.x))
            and math.isfinite(float(center.y))
            and math.isfinite(float(zoom))
        ):
            return False
        if not isinstance(mode, CanvasZoomMode):
            try:
                mode = CanvasZoomMode(str(mode))
            except ValueError:
                mode = CanvasZoomMode.CUSTOM
        previous_zoom = float(self._zoom)
        minimum = self._whole_slide_fit_zoom()
        requested = max(minimum, min(40.0, float(zoom)))
        threshold = self._native_field_fit_zoom()
        if (
            requested + _PIXEL_WORK_EPSILON < threshold
            and self._zoom + _PIXEL_WORK_EPSILON >= threshold
            and (
                self._has_pointer_edit_operation()
                or self.has_pending_path_drawing()
                or self.has_magic_segment_session()
                or self.has_reference_instance_session()
                or self.has_fiber_quick_session()
            )
        ):
            self.browseNoticeRequested.emit(
                "当前正在绘制或编辑，请先完成或取消操作后再缩小到大范围浏览。"
            )
            return False
        self._hovered_line_endpoint = None
        self._hovered_construction_id = None
        self._hovered_construction_handle = None
        self._set_active_snap_candidate(None)
        self._zoom = requested
        self._zoom_mode = mode
        self._browse_center = Point(float(center.x), float(center.y))
        self._clamp_browse_center()
        self._sync_pan_from_browse_center()
        previous_key = self._native_request_key()
        self._update_native_viewport_origin()
        if self._native_request_key() != previous_key:
            self._native_frame_pending_key = None
        self._reset_proxy_warming()
        self._cancel_overlay_requests()
        self._persist_view_state()
        self._request_display_frame()
        self._request_native_frame()
        self._update_pixel_work_state()
        effective_zoom_change = bool(
            zoom_changed
            and not math.isclose(
                previous_zoom,
                requested,
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            )
        )
        if effective_zoom_change:
            self._show_native_viewport_indicator()
        self._publish_viewport_state(
            throttled=False,
            zoom_changed=effective_zoom_change,
        )
        self.update()
        return True

    def _show_native_viewport_indicator(self) -> None:
        self._native_viewport_indicator_visible = True
        self._native_viewport_indicator_timer.start()

    def _hide_native_viewport_indicator(self) -> None:
        if not self._native_viewport_indicator_visible:
            return
        self._native_viewport_indicator_visible = False
        self.update()

    def _start_renderer(self) -> None:
        if self._renderer is not None or self._slide_store is None or self._slide_manifest is None:
            return
        canvas_ref = ref(self)

        def publish_result(frame: DigitalSlideRenderFrame) -> None:
            canvas = canvas_ref()
            if canvas is not None and is_qobject_valid(canvas):
                canvas._renderFrameReady.emit(frame)

        def publish_failure(failure: DigitalSlideRenderFailure) -> None:
            canvas = canvas_ref()
            if canvas is not None and is_qobject_valid(canvas):
                canvas._renderFrameFailed.emit(failure)

        cache_gib = max(
            0,
            min(
                32,
                int(
                    getattr(
                        self._settings,
                        "digital_slide_render_cache_gib",
                        2,
                    )
                    or 0
                ),
            ),
        )
        self._renderer = DigitalSlideRenderer(
            self._slide_store.path,
            self._slide_manifest,
            source_identity=self._render_source_identity(),
            cache_root=digital_slide_render_cache_directory(),
            disk_cache_bytes=cache_gib * 1024 * 1024 * 1024,
            result_callback=publish_result,
            failure_callback=publish_failure,
        )

    def _render_source_identity(self) -> str | None:
        document = self._document
        if document is None:
            return None
        return str(document.absolute_path or document.path or "") or None

    def _stop_renderer(self) -> None:
        renderer = self._renderer
        self._renderer = None
        self._latest_display_request_id = 0
        self._latest_native_request_id = 0
        self._latest_overview_request_id = 0
        self._native_frame_pending_key = None
        if renderer is not None:
            renderer.close()

    def _blend_width(self) -> int:
        if self._slide_manifest is None or not isinstance(self._slide_manifest.metadata, dict):
            return 0
        try:
            return max(0, int(self._slide_manifest.metadata.get("blend_width", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _request_display_frame(self) -> None:
        if (
            self._viewport_buffer_error_blocked
            or self._slide_manifest is None
            or not self.isVisible()
        ):
            return
        self._start_renderer()
        renderer = self._renderer
        content = self._content_rect()
        source = self._source_view_rect()
        if renderer is None or content.isEmpty() or source.isEmpty():
            return
        dpr = max(1.0, float(self.devicePixelRatioF()))
        self._render_request_id += 1
        self._latest_display_request_id = self._render_request_id
        renderer.submit(
            DigitalSlideRenderRequest(
                request_id=self._render_request_id,
                purpose="display",
                source_rect=(source.x(), source.y(), source.width(), source.height()),
                output_size_px=(
                    max(1, int(round(content.width() * dpr))),
                    max(1, int(round(content.height() * dpr))),
                ),
                focus_index=int(self._focus_index),
                device_pixel_ratio=dpr,
                blend_width=self._blend_width(),
                velocity_px_per_second=(
                    float(self._navigation_velocity.x),
                    float(self._navigation_velocity.y),
                ),
            )
        )

    def _request_native_frame(self) -> None:
        if (
            self._viewport_buffer_error_blocked
            or self._slide_manifest is None
            or not self.isVisible()
        ):
            return
        if self._zoom + _PIXEL_WORK_EPSILON < self._native_field_fit_zoom():
            self._native_frame_pending_key = None
            return
        key = self._native_request_key()
        if self._native_frame_key == key or self._native_frame_pending_key == key:
            return
        self._start_renderer()
        renderer = self._renderer
        if renderer is None:
            return
        self._render_request_id += 1
        self._latest_native_request_id = self._render_request_id
        self._native_frame_pending_key = key
        renderer.submit(
            DigitalSlideRenderRequest(
                request_id=self._render_request_id,
                purpose="native",
                source_rect=(
                    float(key[0]),
                    float(key[1]),
                    float(self._slide_manifest.viewport_width),
                    float(self._slide_manifest.viewport_height),
                ),
                output_size_px=(
                    int(self._slide_manifest.viewport_width),
                    int(self._slide_manifest.viewport_height),
                ),
                focus_index=int(self._focus_index),
                device_pixel_ratio=1.0,
                blend_width=self._blend_width(),
                force_lod=0,
            )
        )

    def _on_render_frame_ready(self, frame: DigitalSlideRenderFrame) -> None:
        if frame.purpose == "overview":
            if frame.request_id != self._latest_overview_request_id:
                return
            target_focus = self._overview_target_focus_index()
            if frame.focus_index != target_focus or frame.image.isNull():
                return
            self._overview_pending = False
            self._overview_cache[target_focus] = frame.image
            self._overview_cache.move_to_end(target_focus)
            while len(self._overview_cache) > _OVERVIEW_CACHE_LIMIT:
                self._overview_cache.popitem(last=False)
            self._overview_image = frame.image
            self._overview_focus_index = target_focus
            self._overview_failed_focuses.discard(target_focus)
            self.overviewImageChanged.emit(frame.image)
            self.update()
            return
        if frame.focus_index != self._focus_index:
            return
        if frame.purpose == "display":
            if frame.request_id != self._latest_display_request_id:
                return
            self._previous_render_frame = self._render_frame
            self._render_frame = frame
            self._focus_transition_frame = None
            self._focus_transition_image = None
            self._focus_transition_native_rect = None
            self._viewport_buffer_error_blocked = False
            self._viewport_buffer_last_error = ""
            self.update()
            return
        if frame.request_id != self._latest_native_request_id:
            return
        key = (
            int(round(frame.source_rect[0])),
            int(round(frame.source_rect[1])),
            int(frame.focus_index),
        )
        if key != self._native_request_key():
            return
        self._image = frame.image
        self._native_frame_key = key
        self._native_frame_pending_key = None
        self._focus_transition_frame = None
        self._focus_transition_image = None
        self._focus_transition_native_rect = None
        self._viewport_buffer_error_blocked = False
        self._viewport_buffer_last_error = ""
        self._update_pixel_work_state()
        self._publish_view_transform()
        self.update()

    def _on_render_frame_failed(self, failure: DigitalSlideRenderFailure) -> None:
        if failure.purpose == "overview":
            if failure.focus_index != self._overview_target_focus_index():
                return
        elif failure.focus_index != self._focus_index:
            return
        expected = (
            self._latest_native_request_id
            if failure.purpose == "native"
            else (
                self._latest_overview_request_id
                if failure.purpose == "overview"
                else self._latest_display_request_id
            )
        )
        if failure.request_id != expected:
            return
        if failure.purpose == "native":
            self._native_frame_pending_key = None
            self._native_frame_key = None
        elif failure.purpose == "overview":
            self._overview_pending = False
            self._overview_failed_focuses.add(failure.focus_index)
            self._overview_image = QImage()
            self._overview_focus_index = -1
            self.overviewImageChanged.emit(QImage())
            append_runtime_log(
                "数字切片导航缩略图读取失败",
                (
                    f"path={self._slide_store.path if self._slide_store is not None else ''}\n"
                    f"request_id={failure.request_id}, focus_index={failure.focus_index}\n"
                    f"error={failure.message}"
                ),
            )
            return
        self._focus_transition_frame = None
        self._focus_transition_image = None
        self._focus_transition_native_rect = None
        self._viewport_buffer_error_blocked = True
        self._viewport_buffer_last_error = failure.message
        append_runtime_log(
            "数字切片异步渲染失败",
            (
                f"path={self._slide_store.path if self._slide_store is not None else ''}\n"
                f"request_id={failure.request_id}, purpose={failure.purpose}, "
                f"focus_index={failure.focus_index}\nerror={failure.message}"
            ),
        )
        self.viewportBufferFailed.emit(failure.message)
        self._update_pixel_work_state()

    def _update_pixel_work_state(self) -> None:
        if self._slide_manifest is None:
            enabled = False
            reason = "未加载数字切片"
        elif self._zoom + _PIXEL_WORK_EPSILON < self._native_field_fit_zoom():
            enabled = False
            reason = "大范围浏览模式：放大到单视场后可测量"
        elif self._native_frame_key != self._native_request_key():
            enabled = False
            reason = (
                "原生工作视场读取失败，请重试"
                if self._viewport_buffer_error_blocked
                else "正在加载原生工作视场，完成后可测量"
            )
        else:
            enabled = True
            reason = ""
        changed = enabled != self._pixel_work_enabled or reason != self._pixel_work_reason
        self._pixel_work_enabled = enabled
        self._pixel_work_reason = reason
        if self._read_only != (not enabled):
            self.set_read_only(not enabled)
        if changed:
            self.pixelWorkAvailabilityChanged.emit(enabled, reason)
            self._publish_view_transform()

    def focus_index(self) -> int:
        return self._focus_index

    def viewport_origin(self) -> Point:
        return Point(self._viewport_origin.x, self._viewport_origin.y)

    def mounted_image_origin(self) -> Point:
        return self.viewport_origin()

    def navigation_mode(self) -> str:
        return self._navigation_mode

    def navigation_mode_label(self) -> str:
        if self._navigation_mode == "smooth":
            return "平滑移动（快速）" if self._shift_navigation_enabled else "平滑移动"
        return "步进移动（整视场）" if self._shift_navigation_enabled else "步进移动"

    def shift_navigation_enabled(self) -> bool:
        return self._shift_navigation_enabled

    def overview_image(self) -> QImage:
        return QImage(self._overview_image)

    def set_overview_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._overview_enabled:
            return
        self._overview_enabled = enabled
        if not enabled:
            self._overview_debounce_timer.stop()
            self._latest_overview_request_id += 1
            self._overview_pending = False
            return
        if self.isVisible():
            QTimer.singleShot(0, self.request_overview)

    def set_dynamic_focus_overview_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._dynamic_focus_overview_enabled:
            return
        old_target = self._overview_target_focus_index()
        self._dynamic_focus_overview_enabled = enabled
        new_target = self._overview_target_focus_index()
        if old_target == new_target:
            return
        self._overview_debounce_timer.stop()
        self._latest_overview_request_id += 1
        self._overview_pending = False
        cached = self._overview_cache.get(new_target)
        if cached is not None:
            self._overview_cache.move_to_end(new_target)
            self._overview_image = cached
            self._overview_focus_index = new_target
            self.overviewImageChanged.emit(cached)
            return
        self._overview_image = QImage()
        self._overview_focus_index = -1
        self.overviewImageChanged.emit(QImage())
        if self._overview_enabled and self.isVisible():
            QTimer.singleShot(0, self.request_overview)

    def dynamic_focus_overview_enabled(self) -> bool:
        return self._dynamic_focus_overview_enabled

    def _overview_target_focus_index(self) -> int:
        if self._dynamic_focus_overview_enabled or self._slide_manifest is None:
            return int(self._focus_index)
        levels = self._slide_manifest.focus_levels
        if not levels:
            return 0
        return max(0, len(levels) // 2)

    def is_navigation_key_active(self, key: int | Qt.Key) -> bool:
        return int(key) in self._smooth_nav_keys

    def set_navigation_mode(self, mode: str) -> None:
        mode = "smooth" if mode == "smooth" else "step"
        if mode == self._navigation_mode:
            return
        navigation_was_active = bool(self._smooth_nav_keys)
        if navigation_was_active:
            self._cancel_overlay_requests()
        self._navigation_mode = mode
        self._smooth_nav_keys.clear()
        self._smooth_nav_keyboard_shift = False
        self._smooth_nav_timer.stop()
        self._smooth_nav_last_at = 0.0
        if navigation_was_active:
            self._publish_viewport_state(throttled=False)
            self._request_viewport_buffer()
            self.update()
        self.navigationModeChanged.emit(mode)

    def toggle_navigation_mode(self) -> str:
        self.set_navigation_mode("smooth" if self._navigation_mode != "smooth" else "step")
        return self._navigation_mode

    def set_shift_navigation_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._shift_navigation_enabled:
            return
        self._shift_navigation_enabled = enabled
        self.shiftNavigationEnabledChanged.emit(enabled)

    def move_viewport_by(self, dx: float, dy: float, *, throttled: bool = False) -> None:
        if self._slide_manifest is None or (not dx and not dy):
            return
        old_center = Point(self._browse_center.x, self._browse_center.y)
        self._browse_center = Point(old_center.x + float(dx), old_center.y + float(dy))
        self._clamp_browse_center()
        if (
            math.isclose(old_center.x, self._browse_center.x, abs_tol=1.0e-9)
            and math.isclose(old_center.y, self._browse_center.y, abs_tol=1.0e-9)
        ):
            self._navigation_velocity = Point(0.0, 0.0)
            return
        self._set_hovered_line_endpoint(None)
        self._hovered_construction_id = None
        self._hovered_construction_handle = None
        self._set_active_snap_candidate(None)
        if self._construction_session is not None:
            self._construction_session.hover_point = None
        self._cancel_overlay_requests()
        now = perf_counter()
        elapsed = max(1.0e-3, now - self._last_navigation_at) if self._last_navigation_at else 0.0
        if elapsed > 0.0:
            self._navigation_velocity = Point(
                (self._browse_center.x - old_center.x) / elapsed,
                (self._browse_center.y - old_center.y) / elapsed,
            )
        self._last_navigation_at = now
        self._last_navigation_origin = Point(self._browse_center.x, self._browse_center.y)
        self._sync_pan_from_browse_center()
        self._update_native_viewport_origin()
        if self._native_frame_key != self._native_request_key():
            self._native_frame_pending_key = None
        self._persist_view_state()
        self._request_display_frame()
        self._request_native_frame()
        self._update_pixel_work_state()
        self._publish_viewport_state(throttled=throttled)
        self.update()

    def center_on_image_point(self, point: Point) -> None:
        if self._slide_manifest is None:
            super().center_on_image_point(point)
            return
        self._apply_browse_view(
            center=Point(float(point.x), float(point.y)),
            zoom=self._zoom,
            mode=self._zoom_mode,
            zoom_changed=False,
        )

    def set_focus_index(self, focus_index: int) -> None:
        focus_index = self._normalized_focus_index(focus_index)
        if focus_index == self._focus_index:
            return
        if (
            self._render_frame is not None
            and self._render_frame.focus_index == self._focus_index
        ):
            self._focus_transition_frame = self._render_frame
        if (
            self._image is not None
            and not self._image.isNull()
            and self._native_frame_key is not None
        ):
            self._focus_transition_image = self._image
            self._focus_transition_native_rect = self.native_viewport_rect()
        self._focus_index = focus_index
        self.focusChanged.emit(focus_index)
        self._render_frame = None
        self._previous_render_frame = None
        self._native_frame_key = None
        self._native_frame_pending_key = None
        self._allow_viewport_buffer_retry()
        self._request_display_frame()
        self._request_native_frame()
        self._update_pixel_work_state()
        self._publish_viewport_state(throttled=False)
        self.update()
        if not self._overview_enabled:
            self._overview_image = QImage()
            self._overview_focus_index = -1
            self.overviewImageChanged.emit(QImage())
            return
        overview_focus_index = self._overview_target_focus_index()
        if not self._dynamic_focus_overview_enabled:
            cached = self._overview_cache.get(overview_focus_index)
            if cached is not None:
                self._overview_cache.move_to_end(overview_focus_index)
                self._overview_image = cached
                self._overview_focus_index = overview_focus_index
                self.overviewImageChanged.emit(cached)
            elif (
                not self._overview_pending
                and overview_focus_index not in self._overview_failed_focuses
            ):
                QTimer.singleShot(0, self.request_overview)
            return
        self._latest_overview_request_id += 1
        cached = self._overview_cache.get(overview_focus_index)
        if cached is not None:
            self._overview_cache.move_to_end(overview_focus_index)
            self._overview_image = cached
            self._overview_focus_index = overview_focus_index
            self.overviewImageChanged.emit(cached)
            return
        if overview_focus_index in self._overview_failed_focuses:
            self._overview_image = QImage()
            self._overview_focus_index = -1
            self.overviewImageChanged.emit(QImage())
            return
        self._overview_debounce_timer.start()

    def widget_to_image(self, position: QPointF) -> Point:
        point = Point(
            (position.x() - self._pan.x) / max(self._zoom, 1.0e-12),
            (position.y() - self._pan.y) / max(self._zoom, 1.0e-12),
        )
        if self._clamp_pointer_to_mounted_viewport:
            return self._clamp_to_mounted_viewport(point)
        return point

    def image_to_widget(self, point: Point) -> QPointF:
        return QPointF(
            self._pan.x + (point.x * self._zoom),
            self._pan.y + (point.y * self._zoom),
        )

    def _overlay_widget_origin(self) -> QPointF:
        return QPointF(float(self._pan.x), float(self._pan.y))

    def _paint_image_bounds(self) -> QRectF:
        if self._slide_manifest is None:
            return QRectF()
        return QRectF(
            0.0,
            0.0,
            float(self._slide_manifest.width),
            float(self._slide_manifest.height),
        )

    def _exact_visible_image_rect(self) -> QRectF:
        return self.visible_slide_rect()

    def visible_source_pixel_rect(self) -> tuple[float, float, float, float] | None:
        native = self.native_viewport_rect()
        if native.isEmpty():
            return None
        return native.x(), native.y(), native.width(), native.height()

    def _base_image_target_rect(self) -> QRectF:
        frame = self._render_frame
        if frame is None:
            return QRectF()
        x, y, width, height = frame.source_rect
        top_left = self.image_to_widget(Point(x, y))
        return QRectF(
            top_left.x(),
            top_left.y(),
            width * self._zoom,
            height * self._zoom,
        )

    def _draw_base_image(self, painter: QPainter) -> QRectF:
        content = self._content_rect()
        if self._slide_manifest is None or content.isEmpty():
            return QRectF()
        painter.save()
        painter.setClipRect(content)
        full_target_top_left = self.image_to_widget(Point(0.0, 0.0))
        full_target = QRectF(
            full_target_top_left.x(),
            full_target_top_left.y(),
            float(self._slide_manifest.width) * self._zoom,
            float(self._slide_manifest.height) * self._zoom,
        )
        if (
            not self._overview_image.isNull()
            and self._overview_focus_index == self._focus_index
        ):
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(full_target, self._overview_image)
        frame = self._render_frame
        if frame is None or frame.focus_index != self._focus_index:
            frame = self._focus_transition_frame
        target = QRectF()
        if frame is not None:
            x, y, width, height = frame.source_rect
            top_left = self.image_to_widget(Point(x, y))
            target = QRectF(
                top_left.x(),
                top_left.y(),
                width * self._zoom,
                height * self._zoom,
            )
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(target, frame.image)
        elif (
            self._focus_transition_image is not None
            and not self._focus_transition_image.isNull()
            and self._focus_transition_native_rect is not None
        ):
            native = self._focus_transition_native_rect
            top_left = self.image_to_widget(Point(native.x(), native.y()))
            target = QRectF(
                top_left.x(),
                top_left.y(),
                native.width() * self._zoom,
                native.height() * self._zoom,
            )
            painter.drawImage(target, self._focus_transition_image)
        elif self._image is not None and self._native_frame_key is not None:
            native = self.native_viewport_rect()
            top_left = self.image_to_widget(Point(native.x(), native.y()))
            target = QRectF(
                top_left.x(),
                top_left.y(),
                native.width() * self._zoom,
                native.height() * self._zoom,
            )
            painter.drawImage(target, self._image)
        painter.restore()

        painter.save()
        border_pen = QPen(canvas_image_border(self.palette()))
        border_pen.setWidthF(1.0)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(full_target)
        painter.restore()
        return target if not target.isEmpty() else full_target

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._slide_manifest is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._native_viewport_indicator_visible:
            native = self.native_viewport_rect()
            top_left = self.image_to_widget(Point(native.x(), native.y()))
            native_target = QRectF(
                top_left.x(),
                top_left.y(),
                native.width() * self._zoom,
                native.height() * self._zoom,
            )
            pen = QPen(QColor("#F4C95D"))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidthF(1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(native_target)
        if self._pixel_work_enabled or not self.large_area_browse_active():
            painter.end()
            return
        message = self._pixel_work_reason or "放大到单视场后可测量"
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(message).adjusted(-10, -6, 10, 6)
        text_rect.moveCenter(
            QPointF(self._content_rect().center().x(), self._content_rect().bottom() - 22.0).toPoint()
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(12, 20, 28, 210))
        painter.drawRoundedRect(QRectF(text_rect), 5.0, 5.0)
        painter.setPen(canvas_workspace_foreground(self.palette()))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, message)

    def resizeEvent(self, event) -> None:
        QWidget.resizeEvent(self, event)
        if self._slide_manifest is None:
            return
        if self._zoom_mode is CanvasZoomMode.FIT:
            self.fit_to_view()
            return
        if self._zoom_mode is CanvasZoomMode.NATIVE_FIELD_FIT:
            self.fit_native_viewport()
            return
        self._zoom = max(self._whole_slide_fit_zoom(), min(40.0, self._zoom))
        self._clamp_browse_center()
        self._sync_pan_from_browse_center()
        self._update_native_viewport_origin()
        self._persist_view_state()
        self._request_display_frame()
        self._request_native_frame()
        self._update_pixel_work_state()
        self._publish_view_transform(zoom_changed=True)
        self.update()

    def _viewport_focus_index(self) -> int | None:
        return int(self._focus_index)

    def fit_to_view(self) -> None:
        self._initial_fit_pending = False
        if self._slide_manifest is None:
            return
        self._initial_fit_done = True
        self._apply_browse_view(
            center=Point(
                float(self._slide_manifest.width) / 2.0,
                float(self._slide_manifest.height) / 2.0,
            ),
            zoom=self._whole_slide_fit_zoom(),
            mode=CanvasZoomMode.FIT,
            zoom_changed=True,
        )

    def fit_native_viewport(self) -> None:
        self._initial_fit_pending = False
        if self._slide_manifest is None:
            return
        self._initial_fit_done = True
        self._apply_browse_view(
            center=Point(self._browse_center.x, self._browse_center.y),
            zoom=self._native_field_fit_zoom(),
            mode=CanvasZoomMode.NATIVE_FIELD_FIT,
            zoom_changed=True,
        )

    def actual_size(self) -> None:
        self._initial_fit_pending = False
        if self._slide_manifest is None:
            return
        self._initial_fit_done = True
        self._apply_browse_view(
            center=Point(self._browse_center.x, self._browse_center.y),
            zoom=1.0,
            mode=CanvasZoomMode.ACTUAL,
            zoom_changed=True,
        )

    def set_view_zoom(self, zoom: float) -> None:
        if self._slide_manifest is None:
            return
        self._set_zoom_at_widget_position(
            float(zoom),
            self._content_rect().center(),
            mode=CanvasZoomMode.CUSTOM,
        )

    def _set_zoom_at_widget_position(
        self,
        zoom: float,
        position: QPointF,
        *,
        mode: CanvasZoomMode,
    ) -> None:
        if self._slide_manifest is None:
            return
        anchor_before = self.widget_to_image(position)
        old_zoom = max(self._zoom, 1.0e-12)
        requested = max(self._whole_slide_fit_zoom(), min(40.0, float(zoom)))
        content_center = self._content_rect().center()
        center = Point(
            anchor_before.x - (position.x() - content_center.x()) / requested,
            anchor_before.y - (position.y() - content_center.y()) / requested,
        )
        self._apply_browse_view(
            center=center,
            zoom=requested,
            mode=mode,
            zoom_changed=not math.isclose(old_zoom, requested),
        )

    def schedule_initial_fit(self) -> None:
        if self._initial_fit_done or self._initial_fit_pending:
            return
        self._initial_fit_pending = True
        self._initial_fit_attempts = 0
        QTimer.singleShot(0, self._apply_initial_fit)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._slide_manifest is None:
            return
        modifiers = getattr(event, "modifiers", lambda: Qt.KeyboardModifier.NoModifier)()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self._zoom_current_viewport(event)
            return
        delta_y = event.angleDelta().y()
        delta_x = event.angleDelta().x()
        effective_delta = delta_y if delta_y != 0 else delta_x
        if effective_delta == 0:
            return
        wheel_step = max(1, int(getattr(self._settings, "digital_slide_focus_wheel_step", 1) or 1))
        self.set_focus_index(self._focus_index + (wheel_step if effective_delta > 0 else -wheel_step))
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # Cocoa tags the ordinary arrow keys as NumericPad.  KeypadModifier is
        # therefore a platform-origin flag here, not a user shortcut modifier.
        modifiers = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        if modifiers == Qt.KeyboardModifier.NoModifier and event.key() == Qt.Key.Key_M:
            self.toggle_navigation_mode()
            event.accept()
            return
        if modifiers in {
            Qt.KeyboardModifier.NoModifier,
            Qt.KeyboardModifier.ShiftModifier,
        } and event.key() in {
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        }:
            if self._navigation_mode == "smooth":
                self._begin_smooth_navigation(event)
                event.accept()
                return
            shift_navigation = (
                self._shift_navigation_enabled
                or modifiers == Qt.KeyboardModifier.ShiftModifier
            )
            if shift_navigation:
                source = self._source_view_rect()
                step_x = float(source.width())
                step_y = float(source.height())
            else:
                step_x, step_y = self._navigation_step()
            if event.key() == Qt.Key.Key_Left:
                self.move_viewport_by(-step_x, 0)
            elif event.key() == Qt.Key.Key_Right:
                self.move_viewport_by(step_x, 0)
            elif event.key() == Qt.Key.Key_Up:
                self.move_viewport_by(0, -step_y)
            else:
                self.move_viewport_by(0, step_y)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if (
            self._navigation_mode == "smooth"
            and event.key() in {Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down}
        ):
            if not getattr(event, "isAutoRepeat", lambda: False)():
                self._smooth_nav_keys.discard(int(event.key()))
                if not self._smooth_nav_keys:
                    self._smooth_nav_keyboard_shift = False
                    self._smooth_nav_timer.stop()
                    self._smooth_nav_last_at = 0.0
                    self._publish_viewport_state(throttled=False)
                    self._request_viewport_buffer()
                    # If the requested frame is already available this is the
                    # single warmable paint for the final viewport.  Otherwise
                    # passive overlay warming stays suspended until the
                    # renderer publishes that exact frame.
                    self.update()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _image_size(self) -> tuple[int, int] | None:
        if self._document is None:
            return None
        return self._document.image_size

    def _point_in_image(self, point: Point) -> bool:
        # Pixel-backed tools always operate on the fixed native work field,
        # even when the browse camera currently shows a larger part of the
        # slide.  The whole-slide bounds remain available to navigation and
        # overlay painting, but must never admit an edit outside ``_image``.
        bounds = self.native_viewport_rect()
        if bounds.isEmpty() or not bounds.isValid():
            return False
        return (
            bounds.left() <= point.x < bounds.right()
            and bounds.top() <= point.y < bounds.bottom()
        )

    def _clamp_to_mounted_viewport(self, point: Point) -> Point:
        """Clamp a pointer coordinate while retaining global slide space."""

        bounds = self.native_viewport_rect()
        if bounds.isEmpty() or not bounds.isValid():
            return point
        return Point(
            max(bounds.left(), min(bounds.right() - 1.0, point.x)),
            max(bounds.top(), min(bounds.bottom() - 1.0, point.y)),
        )

    def _query_object_snap(self, image_point: Point):
        # Geometry outside the mounted raster may be nearby in whole-slide
        # coordinates, but it is neither visible nor a valid pointer target.
        if not self._point_in_image(image_point):
            self._set_active_snap_candidate(None)
            return None
        candidate = super()._query_object_snap(image_point)
        if candidate is not None and not self._point_in_image(candidate.point_px):
            self._set_active_snap_candidate(None)
            return None
        return candidate

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # A press in FIT padding must remain outside the image so it cannot
        # start a measurement or construction command.
        previous = self._clamp_pointer_to_mounted_viewport
        self._clamp_pointer_to_mounted_viewport = False
        try:
            super().mousePressEvent(event)
        finally:
            self._clamp_pointer_to_mounted_viewport = previous

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.position() - self._last_mouse_pos
            self._last_mouse_pos = event.position()
            if delta.x() or delta.y():
                if self._zoom_mode in {
                    CanvasZoomMode.FIT,
                    CanvasZoomMode.NATIVE_FIELD_FIT,
                }:
                    self._zoom_mode = CanvasZoomMode.CUSTOM
                self.move_viewport_by(
                    -delta.x() / max(self._zoom, 1.0e-12),
                    -delta.y() / max(self._zoom, 1.0e-12),
                    throttled=True,
                )
            return
        # Once a pointer operation has started, dragging into the padding pins
        # the preview to the nearest mounted pixel instead of writing a global
        # coordinate belonging to an invisible slide region.
        previous = self._clamp_pointer_to_mounted_viewport
        self._clamp_pointer_to_mounted_viewport = self._has_pointer_edit_operation()
        try:
            super().mouseMoveEvent(event)
        finally:
            self._clamp_pointer_to_mounted_viewport = previous

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        was_panning = bool(self._panning)
        previous = self._clamp_pointer_to_mounted_viewport
        self._clamp_pointer_to_mounted_viewport = self._has_pointer_edit_operation()
        try:
            super().mouseReleaseEvent(event)
        finally:
            self._clamp_pointer_to_mounted_viewport = previous
        if was_panning and not self._panning:
            self._navigation_velocity = Point(0.0, 0.0)
            self._publish_viewport_state(throttled=False)
            self._request_display_frame()
            self._request_native_frame()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        # Double-click completes paths, but padding must not contribute a new
        # point or silently clamp an otherwise invalid click onto the border.
        point = self.widget_to_image(event.position())
        if not self._point_in_image(point):
            return
        previous = self._clamp_pointer_to_mounted_viewport
        self._clamp_pointer_to_mounted_viewport = False
        try:
            super().mouseDoubleClickEvent(event)
        finally:
            self._clamp_pointer_to_mounted_viewport = previous

    def _visible_image_rect(self) -> QRectF:
        if self._image is None:
            return QRectF()
        return self._paint_context().image_rect

    def _persist_view_state(self) -> None:
        if self._document is None:
            return
        slide_meta = dict(self._document.metadata.get("digital_slide", {})) if isinstance(self._document.metadata.get("digital_slide"), dict) else {}
        slide_meta["viewport_origin"] = [int(round(self._viewport_origin.x)), int(round(self._viewport_origin.y))]
        slide_meta["focus_index"] = int(self._focus_index)
        slide_meta["browse_view"] = {
            "version": _BROWSE_VIEW_VERSION,
            "center": [float(self._browse_center.x), float(self._browse_center.y)],
            "zoom": float(self._zoom),
            "mode": self._zoom_mode.value,
        }
        self._document.metadata["digital_slide"] = slide_meta

    def _normalized_focus_index(self, focus_index: int) -> int:
        if self._slide_manifest is None:
            return 0
        return max(0, min(int(focus_index), max(0, len(self._slide_manifest.focus_levels) - 1)))

    def _navigation_step(self) -> tuple[float, float]:
        source = self._source_view_rect()
        if source.isEmpty():
            return 0.0, 0.0
        return max(1.0, source.width() * 0.25), max(1.0, source.height() * 0.25)

    def _begin_smooth_navigation(self, event: QKeyEvent) -> None:
        if getattr(event, "isAutoRepeat", lambda: False)():
            return
        if not self._smooth_nav_keys:
            self._cancel_overlay_requests()
        self._smooth_nav_keys.add(int(event.key()))
        modifiers = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        self._smooth_nav_keyboard_shift = (
            modifiers == Qt.KeyboardModifier.ShiftModifier
        )
        self._smooth_nav_last_at = perf_counter()
        if not self._smooth_nav_timer.isActive():
            self._smooth_nav_timer.start()
        self._request_viewport_buffer()
        self._apply_smooth_navigation()

    def _apply_smooth_navigation(self) -> None:
        if self._slide_manifest is None or not self._smooth_nav_keys:
            self._smooth_nav_timer.stop()
            self._smooth_nav_last_at = 0.0
            self._smooth_nav_keyboard_shift = False
            return
        now = perf_counter()
        previous = self._smooth_nav_last_at or now
        self._smooth_nav_last_at = now
        dt = max(0.001, min(0.12, now - previous))
        multiplier = (
            3.0
            if self._shift_navigation_enabled or self._smooth_nav_keyboard_shift
            else 1.0
        )
        screen_speed = (
            min(self._content_rect().width(), self._content_rect().height())
            * 0.75
            * multiplier
        )
        source_step = max(1.0, screen_speed * dt / max(self._zoom, 1.0e-12))
        step_x = source_step
        step_y = source_step
        dx = 0.0
        dy = 0.0
        if int(Qt.Key.Key_Left) in self._smooth_nav_keys:
            dx -= step_x
        if int(Qt.Key.Key_Right) in self._smooth_nav_keys:
            dx += step_x
        if int(Qt.Key.Key_Up) in self._smooth_nav_keys:
            dy -= step_y
        if int(Qt.Key.Key_Down) in self._smooth_nav_keys:
            dy += step_y
        if dx or dy:
            if dx and dy:
                diagonal_scale = 1.0 / math.sqrt(2.0)
                dx *= diagonal_scale
                dy *= diagonal_scale
            self.move_viewport_by(dx, dy, throttled=True)

    def _apply_initial_fit(self) -> None:
        if not self._initial_fit_pending or self._image is None:
            return
        if (self.width() < 120 or self.height() < 120) and self._initial_fit_attempts < 6:
            self._initial_fit_attempts += 1
            QTimer.singleShot(50, self._apply_initial_fit)
            return
        self.fit_native_viewport()

    def _zoom_current_viewport(self, event: QWheelEvent) -> None:
        if self._image is None:
            return
        delta_y = event.angleDelta().y()
        delta_x = event.angleDelta().x()
        effective_delta = delta_y if delta_y != 0 else delta_x
        if effective_delta == 0:
            return
        self._initial_fit_pending = False
        self._initial_fit_done = True
        cursor_position = event.position()
        zoom_factor = 1.15 if effective_delta > 0 else 1 / 1.15
        self._set_zoom_at_widget_position(
            self._zoom * zoom_factor,
            cursor_position,
            mode=CanvasZoomMode.CUSTOM,
        )
        event.accept()

    def _overlay_navigation_is_transient(self) -> bool:
        """Return whether the mounted raster is still following navigation.

        Measurements use global slide coordinates, so direct vector rendering
        remains exact while the viewport changes.  Building passive tiles at
        every 16 ms navigation tick, however, only creates work whose
        device-pixel phase is obsolete before completion.
        """

        stats = self.renderer_stats()
        return bool(
            self._smooth_nav_keys
            or self._smooth_nav_timer.isActive()
            or (stats is not None and stats.pending_requests > 0)
        )

    def _overlay_motion_active(self) -> bool:
        return (
            super()._overlay_motion_active()
            or self._overlay_navigation_is_transient()
        )

    def _enqueue_overlay_tiles(self, keys: list[CanvasOverlayTileKey]) -> None:
        """Warm passive tiles only after the viewport raster has stabilized."""

        if self._overlay_navigation_is_transient():
            # Cancellation is cooperative in CanvasOverlayTileCache.  Clearing
            # the local queue here also prevents a stale tile from starting
            # after the current worker acknowledges cancellation.
            self._cancel_overlay_requests()
            return
        super()._enqueue_overlay_tiles(keys)

    def _clamp_viewport(self) -> None:
        if self._slide_manifest is None:
            return
        view_width = self._slide_manifest.viewport_width
        view_height = self._slide_manifest.viewport_height
        max_x = max(0, self._slide_manifest.width - view_width)
        max_y = max(0, self._slide_manifest.height - view_height)
        self._viewport_origin = Point(
            max(0.0, min(float(max_x), self._viewport_origin.x)),
            max(0.0, min(float(max_y), self._viewport_origin.y)),
        )

    def _invalidate_viewport_buffer(self) -> None:
        self._latest_display_request_id += 1
        self._latest_native_request_id += 1
        self._render_frame = None
        self._previous_render_frame = None
        self._focus_transition_frame = None
        self._focus_transition_image = None
        self._focus_transition_native_rect = None
        self._native_frame_pending_key = None
        self._allow_viewport_buffer_retry()

    def _allow_viewport_buffer_retry(self) -> None:
        """Clear a permanent-error latch at an explicit user retry boundary."""

        self._viewport_buffer_error_blocked = False

    def refresh_viewport_buffer(self) -> None:
        """Explicitly retry the background viewport buffer after a read error."""

        self._allow_viewport_buffer_retry()
        self._request_viewport_buffer()

    def _request_viewport_buffer(self) -> None:
        self._request_display_frame()
        self._request_native_frame()

    def request_overview(self) -> None:
        """Queue the lowest-priority whole-slide preview on the slide worker."""

        if (
            not self._overview_enabled
            or self._slide_manifest is None
            or not self.isVisible()
        ):
            return
        focus_index = self._overview_target_focus_index()
        cached = self._overview_cache.get(focus_index)
        if cached is not None:
            self._overview_cache.move_to_end(focus_index)
            if self._overview_focus_index != focus_index:
                self._overview_image = cached
                self._overview_focus_index = focus_index
                self.overviewImageChanged.emit(cached)
            return
        if focus_index in self._overview_failed_focuses:
            return
        self._start_renderer()
        renderer = self._renderer
        if renderer is None:
            return
        slide_width = max(1, int(self._slide_manifest.width))
        slide_height = max(1, int(self._slide_manifest.height))
        scale = min(
            1.0,
            float(_OVERVIEW_MAX_EDGE) / max(slide_width, slide_height),
        )
        self._render_request_id += 1
        self._latest_overview_request_id = self._render_request_id
        self._overview_pending = True
        renderer.submit(
            DigitalSlideRenderRequest(
                request_id=self._render_request_id,
                purpose="overview",
                source_rect=(0.0, 0.0, float(slide_width), float(slide_height)),
                output_size_px=(
                    max(1, int(round(slide_width * scale))),
                    max(1, int(round(slide_height * scale))),
                ),
                focus_index=focus_index,
                device_pixel_ratio=1.0,
                blend_width=self._blend_width(),
            )
        )

    def _publish_viewport_state(
        self,
        *,
        throttled: bool,
        zoom_changed: bool = False,
    ) -> None:
        now = perf_counter()
        if throttled and (now - self._viewport_last_publish_at) < 0.12:
            self._publish_view_transform(zoom_changed=zoom_changed)
            return
        self._viewport_last_publish_at = now
        self._persist_view_state()
        self.viewportChanged.emit(
            int(round(self._viewport_origin.x)),
            int(round(self._viewport_origin.y)),
            int(self._focus_index),
        )
        self._publish_view_transform(zoom_changed=zoom_changed)

    def _reload_viewport(self, *, throttled: bool = False) -> None:
        self._request_display_frame()
        self._request_native_frame()
        self._update_pixel_work_state()
        self._publish_viewport_state(throttled=throttled)
