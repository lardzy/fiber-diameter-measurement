from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import math
from time import perf_counter
from weakref import ref

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QRegion,
    QWheelEvent,
)
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
_FOCUS_SETTLE_MS = 75
_COARSE_FRAME_MAX_EDGE = 512
_PRESENTATION_PREVIEW_MAX_EDGE = 1024
# Whole-slide display frames remain byte-bounded below.  Keeping several exact
# native focus planes avoids falling back to a proxy when users scrub the same
# acquisition field back and forth across more than three focus levels.
_DISPLAY_FRAME_CACHE_LIMIT = 8
_DISPLAY_FRAME_CACHE_BYTES = 96 * 1024 * 1024
_VECTOR_MEASUREMENT_TOOLS = {"manual", "snap", "continuous_manual"}


@dataclass(frozen=True, slots=True)
class DigitalSlideBrowseView:
    center_px: Point
    zoom: float
    mode: CanvasZoomMode


@dataclass(frozen=True, slots=True)
class DigitalSlidePresentationState:
    generation: int
    focus_index: int
    quality: str
    pixel_exact: bool
    coverage_rects: tuple[tuple[float, float, float, float], ...]


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
        self._view_generation = 0
        self._render_frame: DigitalSlideRenderFrame | None = None
        self._previous_render_frame: DigitalSlideRenderFrame | None = None
        self._coarse_render_frame: DigitalSlideRenderFrame | None = None
        self._presentation_preview_frame: DigitalSlideRenderFrame | None = None
        self._presentation_preview_cache: OrderedDict[
            int, DigitalSlideRenderFrame
        ] = OrderedDict()
        self._display_frame_cache: OrderedDict[
            tuple[object, ...], DigitalSlideRenderFrame
        ] = OrderedDict()
        self._display_frame_cache_bytes = 0
        # Paint-only handoff during focus changes.  These frames never become
        # the authoritative ``_image`` consumed by pixel algorithms.
        self._focus_transition_frame: DigitalSlideRenderFrame | None = None
        self._focus_transition_image: QImage | None = None
        self._focus_transition_native_rect: QRectF | None = None
        self._renderer: DigitalSlideRenderer | None = None
        self._render_request_id = 0
        self._latest_preview_request_id = 0
        self._latest_coarse_request_id = 0
        self._latest_display_request_id = 0
        self._latest_native_request_id = 0
        self._latest_overview_request_id = 0
        self._native_frame_key: tuple[int, int, int] | None = None
        self._native_frame_pending_key: tuple[int, int, int] | None = None
        self._native_frame_ever_ready = False
        self._pixel_work_enabled = True
        self._pixel_work_reason = ""
        self._navigation_velocity = Point(0.0, 0.0)
        self._last_navigation_origin = Point(0.0, 0.0)
        self._last_navigation_at = 0.0
        self._last_vector_input_notice_at = 0.0
        self._focus_index = 0
        self._focus_direction = 0
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
        self._final_render_timer = QTimer(self)
        self._final_render_timer.setSingleShot(True)
        self._final_render_timer.setInterval(_FOCUS_SETTLE_MS)
        self._final_render_timer.timeout.connect(self._request_final_frame)
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
        self._final_render_timer.stop()
        self._view_generation += 1
        self._coarse_render_frame = None
        self._presentation_preview_frame = None
        self._presentation_preview_cache.clear()
        self._clear_display_frame_cache()
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
        self._native_frame_ever_ready = False
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
        self._focus_direction = 0
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
        self._native_frame_ever_ready = False
        self._start_renderer()
        self._request_presentation_preview()
        self._request_interactive_frames()
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
        self._final_render_timer.stop()
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
        self._coarse_render_frame = None
        self._presentation_preview_frame = None
        self._presentation_preview_cache.clear()
        self._clear_display_frame_cache()
        self._focus_transition_frame = None
        self._focus_transition_image = None
        self._focus_transition_native_rect = None
        self._native_frame_key = None
        self._native_frame_pending_key = None
        self._native_frame_ever_ready = False
        self._slide_store = None
        self._slide_manifest = None

    def hideEvent(self, event) -> None:
        """Cancel navigation/buffer work when another document tab takes over."""

        self._smooth_nav_keys.clear()
        self._smooth_nav_timer.stop()
        self._final_render_timer.stop()
        self._smooth_nav_last_at = 0.0
        self._overview_debounce_timer.stop()
        self._native_viewport_indicator_timer.stop()
        self._native_viewport_indicator_visible = False
        self._latest_overview_request_id += 1
        self._overview_pending = False
        self._stop_renderer()
        self._view_generation += 1
        self._latest_preview_request_id += 1
        self._latest_coarse_request_id += 1
        self._latest_display_request_id += 1
        self._latest_native_request_id += 1
        self._native_frame_pending_key = None
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._allow_viewport_buffer_retry()
        if self._initial_fit_pending:
            self._apply_initial_fit()
        self._start_renderer()
        self._request_presentation_preview()
        self._request_interactive_frames()
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
            self._request_presentation_preview()
            self._request_interactive_frames()

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

    def presentation_state(self) -> DigitalSlidePresentationState:
        frame = self._current_presentation_frame()
        if frame is None:
            return DigitalSlidePresentationState(
                generation=int(self._view_generation),
                focus_index=int(self._focus_index),
                quality="placeholder",
                pixel_exact=False,
                coverage_rects=(),
            )
        return DigitalSlidePresentationState(
            generation=int(frame.generation),
            focus_index=int(frame.focus_index),
            quality=str(frame.quality),
            pixel_exact=bool(frame.pixel_exact),
            coverage_rects=tuple(frame.coverage_rects),
        )

    def vector_measurement_available(self, point: Point | None = None) -> bool:
        """Return whether a real current-focus frame can accept vector input."""

        if self._slide_manifest is None or self.large_area_browse_active():
            return False
        frame = self._current_presentation_frame()
        if (
            frame is None
            or frame.generation != self._view_generation
            or frame.focus_index != self._focus_index
            or frame.quality == "placeholder"
        ):
            return False
        if point is None:
            return bool(frame.coverage_rects)
        return any(
            left <= point.x < left + width
            and top <= point.y < top + height
            for left, top, width, height in frame.coverage_rects
        )

    def vector_measurement_controls_enabled(self) -> bool:
        """Keep vector tools selectable while native-scale pixels refine.

        Actual clicks remain guarded by :meth:`vector_measurement_available`.
        Separating the stable tool-control state from point-level coverage
        avoids flashing the toolbar on every camera/focus generation while
        still rejecting placeholders and unresolved source areas.
        """

        return bool(
            self._slide_manifest is not None
            and not self.large_area_browse_active()
        )

    def _current_presentation_frame(self) -> DigitalSlideRenderFrame | None:
        for frame in (self._render_frame, self._coarse_render_frame):
            if (
                frame is not None
                and frame.generation == self._view_generation
                and frame.focus_index == self._focus_index
            ):
                return frame
        return None

    def large_area_browse_active(self) -> bool:
        """Return whether the camera is below the native pixel-work scale."""

        return bool(
            self._slide_manifest is not None
            and self._zoom + _PIXEL_WORK_EPSILON < self._native_field_fit_zoom()
        )

    def native_viewport_indicator_visible(self) -> bool:
        return bool(self._native_viewport_indicator_visible)

    def pixel_work_controls_blocked(self) -> bool:
        """Return whether pixel-tool controls should visibly enter a blocked state.

        A native frame reload after movement or focus change is normally very
        short.  Exact operations remain guarded by ``pixel_work_enabled()``,
        but keeping the controls visually stable avoids disabling and enabling
        the whole toolbar for every focus tick.
        """

        return bool(
            self._slide_manifest is None
            or not self._native_frame_ever_ready
            or self.large_area_browse_active()
            or self._viewport_buffer_error_blocked
        )

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
        self._presentation_preview_frame = None
        self._presentation_preview_cache.clear()
        self._clear_display_frame_cache()
        self._native_frame_key = None
        self._native_frame_pending_key = None
        self._advance_view_generation()
        self._request_presentation_preview()
        self._request_interactive_frames()
        self._update_pixel_work_state()

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
        self._focus_direction = 0
        self._clamp_browse_center()
        self._sync_pan_from_browse_center()
        previous_key = self._native_request_key()
        self._update_native_viewport_origin()
        if self._native_request_key() != previous_key:
            self._native_frame_pending_key = None
        self._reset_proxy_warming()
        self._cancel_overlay_requests()
        self._persist_view_state()
        self._advance_view_generation()
        self._request_interactive_frames()
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
        self._latest_preview_request_id = 0
        self._latest_coarse_request_id = 0
        self._latest_display_request_id = 0
        self._latest_native_request_id = 0
        self._latest_overview_request_id = 0
        self._native_frame_pending_key = None
        if renderer is not None:
            renderer.close()

    def _blend_width(self) -> int:
        metadata = (
            getattr(self._slide_manifest, "metadata", None)
            if self._slide_manifest is not None
            else None
        )
        if not isinstance(metadata, dict):
            return 0
        try:
            return max(0, int(metadata.get("blend_width", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _advance_view_generation(self) -> None:
        self._view_generation += 1
        advance_generation = getattr(self._renderer, "advance_generation", None)
        if callable(advance_generation):
            advance_generation(int(self._view_generation))
        if self._overview_enabled:
            # Whole-slide navigator work is intentionally the lowest priority.
            # Camera motion cancels its old generation, then this debounce
            # retries only after interaction settles so a cancelled initial
            # fit can never leave the navigator permanently blank.
            self._latest_overview_request_id += 1
            self._overview_pending = False
            target_focus = self._overview_target_focus_index()
            if (
                target_focus not in self._overview_cache
                and target_focus not in self._overview_failed_focuses
                and self.isVisible()
            ):
                self._overview_debounce_timer.start()
        if self._render_frame is not None:
            self._previous_render_frame = self._history_frame(
                self._render_frame
            )
            self._render_frame = None
        self._coarse_render_frame = None
        self._focus_transition_frame = None
        self._focus_transition_image = None
        self._focus_transition_native_rect = None
        self._native_frame_pending_key = None

    def _request_interactive_frames(self) -> None:
        if self._restore_cached_final_frame():
            self._update_pixel_work_state()
            self.update()
            return
        if self.large_area_browse_active():
            self._request_coarse_frame()
            # Do not restart the settle timer at every 16 ms navigation tick.
            # Let it expire periodically during continuous motion so current
            # display-quality work gets a chance to replace the proxy before
            # the user releases the key or mouse.
            if not self._final_render_timer.isActive():
                self._final_render_timer.start()
            return
        # At native-field scale a 512px display proxy is visibly softer than
        # the acquisition frame and creates a moving double-image when painted
        # over the previous exact field.  Request the single canonical LOD0
        # frame immediately; the renderer's generation/latest-wins policy keeps
        # continuous navigation bounded without introducing a second display
        # composition.
        self._final_render_timer.stop()
        self._request_native_frame()

    def _request_focus_frames(self) -> None:
        """Schedule a focus handoff without painting native-scale proxies."""

        if self._restore_cached_final_frame():
            self._update_pixel_work_state()
            self.update()
            return
        if self.large_area_browse_active():
            self._request_presentation_preview()
            self._request_coarse_frame()
        # Rapid wheel input restarts this timer, so intermediate focus planes
        # never launch full LOD0 work.  The exact old-focus handoff stays visible
        # until the final target focus is requested.
        self._final_render_timer.start()

    def _request_presentation_preview(self) -> None:
        if self._slide_manifest is None or not self.isVisible():
            return
        cached = self._presentation_preview_cache.get(int(self._focus_index))
        if cached is not None:
            self._presentation_preview_cache.move_to_end(int(self._focus_index))
            record_preview_hit = getattr(
                self._renderer,
                "record_preview_memory_hit",
                None,
            )
            if callable(record_preview_hit):
                record_preview_hit()
            self._presentation_preview_frame = replace(
                cached,
                generation=int(self._view_generation),
            )
            self.update()
            return
        self._start_renderer()
        renderer = self._renderer
        if renderer is None:
            return
        slide_width = max(1, int(self._slide_manifest.width))
        slide_height = max(1, int(self._slide_manifest.height))
        scale = min(
            1.0,
            float(_PRESENTATION_PREVIEW_MAX_EDGE)
            / max(slide_width, slide_height),
        )
        self._render_request_id += 1
        self._latest_preview_request_id = self._render_request_id
        renderer.submit(
            DigitalSlideRenderRequest(
                request_id=self._render_request_id,
                purpose="preview",
                source_rect=(0.0, 0.0, float(slide_width), float(slide_height)),
                output_size_px=(
                    max(1, int(round(slide_width * scale))),
                    max(1, int(round(slide_height * scale))),
                ),
                focus_index=int(self._focus_index),
                device_pixel_ratio=1.0,
                generation=int(self._view_generation),
                quality="coarse",
                preview_max_edge=_PRESENTATION_PREVIEW_MAX_EDGE,
                priority=0,
            )
        )

    def _request_coarse_frame(self) -> None:
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
        physical_width = max(1, int(round(content.width() * dpr)))
        physical_height = max(1, int(round(content.height() * dpr)))
        coarse_edge = (
            _PRESENTATION_PREVIEW_MAX_EDGE
            if self.large_area_browse_active()
            and self._slide_manifest is not None
            and source.width() >= float(self._slide_manifest.width) - 1.0e-6
            and source.height() >= float(self._slide_manifest.height) - 1.0e-6
            else _COARSE_FRAME_MAX_EDGE
        )
        scale = min(
            1.0,
            float(coarse_edge)
            / max(physical_width, physical_height),
        )
        self._render_request_id += 1
        self._latest_coarse_request_id = self._render_request_id
        renderer.submit(
            DigitalSlideRenderRequest(
                request_id=self._render_request_id,
                purpose="coarse",
                source_rect=(
                    source.x(),
                    source.y(),
                    source.width(),
                    source.height(),
                ),
                output_size_px=(
                    max(1, int(round(physical_width * scale))),
                    max(1, int(round(physical_height * scale))),
                ),
                focus_index=int(self._focus_index),
                device_pixel_ratio=dpr,
                blend_width=self._blend_width(),
                velocity_px_per_second=(
                    float(self._navigation_velocity.x),
                    float(self._navigation_velocity.y),
                ),
                generation=int(self._view_generation),
                quality="coarse",
                preview_max_edge=(
                    _PRESENTATION_PREVIEW_MAX_EDGE
                    if coarse_edge == _PRESENTATION_PREVIEW_MAX_EDGE
                    else 0
                ),
                priority=1,
            )
        )

    def _request_final_frame(self) -> None:
        if self._restore_cached_final_frame():
            self._update_pixel_work_state()
            self.update()
            return
        if self.large_area_browse_active():
            self._request_display_frame()
        else:
            self._request_native_frame()

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
        contains_whole_slide = bool(
            source.x() <= 1.0e-6
            and source.y() <= 1.0e-6
            and source.x() + source.width()
            >= float(self._slide_manifest.width) - 1.0e-6
            and source.y() + source.height()
            >= float(self._slide_manifest.height) - 1.0e-6
        )
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
                generation=int(self._view_generation),
                quality="final",
                preview_max_edge=(
                    _PRESENTATION_PREVIEW_MAX_EDGE
                    if contains_whole_slide
                    else 0
                ),
                priority=2,
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
                generation=int(self._view_generation),
                quality="final",
                priority=2,
                focus_direction=int(self._focus_direction),
            )
        )

    def _frame_cache_key(
        self,
        *,
        purpose: str,
        focus_index: int,
        source_rect: tuple[float, float, float, float],
        output_size_px: tuple[int, int],
        device_pixel_ratio: float,
    ) -> tuple[object, ...]:
        return (
            str(purpose),
            int(focus_index),
            *(round(float(value), 4) for value in source_rect),
            int(output_size_px[0]),
            int(output_size_px[1]),
            round(float(device_pixel_ratio), 4),
            int(self._blend_width()),
        )

    def _desired_final_cache_key(self) -> tuple[object, ...] | None:
        if self._slide_manifest is None:
            return None
        if not self.large_area_browse_active():
            key = self._native_request_key()
            return self._frame_cache_key(
                purpose="native",
                focus_index=self._focus_index,
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
                device_pixel_ratio=1.0,
            )
        content = self._content_rect()
        source = self._source_view_rect()
        if content.isEmpty() or source.isEmpty():
            return None
        dpr = max(1.0, float(self.devicePixelRatioF()))
        return self._frame_cache_key(
            purpose="display",
            focus_index=self._focus_index,
            source_rect=(source.x(), source.y(), source.width(), source.height()),
            output_size_px=(
                max(1, int(round(content.width() * dpr))),
                max(1, int(round(content.height() * dpr))),
            ),
            device_pixel_ratio=dpr,
        )

    def _restore_cached_final_frame(self) -> bool:
        key = self._desired_final_cache_key()
        if key is None:
            return False
        cached = self._display_frame_cache.get(key)
        if cached is None:
            return False
        self._display_frame_cache.move_to_end(key)
        self._render_request_id += 1
        restored = replace(
            cached,
            request_id=self._render_request_id,
            generation=int(self._view_generation),
            elapsed_ms=0.0,
            decoded_tiles=0,
            cache_hits=max(1, int(cached.cache_hits)),
        )
        record_cache_hit = getattr(
            self._renderer,
            "record_canvas_cache_hit",
            None,
        )
        if callable(record_cache_hit):
            record_cache_hit(
                exact=restored.purpose == "native" and restored.pixel_exact
            )
        if restored.purpose == "native":
            self._latest_native_request_id = restored.request_id
            self._native_frame_pending_key = self._native_request_key()
        else:
            self._latest_display_request_id = restored.request_id
        self._on_render_frame_ready(restored)
        return True

    def _remember_display_frame(self, frame: DigitalSlideRenderFrame) -> None:
        if frame.quality != "final" or frame.image.isNull():
            return
        image_bytes = max(0, int(frame.image.sizeInBytes()))
        if image_bytes <= 0 or image_bytes > _DISPLAY_FRAME_CACHE_BYTES:
            return
        key = self._frame_cache_key(
            purpose=frame.purpose,
            focus_index=frame.focus_index,
            source_rect=frame.source_rect,
            output_size_px=frame.output_size_px,
            device_pixel_ratio=frame.device_pixel_ratio,
        )
        previous = self._display_frame_cache.pop(key, None)
        if previous is not None:
            self._display_frame_cache_bytes -= max(
                0,
                int(previous.image.sizeInBytes()),
            )
        self._display_frame_cache[key] = frame
        self._display_frame_cache_bytes += image_bytes
        while self._display_frame_cache and (
            len(self._display_frame_cache) > _DISPLAY_FRAME_CACHE_LIMIT
            or self._display_frame_cache_bytes > _DISPLAY_FRAME_CACHE_BYTES
        ):
            protected_keys = {key}
            for candidate_key, candidate in reversed(
                self._display_frame_cache.items()
            ):
                if (
                    candidate.purpose == "display"
                    and self._frame_contains_whole_slide(candidate)
                ):
                    protected_keys.add(candidate_key)
                    break
            remove_key = next(
                (
                    candidate_key
                    for candidate_key in self._display_frame_cache
                    if candidate_key not in protected_keys
                ),
                None,
            )
            if remove_key is None:
                remove_key = next(iter(self._display_frame_cache))
            removed = self._display_frame_cache.pop(remove_key)
            self._display_frame_cache_bytes = max(
                0,
                self._display_frame_cache_bytes
                - max(0, int(removed.image.sizeInBytes())),
            )

    def _clear_display_frame_cache(self) -> None:
        self._display_frame_cache.clear()
        self._display_frame_cache_bytes = 0

    @staticmethod
    def _history_frame(
        frame: DigitalSlideRenderFrame | None,
    ) -> DigitalSlideRenderFrame | None:
        if (
            frame is None
            or frame.image.isNull()
            or int(frame.image.sizeInBytes()) > _DISPLAY_FRAME_CACHE_BYTES
        ):
            return None
        return frame

    def _frame_contains_whole_slide(
        self,
        frame: DigitalSlideRenderFrame,
    ) -> bool:
        if self._slide_manifest is None:
            return False
        x, y, width, height = frame.source_rect
        return bool(
            x <= 1.0e-6
            and y <= 1.0e-6
            and x + width >= float(self._slide_manifest.width) - 1.0e-6
            and y + height >= float(self._slide_manifest.height) - 1.0e-6
        )

    def _presentation_preview_from_frame(
        self,
        frame: DigitalSlideRenderFrame,
    ) -> DigitalSlideRenderFrame | None:
        """Crop FIT padding before reusing a whole-view frame as preview."""

        if (
            self._slide_manifest is None
            or frame.image.isNull()
            or not self._frame_contains_whole_slide(frame)
        ):
            return None
        x, y, width, height = frame.source_rect
        if width <= 0.0 or height <= 0.0:
            return None
        scale_x = frame.image.width() / float(width)
        scale_y = frame.image.height() / float(height)
        left = max(0, int(math.floor((0.0 - x) * scale_x)))
        top = max(0, int(math.floor((0.0 - y) * scale_y)))
        right = min(
            frame.image.width(),
            int(
                math.ceil(
                    (float(self._slide_manifest.width) - x) * scale_x
                )
            ),
        )
        bottom = min(
            frame.image.height(),
            int(
                math.ceil(
                    (float(self._slide_manifest.height) - y) * scale_y
                )
            ),
        )
        if right <= left or bottom <= top:
            return None
        exact_slide_frame = bool(
            math.isclose(x, 0.0, abs_tol=1.0e-6)
            and math.isclose(y, 0.0, abs_tol=1.0e-6)
            and math.isclose(
                width,
                float(self._slide_manifest.width),
                abs_tol=1.0e-6,
            )
            and math.isclose(
                height,
                float(self._slide_manifest.height),
                abs_tol=1.0e-6,
            )
        )
        image = (
            QImage(frame.image)
            if exact_slide_frame
            else frame.image.copy(QRect(left, top, right - left, bottom - top))
        )
        if image.isNull():
            return None
        bounded_edge = min(
            _PRESENTATION_PREVIEW_MAX_EDGE,
            max(image.width(), image.height()),
        )
        scale = min(
            1.0,
            float(bounded_edge)
            / max(
                float(self._slide_manifest.width),
                float(self._slide_manifest.height),
                1.0,
            ),
        )
        output_size = (
            max(1, int(round(float(self._slide_manifest.width) * scale))),
            max(1, int(round(float(self._slide_manifest.height) * scale))),
        )
        if image.size().toTuple() != output_size:
            image = image.scaled(
                output_size[0],
                output_size[1],
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        return replace(
            frame,
            purpose="preview",
            source_rect=(
                0.0,
                0.0,
                float(self._slide_manifest.width),
                float(self._slide_manifest.height),
            ),
            output_size_px=output_size,
            device_pixel_ratio=1.0,
            image=image,
            pixel_exact=False,
            coverage_rects=(),
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
        if frame.purpose == "preview":
            if (
                frame.request_id != self._latest_preview_request_id
                or frame.generation != self._view_generation
                or frame.focus_index != self._focus_index
                or frame.image.isNull()
            ):
                return
            self._presentation_preview_frame = frame
            if frame.quality != "placeholder":
                self._presentation_preview_cache[int(frame.focus_index)] = frame
                self._presentation_preview_cache.move_to_end(int(frame.focus_index))
                while len(self._presentation_preview_cache) > _OVERVIEW_CACHE_LIMIT:
                    self._presentation_preview_cache.popitem(last=False)
                if self.large_area_browse_active():
                    # A source/derived preview is an appropriate handoff only
                    # while the camera is intentionally displaying an LOD.
                    self._focus_transition_frame = None
                    self._focus_transition_image = None
                    self._focus_transition_native_rect = None
            self.update()
            return
        if (
            frame.focus_index != self._focus_index
            or frame.generation != self._view_generation
        ):
            return
        if frame.purpose == "coarse":
            if frame.request_id != self._latest_coarse_request_id:
                return
            if (
                self._render_frame is not None
                and self._render_frame.generation == self._view_generation
                and self._render_frame.quality == "final"
            ):
                return
            self._coarse_render_frame = frame
            if frame.complete and self.large_area_browse_active():
                # Keep the old-focus visual beneath progressive snapshots so
                # unresolved tiles do not flash as a flat loading colour.  A
                # complete target-focus coarse frame no longer needs it.
                self._focus_transition_frame = None
                self._focus_transition_image = None
                self._focus_transition_native_rect = None
            preview = (
                self._presentation_preview_from_frame(frame)
                if frame.complete
                else None
            )
            if preview is not None:
                self._presentation_preview_frame = preview
                if frame.complete:
                    self._presentation_preview_cache[int(frame.focus_index)] = preview
                    self._presentation_preview_cache.move_to_end(
                        int(frame.focus_index)
                    )
                    while (
                        len(self._presentation_preview_cache)
                        > _OVERVIEW_CACHE_LIMIT
                    ):
                        self._presentation_preview_cache.popitem(last=False)
            self.update()
            return
        if frame.purpose == "display":
            if frame.request_id != self._latest_display_request_id:
                return
            self._previous_render_frame = self._history_frame(
                self._render_frame
            )
            self._render_frame = frame
            self._coarse_render_frame = None
            self._remember_display_frame(frame)
            preview = self._presentation_preview_from_frame(frame)
            if preview is not None:
                self._presentation_preview_frame = preview
                self._presentation_preview_cache[int(frame.focus_index)] = preview
                self._presentation_preview_cache.move_to_end(
                    int(frame.focus_index)
                )
                while (
                    len(self._presentation_preview_cache)
                    > _OVERVIEW_CACHE_LIMIT
                ):
                    self._presentation_preview_cache.popitem(last=False)
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
        self._native_frame_ever_ready = True
        self._previous_render_frame = self._history_frame(self._render_frame)
        self._render_frame = frame
        self._coarse_render_frame = None
        self._remember_display_frame(frame)
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
        expected = {
            "native": self._latest_native_request_id,
            "overview": self._latest_overview_request_id,
            "preview": self._latest_preview_request_id,
            "coarse": self._latest_coarse_request_id,
            "display": self._latest_display_request_id,
        }.get(failure.purpose, 0)
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
        elif failure.purpose == "preview":
            return
        elif failure.purpose == "coarse":
            # The exact request remains scheduled.  Preserve all active vector
            # drafts and the last presentation while it retries/refines.
            return
        # A failed target-focus read must not expose the flat loading surface.
        # Retain the paint-only handoff while the error is reported/retried;
        # its focus/generation still cannot satisfy any input or pixel gate.
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
        # A missing exact native frame gates pixel algorithms, but it must not
        # be translated into DocumentCanvas read-only: that operation destroys
        # active line/polyline drafts.  Only the deliberate large-area mode is
        # a destructive interaction boundary, and crossing it is already
        # blocked while a draft exists.
        hard_read_only = self._slide_manifest is None or self.large_area_browse_active()
        if self._read_only != hard_read_only:
            self.set_read_only(hard_read_only)
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
        else:
            # A single full-field step has no preceding timestamp, but its
            # destination still needs forward protection immediately.  The
            # renderer predicts 180 ms ahead, so this synthetic velocity maps
            # exactly one first-step displacement into that guard region.
            self._navigation_velocity = Point(
                (self._browse_center.x - old_center.x) / 0.18,
                (self._browse_center.y - old_center.y) / 0.18,
            )
        self._last_navigation_at = now
        self._last_navigation_origin = Point(self._browse_center.x, self._browse_center.y)
        self._focus_direction = 0
        self._sync_pan_from_browse_center()
        self._update_native_viewport_origin()
        if self._native_frame_key != self._native_request_key():
            self._native_frame_pending_key = None
        self._persist_view_state()
        self._advance_view_generation()
        self._request_interactive_frames()
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
        self._focus_direction = 1 if focus_index > self._focus_index else -1
        # Capture the currently painted content before advancing generation.
        # ``_advance_view_generation()`` deliberately clears obsolete current
        # frames, so assigning the handoff before that call would immediately
        # discard it and expose the light loading surface for one or more
        # paints.  The handoff remains paint-only and therefore cannot satisfy
        # either vector coverage or exact pixel-work checks.
        handoff_candidates = (
            self._render_frame,
            (
                self._coarse_render_frame
                if self._coarse_render_frame is not None
                and self._coarse_render_frame.complete
                else None
            ),
            (
                self._presentation_preview_frame
                if self._presentation_preview_frame is not None
                and self._presentation_preview_frame.quality != "placeholder"
                else None
            ),
            self._focus_transition_frame,
            self._coarse_render_frame,
        )
        handoff_frame = next(
            (
                frame
                for frame in handoff_candidates
                if frame is not None and not frame.image.isNull()
            ),
            None,
        )
        handoff_image = (
            self._image
            if self._image is not None
            and not self._image.isNull()
            and self._native_frame_key is not None
            else self._focus_transition_image
        )
        handoff_native_rect = (
            self.native_viewport_rect()
            if self._image is not None
            and not self._image.isNull()
            and self._native_frame_key is not None
            else self._focus_transition_native_rect
        )
        # Install it before emitting ``focusChanged`` as connected UI slots may
        # request an immediate repaint.  It is restored once more after the
        # generation reset below, which intentionally clears transition state.
        self._focus_transition_frame = handoff_frame
        self._focus_transition_image = handoff_image
        self._focus_transition_native_rect = handoff_native_rect
        self._focus_index = focus_index
        self.focusChanged.emit(focus_index)
        self._native_frame_key = None
        self._native_frame_pending_key = None
        self._allow_viewport_buffer_retry()
        self._advance_view_generation()
        self._focus_transition_frame = handoff_frame
        self._focus_transition_image = handoff_image
        self._focus_transition_native_rect = handoff_native_rect
        if self.large_area_browse_active():
            cached_preview = self._presentation_preview_cache.get(int(focus_index))
            self._presentation_preview_frame = (
                replace(cached_preview, generation=int(self._view_generation))
                if cached_preview is not None
                else None
            )
        else:
            # Whole-slide previews are useful in large-area browsing, but at
            # native scale they are the low-resolution overlay reported by
            # users.  Keep the exact paint-only handoff instead.
            self._presentation_preview_frame = None
        self._request_focus_frames()
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
        # A target-focus loading surface prevents white slide content from
        # flashing through the dark application workspace while the first real
        # low-resolution frame is being decoded.  It is never measurement-valid.
        painter.fillRect(full_target.intersected(content), QColor("#E6E6E6"))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        target = QRectF()

        def draw_frame(
            frame: DigitalSlideRenderFrame | None,
            *,
            allow_other_focus: bool = False,
            coverage_only: bool = False,
        ) -> None:
            nonlocal target
            if (
                frame is None
                or (
                    not allow_other_focus
                    and frame.focus_index != self._focus_index
                )
                or frame.image.isNull()
            ):
                return
            x, y, width, height = frame.source_rect
            top_left = self.image_to_widget(Point(x, y))
            frame_target = QRectF(
                top_left.x(),
                top_left.y(),
                width * self._zoom,
                height * self._zoom,
            )
            if not frame_target.intersects(content):
                return
            if coverage_only:
                coverage_region = QRegion()
                for left, top, coverage_width, coverage_height in frame.coverage_rects:
                    coverage_top_left = self.image_to_widget(Point(left, top))
                    coverage_target = QRectF(
                        coverage_top_left.x(),
                        coverage_top_left.y(),
                        coverage_width * self._zoom,
                        coverage_height * self._zoom,
                    )
                    coverage_rect = coverage_target.toAlignedRect().intersected(
                        content.toAlignedRect()
                    )
                    if not coverage_rect.isEmpty():
                        coverage_region = coverage_region.united(
                            QRegion(coverage_rect)
                        )
                if coverage_region.isEmpty():
                    return
                painter.save()
                painter.setClipRegion(coverage_region)
                painter.drawImage(frame_target, frame.image)
                painter.restore()
            else:
                painter.drawImage(frame_target, frame.image)
            target = frame_target

        # Retain the last real presentation until the requested focus has
        # produced real pixels.  This frame is intentionally outside
        # ``_current_presentation_frame()`` and cannot enable measurement or
        # pixel algorithms; it only prevents a flat white paint between wheel
        # input and the worker's first target-focus result.
        transition_active = self._focus_transition_frame is not None
        if transition_active:
            draw_frame(
                self._focus_transition_frame,
                allow_other_focus=True,
            )
        elif (
            self._focus_transition_image is not None
            and not self._focus_transition_image.isNull()
            and self._focus_transition_native_rect is not None
        ):
            native = self._focus_transition_native_rect
            top_left = self.image_to_widget(Point(native.x(), native.y()))
            transition_target = QRectF(
                top_left.x(),
                top_left.y(),
                native.width() * self._zoom,
                native.height() * self._zoom,
            )
            if transition_target.intersects(content):
                painter.drawImage(
                    transition_target,
                    self._focus_transition_image,
                )
                target = transition_target
            transition_active = True

        large_area = self.large_area_browse_active()
        sharp_visual_active = bool(
            transition_active
            or (
                self._previous_render_frame is not None
                and self._previous_render_frame.focus_index == self._focus_index
                and self._previous_render_frame.quality == "final"
            )
            or (
                self._render_frame is not None
                and self._render_frame.focus_index == self._focus_index
                and self._render_frame.quality == "final"
            )
        )
        preview = self._presentation_preview_frame
        if (
            (large_area or not sharp_visual_active)
            and preview is not None
            and preview.focus_index == self._focus_index
            and preview.quality != "placeholder"
        ):
            painter.drawImage(full_target, preview.image)
            target = full_target
        elif (
            (large_area or not sharp_visual_active)
            and not self._overview_image.isNull()
            and self._overview_focus_index == self._focus_index
        ):
            painter.drawImage(full_target, self._overview_image)
            target = full_target

        # The current coarse frame fills newly exposed source regions first.
        # Existing final frames are then painted above it at their true global
        # coordinates, so continuous large-area motion cannot blur content that
        # has already reached final display quality.
        if large_area or not sharp_visual_active:
            draw_frame(
                self._coarse_render_frame,
                coverage_only=bool(
                    transition_active
                    and self._coarse_render_frame is not None
                    and not self._coarse_render_frame.complete
                ),
            )
        # Historical frames retain their true slide coordinates; they are not
        # stretched to impersonate a destination that has not loaded yet.
        draw_frame(self._previous_render_frame)
        if (
            self._render_frame is not None
            and self._render_frame.generation != self._view_generation
        ):
            draw_frame(self._render_frame)
        if (
            self._render_frame is not None
            and self._render_frame.generation == self._view_generation
        ):
            draw_frame(self._render_frame)
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
        self._advance_view_generation()
        self._request_interactive_frames()
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
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._pixel_pointer_event_allowed(event.position())
        ):
            self._notify_vector_input_waiting()
            return
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
        if (
            self._tool_mode in _VECTOR_MEASUREMENT_TOOLS
            and not self._pixel_work_enabled
            and not self.vector_measurement_available(
                self.widget_to_image(event.position())
            )
        ):
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
        if (
            not was_panning
            and event.button() == Qt.MouseButton.LeftButton
            and not self._pixel_pointer_event_allowed(event.position())
        ):
            self._notify_vector_input_waiting()
            return
        previous = self._clamp_pointer_to_mounted_viewport
        self._clamp_pointer_to_mounted_viewport = self._has_pointer_edit_operation()
        try:
            super().mouseReleaseEvent(event)
        finally:
            self._clamp_pointer_to_mounted_viewport = previous
        if was_panning and not self._panning:
            self._navigation_velocity = Point(0.0, 0.0)
            self._publish_viewport_state(throttled=False)
            self._final_render_timer.start()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        # Double-click completes paths, but padding must not contribute a new
        # point or silently clamp an otherwise invalid click onto the border.
        point = self.widget_to_image(event.position())
        if (
            not self._point_in_image(point)
            or not self._pixel_pointer_event_allowed(event.position())
        ):
            return
        previous = self._clamp_pointer_to_mounted_viewport
        self._clamp_pointer_to_mounted_viewport = False
        try:
            super().mouseDoubleClickEvent(event)
        finally:
            self._clamp_pointer_to_mounted_viewport = previous

    def _pixel_pointer_event_allowed(self, position: QPointF) -> bool:
        if self._pixel_work_enabled:
            return True
        if self._tool_mode not in _VECTOR_MEASUREMENT_TOOLS:
            return False
        point = self.widget_to_image(position)
        return self._point_in_image(point) and self.vector_measurement_available(point)

    def _notify_vector_input_waiting(self) -> None:
        now = perf_counter()
        if now - self._last_vector_input_notice_at < 1.0:
            return
        self._last_vector_input_notice_at = now
        reason = (
            "目标焦层尚未覆盖当前位置，请等待低分辨率图像后继续测量。"
            if self._tool_mode in _VECTOR_MEASUREMENT_TOOLS
            else self._pixel_work_reason
        )
        if reason:
            self.browseNoticeRequested.emit(reason)

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
        self._request_interactive_frames()

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
                generation=int(self._view_generation),
                quality="coarse",
                preview_max_edge=_OVERVIEW_MAX_EDGE,
                priority=6,
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
        self._request_interactive_frames()
        self._update_pixel_work_state()
        self._publish_viewport_state(throttled=throttled)
