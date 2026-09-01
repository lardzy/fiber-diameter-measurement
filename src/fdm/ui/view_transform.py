from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QRectF


MIN_VIEW_ZOOM = 1.0e-4
MAX_VIEW_ZOOM = 40.0


class CanvasZoomMode(str, Enum):
    """Describe how a canvas zoom should react to viewport size changes."""

    FIT = "fit"
    NATIVE_FIELD_FIT = "native_field_fit"
    ACTUAL = "actual"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class CanvasViewportSnapshot:
    """An immutable publication of the canvas' current view transform.

    Rectangles use global image coordinates.  For an ordinary image the full
    and mounted rectangles are identical.  A virtualized digital slide keeps
    the full slide in ``full_image_rect``, publishes the camera field in
    ``visible_image_rect`` and its fixed pixel-work field in
    ``native_viewport_rect``.
    """

    document_id: str
    full_image_rect: QRectF
    mounted_image_rect: QRectF
    visible_image_rect: QRectF
    zoom: float
    mode: CanvasZoomMode
    device_pixel_ratio: float
    focus_index: int | None = None
    native_viewport_rect: QRectF | None = None
    pixel_work_enabled: bool = True
