"""Camera preview with native-pixel geometry and an optional fitted raster."""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage

from fdm.ui.canvas import DocumentCanvas
from fdm.ui.view_transform import CanvasZoomMode


class AcquisitionPreviewCanvas(DocumentCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fit_preview_width: int | None = None
        self._fit_raster = QImage()
        self._fit_frame_key = None

    def set_fit_preview_width(self, width: int | None) -> None:
        if self._fit_preview_width != width:
            self._fit_preview_width = width
            self._fit_raster = QImage()
            self._fit_frame_key = None
            self.update()

    def display_raster_size(self) -> QSize:
        if self._image is None:
            return QSize()
        size = self._image.size()
        limit = self._fit_preview_width
        if self.zoom_mode() is CanvasZoomMode.FIT and limit and size.width() > limit:
            return QSize(limit, max(1, size.height() * limit // size.width()))
        return size

    def _base_image_raster(self) -> QImage:
        # All transforms, including the first pan out of FIT, use camera
        # coordinates. Only the pixels painted into that rectangle may shrink.
        size = self.display_raster_size()
        if size == self._image.size():
            return self._image
        key = self._image.cacheKey()
        if key != self._fit_frame_key:
            self._fit_raster = self._image.scaled(
                size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self._fit_frame_key = key
        return self._fit_raster

    def clear_document(self) -> None:
        self._fit_raster = QImage()
        self._fit_frame_key = None
        super().clear_document()
