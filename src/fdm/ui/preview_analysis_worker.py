from __future__ import annotations

from time import perf_counter

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage

from fdm.services.preview_analysis import (
    FocusStackAnalyzer,
    FocusStackFinalResult,
    FocusStackReport,
    MapBuildAnalyzer,
    MapBuildFinalResult,
    MapBuildReport,
    log_preview_analysis_perf,
)


class FocusStackSessionWorker(QObject):
    frameSubmitted = Signal(object)
    finalizeRequested = Signal()
    cancelRequested = Signal()
    previewUpdated = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, *, device_id: str, device_name: str) -> None:
        super().__init__()
        self._analyzer = FocusStackAnalyzer(device_id=device_id, device_name=device_name)
        self._cancelled = False
        self._post_sharpen_enabled = False
        self.frameSubmitted.connect(self.add_frame, Qt.ConnectionType.QueuedConnection)
        self.finalizeRequested.connect(self.finalize, Qt.ConnectionType.QueuedConnection)
        self.cancelRequested.connect(self.cancel, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def add_frame(self, image: QImage) -> None:
        if self._cancelled or not isinstance(image, QImage) or image.isNull():
            return
        started_at = perf_counter()
        try:
            report = self._analyzer.add_frame(image)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        log_preview_analysis_perf(
            "Focus stack preprocess",
            (perf_counter() - started_at) * 1000.0,
            detail=(
                f"sampled={report.sampled_frames}, "
                f"accepted={report.accepted_frames}, "
                f"size={report.preview_image.width()}x{report.preview_image.height()}"
            ),
        )
        self.previewUpdated.emit(report)

    @Slot()
    def finalize(self) -> None:
        if self._cancelled:
            return
        try:
            result = self._analyzer.finalize(post_sharpen=self._post_sharpen_enabled)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)

    @Slot()
    def cancel(self) -> None:
        self._cancelled = True

    def set_post_sharpen_enabled(self, enabled: bool) -> None:
        self._post_sharpen_enabled = bool(enabled)


class MapBuildSessionWorker(QObject):
    frameSubmitted = Signal(object)
    finalizeRequested = Signal()
    cancelRequested = Signal()
    previewUpdated = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, *, device_id: str, device_name: str) -> None:
        super().__init__()
        self._analyzer = MapBuildAnalyzer(device_id=device_id, device_name=device_name)
        self._cancelled = False
        self.frameSubmitted.connect(self.add_frame, Qt.ConnectionType.QueuedConnection)
        self.finalizeRequested.connect(self.finalize, Qt.ConnectionType.QueuedConnection)
        self.cancelRequested.connect(self.cancel, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def add_frame(self, image: QImage) -> None:
        if self._cancelled or not isinstance(image, QImage) or image.isNull():
            return
        started_at = perf_counter()
        try:
            report = self._analyzer.add_frame(image)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        log_preview_analysis_perf(
            "Map build preprocess",
            (perf_counter() - started_at) * 1000.0,
            detail=(
                f"sampled={report.sampled_frames}, "
                f"accepted={report.accepted_frames}, "
                f"tiles={report.tile_count}, "
                f"motion_state={report.motion_state}, "
                f"stable={report.stable_streak}, "
                f"translation_px={report.translation_px:.2f}, "
                f"response={report.correlation_response:.4f}, "
                f"transition_pred=({report.transition_predicted_dx:.2f},{report.transition_predicted_dy:.2f}), "
                f"transition_refined=({report.transition_refined_dx:.2f},{report.transition_refined_dy:.2f}), "
                f"transition_method={report.transition_method or 'none'}, "
                f"transition_ncc={report.transition_ncc:.4f}, "
                f"transition_delta={report.transition_delta_from_prediction:.2f}"
            ),
        )
        self.previewUpdated.emit(report)

    @Slot()
    def finalize(self) -> None:
        if self._cancelled:
            return
        try:
            result = self._analyzer.finalize()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)

    @Slot()
    def cancel(self) -> None:
        self._cancelled = True
