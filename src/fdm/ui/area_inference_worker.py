from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal, Slot

from fdm.services.area_inference import AreaInferenceService
from fdm.settings import AppSettings


@dataclass(slots=True)
class AreaInferenceRequest:
    document_id: str
    image_path: str
    model_name: str
    model_file: str


class AreaBatchInferenceWorker(QObject):
    progress = Signal(int, int, str)
    succeeded = Signal(str, object)
    failed = Signal(str, str, str)
    finished = Signal(bool, int, int)

    def __init__(self, requests: list[AreaInferenceRequest], *, settings: AppSettings) -> None:
        super().__init__()
        self._requests = list(requests)
        self._settings = settings.normalized_copy()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        service = AreaInferenceService()
        total = len(self._requests)
        completed_count = 0
        failed_count = 0
        for index, request in enumerate(self._requests, start=1):
            if self._cancelled:
                break
            self.progress.emit(index, total, request.image_path)
            try:
                result = service.infer_image(
                    image_path=request.image_path,
                    model_name=request.model_name,
                    model_file=request.model_file,
                    settings=self._settings,
                )
                self.succeeded.emit(request.document_id, result.instances)
            except Exception as exc:  # noqa: BLE001
                failed_count += 1
                self.failed.emit(request.document_id, request.image_path, str(exc))
            completed_count += 1
        self.finished.emit(self._cancelled, completed_count, failed_count)

