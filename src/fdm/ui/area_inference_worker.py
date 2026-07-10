from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from PySide6.QtCore import QObject, Signal, Slot

from fdm.cancellation import CancellationSource, CancellationToken
from fdm.services.area_inference import (
    DEFAULT_AREA_INFERENCE_TIMEOUT_S,
    AreaInferenceCancelledError,
    AreaInferenceService,
)
from fdm.settings import AppSettings


@dataclass(slots=True)
class AreaInferenceRequest:
    document_id: str
    image_path: str
    model_name: str
    model_file: str
    request_id: str = field(default_factory=lambda: uuid4().hex)
    generation: int = 0


class AreaBatchInferenceWorker(QObject):
    progress = Signal(int, int, str, str, int)
    succeeded = Signal(str, object, str, int)
    failed = Signal(str, str, str, str, int)
    finished = Signal(bool, int, int, int)

    def __init__(
        self,
        requests: list[AreaInferenceRequest],
        *,
        settings: AppSettings,
        timeout_s: float = DEFAULT_AREA_INFERENCE_TIMEOUT_S,
    ) -> None:
        super().__init__()
        self._requests = list(requests)
        generations = {int(request.generation) for request in self._requests}
        if len(generations) > 1:
            raise ValueError("同一面积识别批次的 generation 必须一致。")
        request_ids = [str(request.request_id) for request in self._requests]
        if len(request_ids) != len(set(request_ids)) or any(not item for item in request_ids):
            raise ValueError("同一面积识别批次的 request_id 必须非空且唯一。")
        self._generation = next(iter(generations), 0)
        self._settings = settings.normalized_copy()
        self._timeout_s = timeout_s
        self._cancellation = CancellationSource()

    @property
    def cancellation_token(self) -> CancellationToken:
        return self._cancellation.token

    def cancel(self) -> None:
        self._cancellation.cancel()

    def request_cancel(self) -> None:
        """Thread-safe cancellation entry point for direct UI connections."""

        self.cancel()

    @Slot()
    def run(self) -> None:
        service = AreaInferenceService()
        worker_session = service.create_batch_session(self._settings)
        total = len(self._requests)
        completed_count = 0
        failed_count = 0
        try:
            for index, request in enumerate(self._requests, start=1):
                if self._cancellation.token.is_cancelled:
                    break
                self.progress.emit(
                    index,
                    total,
                    request.image_path,
                    request.request_id,
                    request.generation,
                )
                try:
                    result = service.infer_image(
                        image_path=request.image_path,
                        model_name=request.model_name,
                        model_file=request.model_file,
                        settings=self._settings,
                        timeout_s=self._timeout_s,
                        cancellation_token=self._cancellation.token,
                        worker_session=worker_session,
                    )
                    if self._cancellation.token.is_cancelled:
                        break
                    self.succeeded.emit(
                        request.document_id,
                        result.instances,
                        request.request_id,
                        request.generation,
                    )
                    if self._cancellation.token.is_cancelled:
                        break
                except AreaInferenceCancelledError:
                    self._cancellation.cancel()
                    break
                except Exception as exc:  # noqa: BLE001
                    if self._cancellation.token.is_cancelled:
                        break
                    failed_count += 1
                    self.failed.emit(
                        request.document_id,
                        request.image_path,
                        str(exc),
                        request.request_id,
                        request.generation,
                    )
                completed_count += 1
        finally:
            worker_session.close()
        self.finished.emit(
            self._cancellation.token.is_cancelled,
            completed_count,
            failed_count,
            self._generation,
        )
