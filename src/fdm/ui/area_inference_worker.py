from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

from PySide6.QtCore import QObject, Signal, Slot

from fdm.cancellation import CancellationSource, CancellationToken
from fdm.services.area_inference import (
    DEFAULT_AREA_INFERENCE_TIMEOUT_S,
    AreaInferenceCancelledError,
    AreaInferenceService,
)
from fdm.runtime_logging import append_runtime_log
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
        total = len(self._requests)
        completed_count = 0
        failed_count = 0
        append_runtime_log(
            "Area inference batch started",
            (
                f"generation={self._generation}, total={total}, transport=one_shot, "
                f"device={self._settings.area_infer_device}, timeout_s={self._timeout_s:g}"
            ),
        )
        try:
            for index, request in enumerate(self._requests, start=1):
                if self._cancellation.token.is_cancelled:
                    break
                request_started_at = monotonic()
                append_runtime_log(
                    "Area inference request started",
                    (
                        f"request_id={request.request_id}, generation={request.generation}, "
                        f"index={index}/{total}, model={request.model_file}, "
                        f"image={request.image_path}"
                    ),
                )
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
                    )
                    if self._cancellation.token.is_cancelled:
                        break
                    self.succeeded.emit(
                        request.document_id,
                        result.instances,
                        request.request_id,
                        request.generation,
                    )
                    append_runtime_log(
                        "Area inference request completed",
                        (
                            f"request_id={request.request_id}, elapsed_s="
                            f"{monotonic() - request_started_at:.3f}, "
                            f"instances={len(result.instances)}"
                        ),
                    )
                    if self._cancellation.token.is_cancelled:
                        break
                except AreaInferenceCancelledError:
                    append_runtime_log(
                        "Area inference request cancelled",
                        (
                            f"request_id={request.request_id}, elapsed_s="
                            f"{monotonic() - request_started_at:.3f}"
                        ),
                    )
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
                    append_runtime_log(
                        "Area inference request failed",
                        (
                            f"request_id={request.request_id}, elapsed_s="
                            f"{monotonic() - request_started_at:.3f}, error={exc}"
                        ),
                    )
                completed_count += 1
        finally:
            cancelled = self._cancellation.token.is_cancelled
            append_runtime_log(
                "Area inference batch finished",
                (
                    f"generation={self._generation}, completed={completed_count}/{total}, "
                    f"failed={failed_count}, cancelled={cancelled}"
                ),
            )
            self.finished.emit(
                cancelled,
                completed_count,
                failed_count,
                self._generation,
            )
