from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QImage

from fdm.models import ImageDocument, ProjectState
from fdm.settings import AppSettings
from fdm.services.area_inference import (
    DEFAULT_AREA_INFERENCE_TIMEOUT_S,
    area_inference_diagnostic_hint,
)
from fdm.ui.area_inference_worker import AreaBatchInferenceWorker, AreaInferenceRequest
from fdm.ui.fiber_quick_geometry_worker import FiberQuickGeometryWorker
from fdm.ui.image_loader import ImageBatchLoaderWorker, ImageLoadRequest
from fdm.ui.prompt_segmentation_worker import PromptSegmentationWorker
from fdm.ui.reference_instance_worker import ReferenceInstancePropagationWorker
from fdm.ui.thread_task_manager import (
    DEFAULT_WAIT_MS,
    REFERENCE_INSTANCE_WAIT_MS,
    TASK_AREA_INFERENCE,
    TASK_FIBER_QUICK_COMMIT_GEOMETRY,
    TASK_FIBER_QUICK_GEOMETRY,
    TASK_IMAGE_LOAD,
    TASK_PROMPT_SEGMENTATION,
    TASK_REFERENCE_INSTANCE,
    ThreadTaskManager,
    TaskStopResult,
)


@dataclass(slots=True)
class BatchLoadState:
    context_label: str
    total: int
    skipped_count: int = 0
    completed_count: int = 0
    loaded_count: int = 0
    failed_count: int = 0
    cancelled: bool = False
    failures: list[str] | None = None
    missing_paths: list[str] | None = None
    repaired_paths: list[str] | None = None
    pending_requests: dict[str, ImageLoadRequest] = field(default_factory=dict)
    generation: int = 0


@dataclass(slots=True)
class AreaInferenceBatchState:
    total: int
    model_name: str = ""
    update_project_group_templates: bool = True
    completed_count: int = 0
    failed_count: int = 0
    cancelled: bool = False
    failures: list[str] | None = None
    global_group_labels: list[str] = field(default_factory=list)
    pending_requests: dict[str, AreaInferenceRequest] = field(default_factory=dict)
    generation: int = 0


class BackgroundTaskHost(Protocol):
    project: ProjectState
    _app_settings: AppSettings
    _pending_project_load_snapshot: bool

    def _create_progress_dialog(self, *, title: str, label_text: str, maximum: int): ...
    def _add_loaded_document(self, request: ImageLoadRequest, image: QImage) -> None: ...
    def _show_batch_load_summary(self, state: BatchLoadState) -> None: ...
    def _mark_project_saved(self) -> None: ...
    def _apply_area_inference_result(
        self,
        document: ImageDocument,
        instances: list[object],
        *,
        global_group_labels: list[str] | None,
        model_name: str,
        update_project_group_templates: bool,
    ) -> None: ...
    def _show_area_inference_warning(self, message: str) -> None: ...
    def _show_status_message(self, message: str, timeout_ms: int = 0) -> None: ...
    def _register_unresolved_project_document(self, request: ImageLoadRequest, reason: str) -> None: ...
    def _on_prompt_segmentation_succeeded(self, document_id: str, request_id: int, result: object) -> None: ...
    def _on_prompt_segmentation_failed(self, document_id: str, request_id: int, reason: str) -> None: ...
    def _on_fiber_quick_geometry_succeeded(self, document_id: str, request_id: int, result: object) -> None: ...
    def _on_fiber_quick_geometry_failed(self, document_id: str, request_id: int, reason: str) -> None: ...
    def _on_fiber_quick_commit_geometry_succeeded(self, document_id: str, request_id: int, result: object) -> None: ...
    def _on_fiber_quick_commit_geometry_failed(self, document_id: str, request_id: int, reason: str) -> None: ...
    def _on_reference_instance_succeeded(self, document_id: str, request_id: int, result: object) -> None: ...
    def _on_reference_instance_failed(self, document_id: str, request_id: int, reason: str) -> None: ...


class BackgroundTaskController(QObject):
    def __init__(
        self,
        host: BackgroundTaskHost,
        task_manager: ThreadTaskManager,
        *,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._tasks = task_manager
        self._load_progress_dialog = None
        self._load_state: BatchLoadState | None = None
        self._area_infer_progress_dialog = None
        self._area_infer_state: AreaInferenceBatchState | None = None
        self._worker_overrides: dict[str, object] = {}
        self._generation = 0

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation

    def _effective_generation(self, explicit: int | None) -> int | None:
        if explicit is not None:
            return explicit
        sender = self.sender()
        if sender is None:
            return None
        value = sender.property("fdm_generation")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def load_state(self) -> BatchLoadState | None:
        return self._load_state

    @property
    def area_infer_state(self) -> AreaInferenceBatchState | None:
        return self._area_infer_state

    def is_image_loading(self) -> bool:
        return self._tasks.is_running(TASK_IMAGE_LOAD)

    def start_batch_image_load(
        self,
        requests: list[ImageLoadRequest],
        *,
        context_label: str,
        skipped_count: int,
        missing_paths: list[str] | None = None,
        repaired_paths: list[str] | None = None,
    ) -> None:
        generation = self._next_generation()
        for request in requests:
            request.generation = generation
        self._load_state = BatchLoadState(
            context_label=context_label,
            total=len(requests),
            skipped_count=skipped_count,
            failures=[],
            missing_paths=list(missing_paths or []),
            repaired_paths=list(repaired_paths or []),
            pending_requests={request.request_id: request for request in requests},
            generation=generation,
        )
        progress = self._host._create_progress_dialog(
            title=context_label,
            label_text="准备加载图片...",
            maximum=len(requests),
        )
        self._load_progress_dialog = progress

        def connect(worker: object) -> None:
            loader = cast(ImageBatchLoaderWorker, worker)
            loader.setProperty("fdm_generation", generation)
            loader.progress.connect(self._on_batch_load_progress)
            loader.loaded.connect(self._on_batch_load_loaded)
            loader.failedRequest.connect(self._on_batch_load_failed_request)
            loader.finished.connect(self._on_batch_load_finished)

        handle = self._tasks.start_one_shot(
            TASK_IMAGE_LOAD,
            worker_factory=lambda: ImageBatchLoaderWorker(requests, skipped_count=skipped_count),
            connect_signals=connect,
        )
        _connect_progress_cancel(progress, handle.worker)
        progress.show()

    def _on_batch_load_progress(self, index: int, total: int, path: str, generation: int | None = None) -> None:
        generation = self._effective_generation(generation)
        if generation is not None and (self._load_state is None or self._load_state.generation != generation):
            return
        if self._load_progress_dialog is None:
            return
        completed = self._load_state.completed_count if self._load_state is not None else 0
        self._load_progress_dialog.setMaximum(total)
        self._load_progress_dialog.setValue(completed)
        self._load_progress_dialog.setLabelText(f"正在加载 ({index}/{total})\n{Path(path).name}")

    def _on_batch_load_loaded(
        self,
        request: ImageLoadRequest,
        image: QImage,
        generation: int | None = None,
    ) -> None:
        generation = self._effective_generation(generation)
        state = self._load_state
        if generation is not None and (state is None or state.generation != generation):
            return
        if state is not None:
            state.pending_requests.pop(request.request_id, None)
            state.completed_count += 1
            state.loaded_count += 1
        self._host._add_loaded_document(request, image)
        if self._load_progress_dialog is not None and state is not None:
            self._load_progress_dialog.setValue(state.completed_count)

    def _on_batch_load_failed(self, path: str, reason: str) -> None:
        state = self._load_state
        if state is not None:
            state.completed_count += 1
            state.failed_count += 1
            if state.failures is not None:
                state.failures.append(f"{Path(path).name}: {reason}")
        if self._load_progress_dialog is not None and state is not None:
            self._load_progress_dialog.setValue(state.completed_count)

    def _on_batch_load_failed_request(
        self,
        request: ImageLoadRequest,
        reason: str,
        generation: int | None = None,
    ) -> None:
        generation = self._effective_generation(generation)
        state = self._load_state
        if generation is not None and (state is None or state.generation != generation):
            return
        if state is not None:
            state.pending_requests.pop(request.request_id, None)
        self._on_batch_load_failed(request.path, reason)
        register = getattr(self._host, "_register_unresolved_project_document", None)
        if request.document is not None and callable(register):
            register(request, reason)

    def _on_batch_load_finished(
        self,
        cancelled: bool,
        loaded_count: int,
        skipped_count: int,
        failed_count: int,
        generation: int | None = None,
    ) -> None:
        generation = self._effective_generation(generation)
        state = self._load_state
        if state is None or (generation is not None and state.generation != generation):
            return
        state.cancelled = cancelled
        if cancelled:
            register = getattr(self._host, "_register_unresolved_project_document", None)
            if callable(register):
                for request in list(state.pending_requests.values()):
                    if request.document is not None:
                        register(request, "加载已取消")
        state.pending_requests.clear()
        state.loaded_count = loaded_count
        state.skipped_count = skipped_count
        state.failed_count = failed_count
        state.completed_count = state.total
        self._close_load_progress()
        self._host._show_batch_load_summary(state)
        if self._host._pending_project_load_snapshot:
            self._host._mark_project_saved()
            self._host._pending_project_load_snapshot = False
        self._load_state = None

    def start_area_inference_batch(
        self,
        requests: list[AreaInferenceRequest],
        *,
        model_name: str,
    ) -> None:
        generation = self._next_generation()
        for request in requests:
            request.generation = generation
        pending_requests = {request.request_id: request for request in requests}
        if len(pending_requests) != len(requests):
            raise ValueError("面积识别请求的 request_id 必须唯一。")
        self._area_infer_state = AreaInferenceBatchState(
            total=len(requests),
            model_name=model_name,
            update_project_group_templates=len(requests) > 1,
            failures=[],
            pending_requests=pending_requests,
            generation=generation,
        )
        progress = self._host._create_progress_dialog(
            title="面积自动识别",
            label_text=(
                f"正在识别 (1/{len(requests)}) · 已完成 0/{len(requests)}"
                f" · 最长等待 {DEFAULT_AREA_INFERENCE_TIMEOUT_S:g} 秒\n"
                f"{Path(requests[0].image_path).name}"
            ),
            maximum=len(requests),
        )
        self._area_infer_progress_dialog = progress

        def connect(worker: object) -> None:
            area_worker = cast(AreaBatchInferenceWorker, worker)
            area_worker.setProperty("fdm_generation", generation)
            area_worker.progress.connect(self._on_area_inference_progress)
            area_worker.succeeded.connect(self._on_area_inference_succeeded)
            area_worker.failed.connect(self._on_area_inference_failed)
            area_worker.finished.connect(self._on_area_inference_finished)

        handle = self._tasks.start_one_shot(
            TASK_AREA_INFERENCE,
            worker_factory=lambda: AreaBatchInferenceWorker(requests, settings=self._host._app_settings),
            connect_signals=connect,
        )
        _connect_progress_cancel(progress, handle.worker)
        progress.show()

    def _on_area_inference_progress(
        self,
        index: int,
        total: int,
        path: str,
        request_id: str = "",
        generation: int | None = None,
    ) -> None:
        generation = self._effective_generation(generation)
        state = self._area_infer_state
        request = state.pending_requests.get(request_id) if state is not None else None
        if (
            state is None
            or generation is None
            or state.generation != generation
            or request is None
            or request.generation != generation
            or request.image_path != path
        ):
            return
        if self._area_infer_progress_dialog is None:
            return
        completed = state.completed_count
        self._area_infer_progress_dialog.setMaximum(total)
        self._area_infer_progress_dialog.setValue(completed)
        self._area_infer_progress_dialog.setLabelText(
            f"正在识别 ({index}/{total}) · 已完成 {completed}/{total}"
            f" · 最长等待 {DEFAULT_AREA_INFERENCE_TIMEOUT_S:g} 秒\n{Path(path).name}"
        )

    def _on_area_inference_succeeded(
        self,
        document_id: str,
        instances: object,
        request_id: str = "",
        generation: int | None = None,
    ) -> None:
        generation = self._effective_generation(generation)
        state = self._area_infer_state
        request = state.pending_requests.get(request_id) if state is not None else None
        if (
            state is None
            or generation is None
            or state.generation != generation
            or request is None
            or request.generation != generation
            or request.document_id != document_id
        ):
            return
        state.pending_requests.pop(request_id, None)
        state.completed_count += 1
        document = self._host.project.get_document(document_id)
        if document is not None and isinstance(instances, list):
            self._host._apply_area_inference_result(
                document,
                instances,
                global_group_labels=state.global_group_labels,
                model_name=state.model_name,
                update_project_group_templates=bool(state.update_project_group_templates),
            )
        if self._area_infer_progress_dialog is not None:
            self._area_infer_progress_dialog.setMaximum(state.total)
            self._area_infer_progress_dialog.setValue(state.completed_count)

    def _on_area_inference_failed(
        self,
        document_id: str,
        path: str,
        reason: str,
        request_id: str = "",
        generation: int | None = None,
    ) -> None:
        generation = self._effective_generation(generation)
        state = self._area_infer_state
        request = state.pending_requests.get(request_id) if state is not None else None
        if (
            state is None
            or generation is None
            or state.generation != generation
            or request is None
            or request.generation != generation
            or request.document_id != document_id
            or request.image_path != path
        ):
            return
        state.pending_requests.pop(request_id, None)
        state.completed_count += 1
        state.failed_count += 1
        if state.failures is not None:
            state.failures.append(f"{Path(path).name}: {reason}")
        if self._area_infer_progress_dialog is not None:
            self._area_infer_progress_dialog.setMaximum(state.total)
            self._area_infer_progress_dialog.setValue(state.completed_count)

    def _on_area_inference_finished(
        self,
        cancelled: bool,
        completed_count: int,
        failed_count: int,
        generation: int | None = None,
    ) -> None:
        generation = self._effective_generation(generation)
        state = self._area_infer_state
        if state is None or generation is None or state.generation != generation:
            return
        state.cancelled = cancelled
        state.completed_count = completed_count
        state.failed_count = failed_count
        state.pending_requests.clear()
        self._close_area_progress()

        if state.failures:
            self._host._show_area_inference_warning(
                "以下图片识别失败:\n"
                + "\n".join(state.failures[:10])
                + "\n\n"
                + area_inference_diagnostic_hint()
            )
        if completed_count > 0:
            self._host._show_status_message(
                f"面积自动识别已处理 {completed_count - failed_count} / {completed_count} 张图片",
                6000,
            )
        self._area_infer_state = None

    def ensure_prompt_segmentation_worker(self) -> PromptSegmentationWorker | object:
        override = self._worker_overrides.get(TASK_PROMPT_SEGMENTATION)
        if override is not None:
            return override

        def connect(worker: object) -> None:
            prompt_worker = cast(PromptSegmentationWorker, worker)
            prompt_worker.succeeded.connect(self._host._on_prompt_segmentation_succeeded)
            prompt_worker.failed.connect(self._host._on_prompt_segmentation_failed)

        handle = self._tasks.ensure_persistent(
            TASK_PROMPT_SEGMENTATION,
            worker_factory=PromptSegmentationWorker,
            connect_signals=connect,
        )
        return handle.worker

    def ensure_fiber_quick_geometry_worker(self) -> FiberQuickGeometryWorker | object:
        override = self._worker_overrides.get(TASK_FIBER_QUICK_GEOMETRY)
        if override is not None:
            return override

        def connect(worker: object) -> None:
            geometry_worker = cast(FiberQuickGeometryWorker, worker)
            geometry_worker.succeeded.connect(self._host._on_fiber_quick_geometry_succeeded)
            geometry_worker.failed.connect(self._host._on_fiber_quick_geometry_failed)

        handle = self._tasks.ensure_persistent(
            TASK_FIBER_QUICK_GEOMETRY,
            worker_factory=FiberQuickGeometryWorker,
            connect_signals=connect,
        )
        return handle.worker

    def ensure_fiber_quick_commit_geometry_worker(self) -> FiberQuickGeometryWorker | object:
        override = self._worker_overrides.get(TASK_FIBER_QUICK_COMMIT_GEOMETRY)
        if override is not None:
            return override

        def connect(worker: object) -> None:
            geometry_worker = cast(FiberQuickGeometryWorker, worker)
            geometry_worker.succeeded.connect(self._host._on_fiber_quick_commit_geometry_succeeded)
            geometry_worker.failed.connect(self._host._on_fiber_quick_commit_geometry_failed)

        handle = self._tasks.ensure_persistent(
            TASK_FIBER_QUICK_COMMIT_GEOMETRY,
            worker_factory=lambda: FiberQuickGeometryWorker(coalesce_latest=False),
            connect_signals=connect,
        )
        return handle.worker

    def ensure_reference_instance_worker(self) -> ReferenceInstancePropagationWorker | object:
        override = self._worker_overrides.get(TASK_REFERENCE_INSTANCE)
        if override is not None:
            return override

        def connect(worker: object) -> None:
            reference_worker = cast(ReferenceInstancePropagationWorker, worker)
            reference_worker.succeeded.connect(self._host._on_reference_instance_succeeded)
            reference_worker.failed.connect(self._host._on_reference_instance_failed)

        handle = self._tasks.ensure_persistent(
            TASK_REFERENCE_INSTANCE,
            worker_factory=ReferenceInstancePropagationWorker,
            connect_signals=connect,
            wait_ms=REFERENCE_INSTANCE_WAIT_MS,
        )
        return handle.worker

    def worker(self, task_name: str) -> object | None:
        override = self._worker_overrides.get(task_name)
        if override is not None:
            return override
        return self._tasks.worker(task_name)

    def set_worker_override(self, task_name: str, worker: object | None) -> None:
        if worker is None:
            self._worker_overrides.pop(task_name, None)
        else:
            self._worker_overrides[task_name] = worker

    def register_external_thread(self, task_name: str, thread) -> None:
        self._tasks.register_external(task_name, thread=thread, wait_ms=_wait_ms_for_task(task_name))

    def thread(self, task_name: str):
        return self._tasks.thread(task_name)

    def shutdown_all(
        self,
        *,
        document_ids: list[str],
        commit_document_ids: list[str],
    ) -> list[TaskStopResult]:
        results: list[TaskStopResult] = []
        image_result = self._tasks.stop(TASK_IMAGE_LOAD, cancel=True)
        results.append(image_result)
        if image_result.stopped:
            self._close_load_progress()
            self._load_state = None

        area_result = self._tasks.stop(TASK_AREA_INFERENCE, cancel=True)
        results.append(area_result)
        if area_result.stopped:
            self._close_area_progress()
            self._area_infer_state = None

        geometry_worker = self.worker(TASK_FIBER_QUICK_GEOMETRY)
        _cancel_documents(geometry_worker, document_ids)
        results.append(self._tasks.stop(TASK_FIBER_QUICK_GEOMETRY, cancel=False))

        commit_worker = self.worker(TASK_FIBER_QUICK_COMMIT_GEOMETRY)
        _cancel_documents(commit_worker, commit_document_ids)
        results.append(self._tasks.stop(TASK_FIBER_QUICK_COMMIT_GEOMETRY, cancel=False))

        prompt_worker = self.worker(TASK_PROMPT_SEGMENTATION)
        _cancel_documents(prompt_worker, document_ids)
        results.append(self._tasks.stop(TASK_PROMPT_SEGMENTATION, cancel=False))

        reference_worker = self.worker(TASK_REFERENCE_INSTANCE)
        _cancel_documents(reference_worker, document_ids)
        results.append(self._tasks.stop(TASK_REFERENCE_INSTANCE, cancel=False))
        if all(result.stopped for result in results):
            self._worker_overrides.clear()
        return results

    def _close_load_progress(self) -> None:
        if self._load_progress_dialog is None:
            return
        self._load_progress_dialog.setValue(self._load_state.total if self._load_state is not None else 0)
        self._load_progress_dialog.close()
        self._load_progress_dialog.deleteLater()
        self._load_progress_dialog = None

    def _close_area_progress(self) -> None:
        if self._area_infer_progress_dialog is None:
            return
        self._area_infer_progress_dialog.setValue(self._area_infer_state.total if self._area_infer_state is not None else 0)
        self._area_infer_progress_dialog.close()
        self._area_infer_progress_dialog.deleteLater()
        self._area_infer_progress_dialog = None


def _cancel_documents(worker: object | None, document_ids: list[str]) -> None:
    cancel_document = getattr(worker, "cancel_document", None)
    if not callable(cancel_document):
        return
    for document_id in document_ids:
        try:
            cancel_document(document_id)
        except Exception:
            continue


def _wait_ms_for_task(task_name: str) -> int:
    if task_name == TASK_REFERENCE_INSTANCE:
        return REFERENCE_INSTANCE_WAIT_MS
    return DEFAULT_WAIT_MS


def _connect_progress_cancel(progress: object, worker: object) -> None:
    cancel = getattr(worker, "request_cancel", None)
    if not callable(cancel):
        cancel = getattr(worker, "cancel", None)
    canceled = getattr(progress, "canceled", None)
    connect = getattr(canceled, "connect", None)
    if not callable(cancel) or not callable(connect):
        return
    try:
        connect(cancel, Qt.ConnectionType.DirectConnection)
    except TypeError:
        connect(cancel)
