from __future__ import annotations

from time import perf_counter
from typing import Protocol, cast

from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QImage

from fdm.services.preview_analysis import FocusStackRenderConfig, log_preview_analysis_perf
from fdm.ui.preview_analysis_dialog import PreviewAnalysisDialog
from fdm.ui.preview_analysis_worker import FocusStackSessionWorker, MapBuildSessionWorker
from fdm.ui.thread_task_manager import TASK_PREVIEW_ANALYSIS, TaskStopResult, ThreadTaskManager


class PreviewAnalysisHost(Protocol):
    def _selected_capture_device(self) -> object | None: ...
    def _clear_magic_segment_sessions(self) -> None: ...
    def _create_preview_analysis_dialog(self, mode: str) -> PreviewAnalysisDialog: ...
    def _analysis_mode_label(self, mode: str) -> str: ...
    def _preview_analysis_finalize_message(self, mode: str) -> str: ...
    def _current_focus_stack_render_config(self) -> FocusStackRenderConfig: ...
    def _preview_analysis_interval_ms(self, mode: str) -> int: ...
    def _request_capture_analysis_frame(self, request_id: int) -> bool: ...
    def _on_preview_analysis_worker_preview(self, payload: object) -> None: ...
    def _on_preview_analysis_worker_finished(self, payload: object) -> None: ...
    def _on_preview_analysis_worker_failed(self, message: str) -> None: ...
    def _sync_preview_analysis_buttons(self) -> None: ...
    def _update_action_states(self) -> None: ...
    def _show_status_message(self, message: str, timeout_ms: int = 0) -> None: ...


class PreviewAnalysisTaskController:
    def __init__(
        self,
        host: PreviewAnalysisHost,
        task_manager: ThreadTaskManager,
        *,
        parent: QObject,
        frame_watchdog_ms: int = 3000,
    ) -> None:
        self._host = host
        self._tasks = task_manager
        self._timer = QTimer(parent)
        self._timer.timeout.connect(self.request_frame)
        self._frame_watchdog = QTimer(parent)
        self._frame_watchdog.setSingleShot(True)
        self._frame_watchdog.setInterval(max(1, int(frame_watchdog_ms)))
        self._frame_watchdog.timeout.connect(self._on_frame_watchdog_timeout)
        self.mode = "none"
        self.dialog: PreviewAnalysisDialog | object | None = None
        self.request_id = 0
        self.request_pending = False
        self.analysis_pending = False
        self.analysis_request_id: int | None = None
        self.request_started_at: float | None = None
        self.finalizing = False
        self._worker_override: object | None = None

    @property
    def worker(self) -> object | None:
        if self._worker_override is not None:
            return self._worker_override
        return self._tasks.worker(TASK_PREVIEW_ANALYSIS)

    @worker.setter
    def worker(self, value: object | None) -> None:
        self._worker_override = value

    def start_session(self, mode: str) -> None:
        if mode not in {"focus_stack", "map_build"}:
            return
        selected = self._host._selected_capture_device()
        if selected is None:
            return
        self._host._clear_magic_segment_sessions()
        self.mode = mode
        self.request_pending = False
        self.analysis_pending = False
        self.analysis_request_id = None
        self.request_started_at = None
        self.finalizing = False
        self.dialog = self._host._create_preview_analysis_dialog(mode)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

        def worker_factory() -> QObject:
            if mode == "focus_stack":
                return FocusStackSessionWorker(
                    device_id=selected.id,
                    device_name=selected.name,
                    render_config=self._host._current_focus_stack_render_config(),
                )
            return MapBuildSessionWorker(device_id=selected.id, device_name=selected.name)

        def connect(worker: object) -> None:
            analysis_worker = cast(FocusStackSessionWorker | MapBuildSessionWorker, worker)
            analysis_worker.previewUpdated.connect(self._host._on_preview_analysis_worker_preview)
            analysis_worker.finished.connect(self._host._on_preview_analysis_worker_finished)
            analysis_worker.failed.connect(self._host._on_preview_analysis_worker_failed)
            analysis_worker.frameProcessed.connect(self.on_frame_processed)
            analysis_worker.resourceLimitReached.connect(self.on_resource_limit_reached)

        self._tasks.ensure_persistent(
            TASK_PREVIEW_ANALYSIS,
            worker_factory=worker_factory,
            connect_signals=connect,
            cancel_callback=_cancel_worker,
        )
        self._timer.setInterval(self._host._preview_analysis_interval_ms(mode))
        self._timer.start()
        self.request_frame()
        self._host._show_status_message(f"{self._host._analysis_mode_label(mode)}已启动", 3000)
        self._host._update_action_states()

    def teardown(self, *, cancel_worker: bool, status_message: str | None = None) -> TaskStopResult:
        self._timer.stop()
        self._frame_watchdog.stop()
        self.request_pending = False
        self.analysis_pending = False
        self.analysis_request_id = None
        self.request_started_at = None
        self.finalizing = False
        worker_override = self._worker_override
        if worker_override is not None and cancel_worker:
            _cancel_worker(worker_override)
        result = self._tasks.stop(TASK_PREVIEW_ANALYSIS, cancel=cancel_worker)
        if result.timed_out:
            self.finalizing = False
            self._host._show_status_message("分析任务未能在时限内退出，当前转换已阻止。", 5000)
            self._host._update_action_states()
            return result
        dialog = self.dialog
        self._worker_override = None
        self.dialog = None
        self.mode = "none"
        if dialog is not None:
            close_silently = getattr(dialog, "close_silently", None)
            if callable(close_silently):
                close_silently()
        if status_message:
            self._host._show_status_message(status_message, 4000)
        self._host._update_action_states()
        return result

    def cancel(self, *, message: str | None = None) -> TaskStopResult | None:
        if self.mode == "none":
            self._host._sync_preview_analysis_buttons()
            return None
        return self.teardown(cancel_worker=True, status_message=message)

    def finalize(self) -> None:
        worker = self.worker
        if self.mode == "none" or worker is None or self.finalizing:
            return
        self.finalizing = True
        self._timer.stop()
        self._frame_watchdog.stop()
        self.request_pending = False
        self.request_started_at = None
        if self.dialog is not None:
            busy_message = self._host._preview_analysis_finalize_message(self.mode)
            set_status = getattr(self.dialog, "set_status", None)
            if callable(set_status):
                set_status(busy_message)
            set_busy = getattr(self.dialog, "set_busy", None)
            if callable(set_busy):
                set_busy(True, busy_message, allow_cancel=True)
        _emit_signal(worker, "finalizeRequested")
        self._host._update_action_states()

    def request_frame(self) -> None:
        if (
            self.mode == "none"
            or self.worker is None
            or self.request_pending
            or self.analysis_pending
            or self.finalizing
        ):
            return
        self.request_id += 1
        request_id = self.request_id
        started_at = perf_counter()
        if self._host._request_capture_analysis_frame(request_id):
            self.request_pending = True
            self.request_started_at = started_at
            self._frame_watchdog.start()

    def on_frame_ready(self, request_id: int, image: object) -> None:
        if request_id != self.request_id or not self.request_pending:
            return
        started_at = self.request_started_at
        self._frame_watchdog.stop()
        self.request_pending = False
        self.request_started_at = None
        if started_at is not None:
            log_preview_analysis_perf(
                f"{self._host._analysis_mode_label(self.mode)} frame request",
                (perf_counter() - started_at) * 1000.0,
                detail=f"request_id={request_id}",
            )
        worker = self.worker
        if self.mode == "none" or worker is None or self.finalizing:
            return
        if isinstance(image, QImage) and not image.isNull():
            frame_submitted_with_id = getattr(worker, "frameSubmittedWithId", None)
            if frame_submitted_with_id is not None:
                self.analysis_pending = True
                self.analysis_request_id = request_id
                frame_submitted_with_id.emit(request_id, image.copy())
                return
            frame_submitted = getattr(worker, "frameSubmitted", None)
            if frame_submitted is not None:
                frame_submitted.emit(image.copy())

    def on_frame_failed(self, request_id: int, message: str) -> None:
        if request_id != self.request_id or not self.request_pending:
            return
        started_at = self.request_started_at
        self._frame_watchdog.stop()
        self.request_pending = False
        self.request_started_at = None
        if started_at is not None:
            log_preview_analysis_perf(
                f"{self._host._analysis_mode_label(self.mode)} frame request failed",
                (perf_counter() - started_at) * 1000.0,
                detail=f"request_id={request_id}, message={message}",
            )
        if self.dialog is not None:
            set_status = getattr(self.dialog, "set_status", None)
            if callable(set_status):
                set_status(message)
        self._host._show_status_message(message, 4000)

    def on_frame_processed(self, request_id: int) -> None:
        if request_id != self.analysis_request_id:
            return
        self.analysis_pending = False
        self.analysis_request_id = None

    def on_resource_limit_reached(self, reason: str) -> None:
        self._timer.stop()
        self.analysis_pending = False
        self.analysis_request_id = None
        message = reason or "分析已达到资源上限，可完成当前结果或取消。"
        if self.dialog is not None:
            set_status = getattr(self.dialog, "set_status", None)
            if callable(set_status):
                set_status(message)
        self._host._show_status_message(message, 5000)

    def _on_frame_watchdog_timeout(self) -> None:
        if not self.request_pending:
            return
        request_id = self.request_id
        self.request_pending = False
        self.request_started_at = None
        message = f"分析帧请求超时（request_id={request_id}），已恢复采样调度。"
        if self.dialog is not None:
            set_status = getattr(self.dialog, "set_status", None)
            if callable(set_status):
                set_status(message)
        self._host._show_status_message(message, 4000)


def _emit_signal(worker: object, signal_name: str) -> None:
    signal = getattr(worker, signal_name, None)
    if signal is None:
        return
    try:
        signal.emit()
    except Exception:
        pass


def _cancel_worker(worker: object) -> None:
    cancel = getattr(worker, "cancel", None)
    if callable(cancel):
        try:
            cancel()
            return
        except Exception:
            pass
    _emit_signal(worker, "cancelRequested")
