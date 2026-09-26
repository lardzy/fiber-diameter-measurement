"""One interactive save writer and one replaceable pending snapshot."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot

from fdm.ui.project_session_controller import (
    PreparedProjectSave,
    ProjectSaveResult,
    ProjectWriteResult,
    write_project_save,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SaveStatus:
    phase: str = "idle"
    text: str = "尚未保存"
    path: str = ""
    saved_at: str = ""
    detail: str = ""
    transition: str = ""


class ProjectSaveWorker(QObject):
    completed = Signal(int, int, object)
    finished = Signal()

    def __init__(
        self, request: PreparedProjectSave, request_number: int, epoch: int
    ) -> None:
        super().__init__()
        self.request = request
        self.request_number = request_number
        self.epoch = epoch

    @Slot()
    def run(self) -> None:
        try:
            outcome = write_project_save(self.request)
        except Exception as exc:
            # Keep the Qt lifecycle balanced even on unexpected failures.
            logger.exception("Unexpected project writer failure")
            outcome = ProjectWriteResult(
                ProjectSaveResult(
                    False, path=self.request.target_path, message=str(exc)
                )
            )
        self.completed.emit(self.request_number, self.epoch, outcome)
        self.finished.emit()


class ProjectSaveCoordinator(QObject):
    statusChanged = Signal(object)
    completed = Signal(object)
    busyChanged = Signal(bool)

    def __init__(self, host, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.host = host
        self.status = SaveStatus()
        self.active: PreparedProjectSave | None = None
        self.pending: PreparedProjectSave | None = None
        self._thread: QThread | None = None
        self._worker: ProjectSaveWorker | None = None
        self._outcome: ProjectWriteResult | None = None
        self._preparing = False
        self._request_number = 0
        self._epoch = 0
        self._active_epoch = 0
        self._transition: Callable[[], object] | None = None
        self._transition_label = ""
        self.last_metrics: dict[str, object] = {}
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(3000)
        self._settle_timer.timeout.connect(self._settle)

    @property
    def busy(self) -> bool:
        return self.active is not None or self._preparing

    def _dirty(self) -> bool:
        return self.host._project_dirty() or any(
            self.host._document_has_unsaved_project_changes(document)
            for document in self.host.project.documents
        )

    def _set_status(
        self,
        phase: str,
        text: str,
        *,
        detail: str = "",
        path_override: str | None = None,
    ) -> None:
        if path_override is not None:
            path = path_override
        elif self.active is not None:
            path = str(self.active.target_path)
        elif phase == "failed" and self.status.phase == "failed":
            path = self.status.path
        else:
            path = str(self.host._project_path or "")
        status = SaveStatus(
            phase, text, path, self.status.saved_at, detail, self._transition_label
        )
        if status != self.status:
            self.status = status
            self.statusChanged.emit(status)

    def refresh_dirty(self) -> None:
        if self.busy or self.status.phase == "failed":
            return
        if self._dirty():
            if self.status.phase != "saved_newer" or not self._settle_timer.isActive():
                self._set_status("dirty", "未保存")
        elif not self._settle_timer.isActive():
            self._set_status(
                "clean" if self.host._project_path else "idle",
                "已保存" if self.host._project_path else "尚未保存",
            )

    def _settle(self) -> None:
        if self.busy or self.status.phase == "failed":
            return
        self._set_status(
            "dirty" if self._dirty() else "clean",
            "未保存" if self._dirty() else "已保存",
        )

    def request_save(self, path: str | None = None) -> bool:
        if self._preparing:
            return False
        previous_status = self.status
        self._preparing = True
        try:
            # Before the first save commits, repeated Ctrl+S uses its chosen
            # destination without reopening the file picker.
            target = path or None
            if target is None and self.active is not None:
                target = str(
                    self.pending.target_path
                    if self.pending
                    else self.active.target_path
                )
            elif target is None and self.status.phase == "failed":
                target = self.status.path or None
            request = self.host.project_session_controller.prepare_save(target)
        except Exception as exc:
            logger.exception("Project save preparation failed")
            request = ProjectSaveResult(False, message=str(exc))
        finally:
            self._preparing = False
        if isinstance(request, ProjectSaveResult):
            if request.cancelled and self.active is None:
                self.cancel_transition()
                self.status = previous_status
                self.statusChanged.emit(previous_status)
            elif not request.cancelled:
                if self.active is not None:
                    self._set_status(
                        "saving",
                        "正在保存…",
                        detail=f"后续保存未能准备：{request.message}",
                    )
                else:
                    self._settle_timer.stop()
                    self.cancel_transition()
                    self._set_status(
                        "failed",
                        "保存失败",
                        detail=request.message,
                        path_override=str(request.path) if request.path else None,
                    )
            return False
        if self.active is not None:
            # Latest keypress wins, including undo back to the active snapshot.
            if request.equivalent(self.active):
                self.pending = None
            elif self.pending is None or not request.equivalent(self.pending):
                self.pending = request
            self._set_status(
                "saving", "正在保存…", detail="另一次保存已排队" if self.pending else ""
            )
            return True
        self._start(request)
        return True

    def _start(self, request: PreparedProjectSave) -> None:
        self._settle_timer.stop()
        self.active = request
        self._active_epoch = self._epoch
        self._request_number += 1
        self._outcome = None
        thread = QThread(self)
        worker = ProjectSaveWorker(request, self._request_number, self._active_epoch)
        worker.moveToThread(thread)
        worker.completed.connect(self._receive)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.finished.connect(worker.deleteLater)
        thread.started.connect(worker.run)
        thread.finished.connect(self._finish)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        self._set_status("saving", "正在保存…")
        self.busyChanged.emit(True)
        thread.start()

    @Slot(int, int, object)
    def _receive(
        self, request_number: int, epoch: int, outcome: ProjectWriteResult
    ) -> None:
        if (
            self.active is not None
            and request_number == self._request_number
            and epoch == self._epoch
        ):
            self._outcome = outcome

    @Slot()
    def _finish(self) -> None:
        request, outcome = self.active, self._outcome
        self._thread = self._worker = None
        if request is None:
            return
        valid = self._active_epoch == self._epoch and request.project_identity == id(
            self.host.project
        )
        started = perf_counter()
        if outcome is None:
            outcome = ProjectWriteResult(
                ProjectSaveResult(False, message="保存线程未返回结果。")
            )
        if valid and outcome.result:
            try:
                self.host.project_session_controller.apply_save(request, outcome)
            except Exception as exc:
                # The disk commit already happened. Do not retry automatically,
                # close the workspace, or claim the old file was restored.
                logger.exception("Project committed but UI acknowledgement failed")
                self.active = self.pending = None
                self._outcome = None
                self.cancel_transition()
                self._set_status(
                    "failed",
                    "已写入，状态异常",
                    detail=f"项目文件已写入，但界面状态未能完整更新。请重试保存或重新打开核对。\n{exc}",
                )
                self.busyChanged.emit(False)
                self.completed.emit(outcome)
                return
        self.last_metrics = {
            "request": self._request_number,
            "prepare_ms": request.preparation_ms,
            "write_ms": outcome.write_ms,
            "apply_ms": (perf_counter() - started) * 1000,
            "success": outcome.result.success,
            "stale": not valid,
        }
        logger.info("project_save %s", self.last_metrics)
        self.active = None
        self._outcome = None
        if not valid:
            self.pending = None
            self.cancel_transition()
            self._settle_timer.stop()
            self.status = SaveStatus()
            self.refresh_dirty()
        elif not outcome.result:
            self.pending = None
            self.cancel_transition()
            self._set_status(
                "failed",
                "保存失败",
                detail=outcome.result.message,
                path_override=str(request.target_path),
            )
        else:
            self.status = SaveStatus(
                saved_at=datetime.now(UTC).astimezone().strftime("%H:%M:%S")
            )
            if self.pending is not None:
                pending, self.pending = self.pending, None
                # Its frozen pixels/content remain unchanged; reuse any newly
                # verified files, avoiding a second encode on the first save.
                from dataclasses import replace

                pending = replace(
                    pending,
                    receipts={
                        **pending.receipts,
                        **dict(outcome.assets.raster_receipts),
                    },
                )
                self._start(pending)
                self.completed.emit(outcome)
                return
            dirty = self._dirty()
            self._set_status(
                "saved_newer" if dirty else "saved",
                "本次已保存，仍有新修改" if dirty else "已保存",
            )
            self._settle_timer.start()
        self.busyChanged.emit(False)
        self.completed.emit(outcome)
        if valid and outcome.result and self._transition is not None:
            callback = self._transition
            epoch = self._epoch
            self.cancel_transition()

            # Leave the completion slot before entering any normal close prompt.
            def resume() -> None:
                if epoch != self._epoch:
                    return
                if not self.defer_transition(callback, "继续关闭或切换"):
                    callback()

            QTimer.singleShot(0, resume)

    def defer_transition(self, callback: Callable[[], object], label: str) -> bool:
        if not self.busy:
            return False
        self._transition, self._transition_label = callback, label
        self._set_status(
            "saving", f"保存后{label}…", detail=f"保存结束后{label}；点击可取消此操作"
        )
        return True

    def cancel_transition(self) -> None:
        was_waiting = bool(self._transition_label)
        self._transition = None
        self._transition_label = ""
        if was_waiting and self.status.phase == "saving":
            self._set_status(
                "saving", "正在保存…", detail="另一次保存已排队" if self.pending else ""
            )
        else:
            self._set_status(
                self.status.phase, self.status.text, detail=self.status.detail
            )

    def reset(self) -> None:
        self._epoch += 1
        self.pending = None
        self._transition = None
        self._transition_label = ""
        self._settle_timer.stop()
        self.status = SaveStatus()
        self.statusChanged.emit(self.status)
