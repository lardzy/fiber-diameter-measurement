from __future__ import annotations

from collections import deque
from enum import Enum
from pathlib import Path
from typing import Protocol

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from fdm.application_launch import ApplicationOpenRequest, build_application_open_request
from fdm.lifecycle import AcquisitionDisposition, TransitionIntent
from fdm.services.digital_slide_store import DigitalSlideStore


class AssociatedSlideDisposition(str, Enum):
    ADD_TO_CURRENT = "add_to_current"
    STANDALONE_WORKSPACE = "standalone_workspace"
    CANCEL = "cancel"


class AssociatedFileOpenHost(Protocol):
    project: object
    _project_path: Path | None
    _transition_in_progress: bool

    def is_image_loading(self) -> bool: ...
    def _activate_from_external_request(self) -> None: ...
    def _associated_open_document_id(self, path: Path) -> str | None: ...
    def _set_current_document(self, document_id: str) -> None: ...
    def _load_project_from_path(self, path: str | Path): ...
    def _slide_acquisition_active(self) -> bool: ...
    def _prepare_transition(
        self,
        intent: TransitionIntent,
        *,
        disposition: AcquisitionDisposition | None = None,
    ): ...
    def _preflight_acquisition_disposition(
        self,
        intent: TransitionIntent,
    ) -> AcquisitionDisposition | None: ...
    def stop_live_preview(self) -> None: ...
    def _confirm_close_documents(self, documents: list[object]) -> bool: ...
    def _reset_workspace(self) -> None: ...
    def _open_image_requests(
        self,
        items: list[tuple[str, object | None]],
        *,
        context_label: str,
    ) -> None: ...
    def _show_status_message(self, message: str, timeout_ms: int = 0) -> None: ...


class AssociatedFileOpenController(QObject):
    POLL_INTERVAL_MS = 150

    def __init__(self, host: AssociatedFileOpenHost, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._host = host
        self._queue: deque[ApplicationOpenRequest] = deque()
        self._queued_path_keys: set[str] = set()
        self._seen_request_ids: deque[str] = deque(maxlen=256)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self.POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._dispatch_next)

    def enqueue(self, request: ApplicationOpenRequest) -> None:
        self._host._activate_from_external_request()
        if request.request_id in self._seen_request_ids:
            return
        self._seen_request_ids.append(request.request_id)
        if not request.paths:
            return
        filtered_paths: list[Path] = []
        for path in request.paths:
            key = self._path_key(path)
            if key in self._queued_path_keys:
                continue
            self._queued_path_keys.add(key)
            filtered_paths.append(path)
        if not filtered_paths:
            return
        filtered_request = build_application_open_request(
            filtered_paths,
            source=request.source,
            request_id=request.request_id,
            require_absolute=True,
        )
        self._queue.append(filtered_request)
        if not self._timer.isActive():
            self._timer.start(0)

    def pending_request_count(self) -> int:
        return len(self._queue)

    def _dispatch_next(self) -> None:
        if not self._queue:
            return
        if self._host._transition_in_progress or self._host.is_image_loading():
            self._timer.start()
            return
        app = QApplication.instance()
        if app is not None and app.activeModalWidget() is not None:
            self._timer.start()
            return
        request = self._queue.popleft()
        for path in request.paths:
            self._queued_path_keys.discard(self._path_key(path))
        try:
            self._dispatch(request)
        finally:
            if self._queue:
                self._timer.start()

    def _dispatch(self, request: ApplicationOpenRequest) -> None:
        project_paths = [path for path in request.paths if path.suffix.lower() == ".fdmproj"]
        if project_paths:
            self._host._load_project_from_path(project_paths[0])
            return
        self._open_digital_slides(list(request.paths))

    def _open_digital_slides(self, paths: list[Path]) -> None:
        valid_paths: list[Path] = []
        failures: list[str] = []
        focus_document_id: str | None = None
        for path in paths:
            existing_id = self._host._associated_open_document_id(path)
            if existing_id is not None:
                focus_document_id = existing_id
                continue
            if not path.is_file():
                failures.append(f"{path.name}: 文件不存在或无法访问")
                continue
            try:
                DigitalSlideStore.read_manifest_read_only(path)
            except Exception as exc:  # noqa: BLE001 - external SQLite is an untrusted input boundary
                failures.append(f"{path.name}: {exc}")
            else:
                valid_paths.append(path)
        if focus_document_id is not None:
            self._host._set_current_document(focus_document_id)
        if failures:
            QMessageBox.warning(
                self._parent_widget(),
                "打开数字化切片",
                "以下数字化切片无法读取，当前工作区未因这些文件而改变：\n\n"
                + "\n".join(failures[:10]),
            )
        if not valid_paths:
            return

        disposition = AssociatedSlideDisposition.ADD_TO_CURRENT
        if self._host._project_path is not None:
            disposition = self._choose_slide_disposition(valid_paths)
        if disposition == AssociatedSlideDisposition.CANCEL:
            return

        documents_to_close = (
            list(getattr(self._host.project, "documents", []))
            if disposition == AssociatedSlideDisposition.STANDALONE_WORKSPACE
            else None
        )
        acquisition_active = self._host._slide_acquisition_active()
        acquisition_disposition = (
            self._host._preflight_acquisition_disposition(TransitionIntent.OPEN_DOCUMENT)
            if acquisition_active
            else None
        )
        if acquisition_disposition == AcquisitionDisposition.CANCEL:
            self._host._show_status_message("操作已取消。", 6000)
            return
        if (
            documents_to_close is not None
            and not self._host._confirm_close_documents(documents_to_close)
        ):
            return

        if acquisition_active:
            transition = self._host._prepare_transition(
                TransitionIntent.OPEN_DOCUMENT,
                disposition=acquisition_disposition,
            )
            if not bool(getattr(transition, "completed", False)):
                reason = str(getattr(transition, "reason", "") or "资源尚未安全退出，已取消打开文件。")
                self._host._show_status_message(reason, 6000)
                return
        else:
            self._host.stop_live_preview()

        if disposition == AssociatedSlideDisposition.STANDALONE_WORKSPACE:
            try:
                self._host._reset_workspace()
            except RuntimeError as exc:
                QMessageBox.information(
                    self._parent_widget(),
                    "打开数字化切片",
                    f"资源尚未安全退出，已取消新建独立工作区。\n\n{exc}",
                )
                return

        self._host._open_image_requests(
            [(str(path), None) for path in valid_paths],
            context_label="关联文件打开",
        )

    def _choose_slide_disposition(self, paths: list[Path]) -> AssociatedSlideDisposition:
        box = QMessageBox(self._parent_widget())
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("打开数字化切片")
        target_text = paths[0].name if len(paths) == 1 else f"选中的 {len(paths)} 个数字化切片"
        box.setText(f"当前已打开一个项目。如何处理{target_text}？")
        add_button = box.addButton("加入当前项目", QMessageBox.ButtonRole.AcceptRole)
        standalone_button = box.addButton("新建独立工作区", QMessageBox.ButtonRole.ActionRole)
        cancel_button = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(add_button)
        box.setEscapeButton(cancel_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked == add_button:
            return AssociatedSlideDisposition.ADD_TO_CURRENT
        if clicked == standalone_button:
            return AssociatedSlideDisposition.STANDALONE_WORKSPACE
        return AssociatedSlideDisposition.CANCEL

    def _parent_widget(self) -> QWidget | None:
        return self._host if isinstance(self._host, QWidget) else None

    @staticmethod
    def _path_key(path: Path) -> str:
        return str(path.resolve(strict=False)).casefold()
