from __future__ import annotations

from pathlib import Path
from typing import Protocol

from fdm import __version__
from fdm.models import ImageDocument, ProjectState, project_assets_root
from fdm.project_io import ProjectIO, resolve_document_load_path
from fdm.settings import AppSettings


class ProjectSessionHost(Protocol):
    project: ProjectState
    _project_path: Path | None
    _pending_project_load_snapshot: bool
    _app_settings: AppSettings

    def _show_project_information(self, title: str, message: str) -> None: ...
    def _show_project_warning(self, title: str, message: str) -> None: ...
    def _select_project_save_path(self, default_path: Path) -> str: ...
    def _select_project_open_path(self) -> str: ...
    def _preferred_dialog_directory(self, *, recent_dir: str = "") -> Path: ...
    def _normalize_dialog_save_path(self, selected_path: str, default_filename: str) -> Path: ...
    def _remember_recent_directory(self, *, setting_name: str, directory: Path, context: str) -> None: ...
    def _document_display_name(self, document: ImageDocument) -> str: ...
    def _project_asset_image_for_save(self, document: ImageDocument): ...
    def _confirm_close_documents(self, documents: list[ImageDocument]) -> bool: ...
    def _merge_legacy_calibration_presets(self, presets: list[object]) -> int: ...
    def _reset_workspace(self) -> None: ...
    def _refresh_preset_combo(self, *, selected_name: str | None = None) -> None: ...
    def _open_image_requests(
        self,
        requests: list[tuple[str, ImageDocument | None]],
        *,
        context_label: str,
        missing_paths: list[str] | None = None,
        repaired_paths: list[str] | None = None,
    ) -> None: ...
    def _mark_project_saved(self) -> None: ...
    def _update_ui_for_current_document(self) -> None: ...
    def _show_status_message(self, message: str, timeout_ms: int = 0) -> None: ...
    def is_image_loading(self) -> bool: ...
    def stop_live_preview(self) -> None: ...


class ProjectSessionController:
    def __init__(self, host: ProjectSessionHost) -> None:
        self._host = host

    def save_project(self, path: str | None = None) -> bool:
        host = self._host
        if not host.project.documents:
            host._show_project_information("保存项目", "请先打开图片。")
            return False
        target_path = Path(path) if path else host._project_path
        if target_path is None:
            default_dir = host._preferred_dialog_directory(
                recent_dir=host._app_settings.recent_project_dir,
            )
            selected_path = host._select_project_save_path(default_dir / "fiber_measurement.fdmproj")
            if not selected_path:
                return False
            target_path = host._normalize_dialog_save_path(selected_path, "fiber_measurement.fdmproj")
        host.project.version = __version__
        if not self.persist_project_assets(target_path):
            return False
        ProjectIO.save(host.project, target_path)
        host._project_path = target_path
        host._remember_recent_directory(
            setting_name="recent_project_dir",
            directory=target_path.parent,
            context="保存项目",
        )
        for document in host.project.documents:
            document.mark_session_saved()
            document.mark_calibration_saved()
        host._mark_project_saved()
        host._update_ui_for_current_document()
        host._show_status_message(f"项目已保存: {target_path}", 5000)
        return True

    def load_project(self) -> None:
        selected_path = self._host._select_project_open_path()
        if not selected_path:
            return
        self.load_project_from_path(Path(selected_path))

    def load_project_from_path(self, path: str | Path) -> None:
        host = self._host
        host.stop_live_preview()
        if host.is_image_loading():
            host._show_project_information("打开项目", "当前仍有图片在加载，请稍候。")
            return
        if not host._confirm_close_documents(host.project.documents):
            return
        project_path = Path(path).expanduser().resolve()
        project = ProjectIO.load(project_path)
        imported_count = host._merge_legacy_calibration_presets(project.calibration_presets)
        missing_paths: list[str] = []
        host._reset_workspace()
        host._project_path = project_path
        host.project = ProjectState(
            version=project.version,
            documents=[],
            project_default_calibration=project.project_default_calibration,
            project_group_templates=list(project.project_group_templates),
        )
        host.project.metadata = project.metadata
        host._refresh_preset_combo()
        load_items: list[tuple[str, ImageDocument | None]] = []
        repaired_paths: list[str] = []
        repaired_path_count = 0
        for document in project.documents:
            resolution = resolve_document_load_path(document, host._project_path)
            if resolution is not None:
                resolved_path = resolution.path
                if document.source_type == "filesystem":
                    original_absolute_path = str(document.absolute_path or document.path or "").strip()
                    document.path = str(resolved_path)
                    document.absolute_path = str(resolved_path)
                    if resolution.repaired_from_missing_absolute:
                        repaired_path_count += 1
                        repaired_paths.append(f"{original_absolute_path} -> {resolved_path}")
                load_items.append((str(resolved_path), document))
            else:
                missing_paths.append(str(document.resolved_path(host._project_path)))
        host._pending_project_load_snapshot = True
        host._open_image_requests(
            load_items,
            context_label="打开项目",
            missing_paths=missing_paths,
            repaired_paths=repaired_paths,
        )
        if not host.is_image_loading():
            host._mark_project_saved()
            host._pending_project_load_snapshot = False
        message = f"项目已加载: {project_path}"
        if imported_count:
            message += f"；已导入 {imported_count} 个旧版标定预设"
        if repaired_path_count:
            message += f"；已自动修复 {repaired_path_count} 张图片路径"
        host._show_status_message(message, 5000)

    def persist_project_assets(self, target_path: Path) -> bool:
        host = self._host
        for document in host.project.documents:
            if not document.is_project_asset():
                continue
            image = host._project_asset_image_for_save(document)
            if image is None or image.isNull():
                host._show_project_warning(
                    "保存项目",
                    f"无法找到项目内图片数据: {host._document_display_name(document)}",
                )
                return False
            output_path = project_assets_root(target_path) / document.path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not image.save(str(output_path), "PNG"):
                host._show_project_warning("保存项目", f"写入项目内图片失败: {output_path}")
                return False
        return True
