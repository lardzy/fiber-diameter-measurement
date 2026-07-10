from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib
import math
import re
from typing import Protocol

from fdm import __version__
from fdm.atomic_io import atomic_replace_file, staged_path_for
from fdm.models import ImageDocument, ProjectState, project_assets_root
from fdm.lifecycle import TransitionIntent
from fdm.project_io import ProjectIO, resolve_document_load_path
from fdm.services.digital_slide_store import copy_slide_file
from fdm.settings import AppSettings


@dataclass(slots=True)
class UnresolvedProjectDocument:
    document: ImageDocument
    original_index: int
    attempted_path: str
    reason: str
    original_path_token: str
    original_absolute_path_token: str | None


@dataclass(frozen=True, slots=True)
class ProjectSaveResult:
    success: bool
    path: Path | None = None
    cancelled: bool = False
    message: str = ""
    unresolved_count: int = 0

    def __bool__(self) -> bool:
        return self.success


@dataclass(slots=True)
class ProjectAssetPersistResult:
    success: bool
    project: ProjectState
    created_paths: list[Path]
    message: str = ""

    def __bool__(self) -> bool:
        return self.success


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
        self._unresolved_documents: dict[str, UnresolvedProjectDocument] = {}
        self._project_document_order: list[str] = []

    def clear_unresolved_documents(self) -> None:
        self._unresolved_documents.clear()
        self._project_document_order.clear()

    def unresolved_documents(self) -> list[UnresolvedProjectDocument]:
        return sorted(self._unresolved_documents.values(), key=lambda item: item.original_index)

    def unresolved_document(self, document_id: str) -> UnresolvedProjectDocument | None:
        return self._unresolved_documents.get(document_id)

    def ui_insert_index(self, document_id: str, mounted_order: list[str]) -> int:
        if document_id not in self._project_document_order:
            return len(mounted_order)
        target_index = self._project_document_order.index(document_id)
        preceding = set(self._project_document_order[:target_index])
        return sum(1 for item in mounted_order if item in preceding)

    def register_unresolved_document(
        self,
        document: ImageDocument,
        *,
        attempted_path: str,
        reason: str,
        original_index: int | None = None,
    ) -> None:
        if document.id not in self._project_document_order:
            self._project_document_order.append(document.id)
        if original_index is None:
            original_index = self._project_document_order.index(document.id)
        existing = self._unresolved_documents.get(document.id)
        self._unresolved_documents[document.id] = UnresolvedProjectDocument(
            document=document,
            original_index=int(original_index),
            attempted_path=str(attempted_path),
            reason=str(reason),
            original_path_token=(existing.original_path_token if existing is not None else str(document.path)),
            original_absolute_path_token=(
                existing.original_absolute_path_token
                if existing is not None
                else (str(document.absolute_path) if document.absolute_path else None)
            ),
        )

    def mark_document_resolved(self, document_id: str) -> None:
        self._unresolved_documents.pop(document_id, None)

    def remove_document(self, document_id: str) -> None:
        self._unresolved_documents.pop(document_id, None)
        self._project_document_order = [item for item in self._project_document_order if item != document_id]

    def persistence_snapshot(self) -> tuple[object, ...]:
        """Return ordered document identity and unresolved state used by save."""

        documents = tuple(
            (
                document.id,
                document.source_type,
                document.document_kind,
                str(document.path),
                str(document.absolute_path or ""),
            )
            for document in self._project_for_save().documents
        )
        unresolved = tuple(
            (
                item.document.id,
                item.original_index,
                item.attempted_path,
                item.reason,
                item.original_path_token,
                item.original_absolute_path_token or "",
            )
            for item in self.unresolved_documents()
        )
        return documents, unresolved

    def _begin_project_load(self, documents: list[ImageDocument]) -> None:
        self._unresolved_documents.clear()
        self._project_document_order = [document.id for document in documents]

    def _project_for_save(self) -> ProjectState:
        host = self._host
        live_by_id = {document.id: document for document in host.project.documents}
        ordered: list[ImageDocument] = []
        seen: set[str] = set()
        for document_id in self._project_document_order:
            document = live_by_id.get(document_id)
            if document is None:
                unresolved = self._unresolved_documents.get(document_id)
                document = unresolved.document if unresolved is not None else None
            if document is not None and document.id not in seen:
                ordered.append(copy.deepcopy(document))
                seen.add(document.id)
        for document in host.project.documents:
            if document.id not in seen:
                ordered.append(copy.deepcopy(document))
                seen.add(document.id)
        return ProjectState(
            version=str(getattr(host.project, "version", __version__)),
            documents=ordered,
            calibration_presets=list(getattr(host.project, "calibration_presets", [])),
            project_default_calibration=getattr(host.project, "project_default_calibration", None),
            project_group_templates=list(getattr(host.project, "project_group_templates", [])),
            metadata=dict(getattr(host.project, "metadata", {})),
            load_issues=list(getattr(host.project, "load_issues", [])),
        )

    def save_project(self, path: str | None = None) -> ProjectSaveResult:
        host = self._host
        project_to_save = self._project_for_save()
        if not project_to_save.documents:
            host._show_project_information("保存项目", "请先打开图片。")
            return ProjectSaveResult(False, message="当前项目没有文档。")
        target_path = Path(path) if path else host._project_path
        if target_path is None:
            default_dir = host._preferred_dialog_directory(
                recent_dir=host._app_settings.recent_project_dir,
            )
            selected_path = host._select_project_save_path(default_dir / "fiber_measurement.fdmproj")
            if not selected_path:
                return ProjectSaveResult(False, cancelled=True, message="用户取消保存。")
            target_path = host._normalize_dialog_save_path(selected_path, "fiber_measurement.fdmproj")
        host.project.version = __version__
        project_to_save.version = __version__
        asset_result = self.persist_project_assets(target_path, project=project_to_save)
        if not asset_result:
            return ProjectSaveResult(
                False,
                path=target_path,
                message=asset_result.message or "项目资产写入失败。",
            )
        try:
            # The project JSON is the commit point and is always published last.
            ProjectIO.save(
                project_to_save,
                target_path,
                preserve_path_document_ids=set(self._unresolved_documents),
            )
        except Exception as exc:  # noqa: BLE001 - preserve the previous project on any storage failure
            for created_path in asset_result.created_paths:
                try:
                    created_path.unlink(missing_ok=True)
                except OSError:
                    pass
            host._show_project_warning("保存项目", f"项目文件写入失败，旧项目保持不变：\n{exc}")
            return ProjectSaveResult(False, path=target_path, message=str(exc))
        saved_by_id = {document.id: document for document in project_to_save.documents}
        for document in host.project.documents:
            saved_document = saved_by_id.get(document.id)
            if saved_document is None or not document.is_project_asset():
                continue
            document.path = saved_document.path
            document.metadata = copy.deepcopy(saved_document.metadata)
        self._cleanup_unreferenced_revision_assets(target_path, project_to_save)
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
        return ProjectSaveResult(
            True,
            path=target_path,
            message="项目已保存。",
            unresolved_count=len(self._unresolved_documents),
        )

    def load_project(self) -> None:
        selected_path = self._host._select_project_open_path()
        if not selected_path:
            return
        self.load_project_from_path(Path(selected_path))

    def load_project_from_path(self, path: str | Path) -> None:
        host = self._host
        prepare_transition = getattr(host, "_prepare_transition", None)
        if callable(prepare_transition):
            transition = prepare_transition(TransitionIntent.OPEN_PROJECT)
            if not bool(getattr(transition, "completed", False)):
                reason = str(getattr(transition, "reason", "") or "资源尚未安全退出，已取消打开项目。")
                host._show_project_information("打开项目", reason)
                return
        else:
            host.stop_live_preview()
        if host.is_image_loading():
            host._show_project_information("打开项目", "图片加载任务尚未安全退出，已阻止项目切换。")
            return
        if not host._confirm_close_documents(host.project.documents):
            return
        project_path = Path(path).expanduser().resolve()
        project = ProjectIO.load(project_path)
        recalculated_area_count = _recalculate_loaded_area_measurements(project)
        missing_paths: list[str] = []
        try:
            host._reset_workspace()
        except RuntimeError as exc:
            host._show_project_information(
                "打开项目",
                f"资源尚未安全退出，已取消打开项目。\n\n{exc}",
            )
            return
        imported_count = host._merge_legacy_calibration_presets(project.calibration_presets)
        self._begin_project_load(project.documents)
        host._project_path = project_path
        host.project = ProjectState(
            version=project.version,
            documents=[],
            project_default_calibration=project.project_default_calibration,
            project_group_templates=list(project.project_group_templates),
            load_issues=list(project.load_issues),
        )
        host.project.metadata = project.metadata
        host._refresh_preset_combo()
        load_items: list[tuple[str, ImageDocument | None]] = []
        repaired_paths: list[str] = []
        repaired_path_count = 0
        for original_index, document in enumerate(project.documents):
            resolution = resolve_document_load_path(document, host._project_path)
            if resolution is not None:
                resolved_path = resolution.path
                if document.source_type == "filesystem":
                    original_absolute_path = str(document.absolute_path or document.path or "").strip()
                    if resolution.repaired_from_missing_absolute:
                        repaired_path_count += 1
                        repaired_paths.append(f"{original_absolute_path} -> {resolved_path}")
                load_items.append((str(resolved_path), document))
            else:
                attempted_path = str(document.resolved_path(host._project_path))
                missing_paths.append(attempted_path)
                self.register_unresolved_document(
                    document,
                    attempted_path=attempted_path,
                    reason="源文件不存在",
                    original_index=original_index,
                )
                add_placeholder = getattr(host, "_ensure_unresolved_project_placeholder", None)
                if callable(add_placeholder):
                    add_placeholder(document, attempted_path, "源文件不存在")
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
        if project.load_issues:
            message += f"；隔离了 {len(project.load_issues)} 条无效标尺数据，请重新标定"
        if recalculated_area_count:
            message += f"；按可见填充规则重算了 {recalculated_area_count} 条面积"
        if self._unresolved_documents:
            message += f"；已保留 {len(self._unresolved_documents)} 个缺失文档记录"
        host._show_status_message(message, 5000)

    def persist_project_assets(
        self,
        target_path: Path,
        *,
        project: ProjectState | None = None,
    ) -> ProjectAssetPersistResult:
        host = self._host
        project_to_persist = project or self._project_for_save()
        source_by_id = {document.id: document for document in host.project.documents}
        source_by_id.update(
            {
                document_id: unresolved.document
                for document_id, unresolved in self._unresolved_documents.items()
            }
        )
        created_paths: list[Path] = []
        for document in project_to_persist.documents:
            if not document.is_project_asset() or document.id in self._unresolved_documents:
                continue
            source_document = source_by_id.get(document.id, document)
            if document.is_digital_slide():
                source_meta = source_document.metadata.get("digital_slide", {})
                source_token = (
                    str(source_meta.get("working_path", "")).strip()
                    if isinstance(source_meta, dict)
                    else ""
                )
                original_target = project_assets_root(target_path) / document.path
                source_path = Path(source_token).expanduser() if source_token else original_target
                if not source_path.exists():
                    host._show_project_warning(
                        "保存项目",
                        f"无法找到项目内数字化切片数据: {host._document_display_name(source_document)}",
                    )
                    return _failed_asset_result(
                        project_to_persist,
                        created_paths,
                        "数字化切片源文件不存在。",
                    )
                slide_target_path = original_target
                try:
                    with staged_path_for(original_target, suffix=".fdmslide") as staged_path:
                        copy_slide_file(source_path, staged_path)
                        digest = _file_sha256(staged_path)
                        revised_relative = _revisioned_asset_path(document.path, digest)
                        slide_target_path = project_assets_root(target_path) / revised_relative
                        if slide_target_path.exists():
                            if _file_sha256(slide_target_path) != digest:
                                raise OSError(
                                    f"修订资产哈希冲突或既有文件已损坏: {slide_target_path}"
                                )
                            staged_path.unlink(missing_ok=True)
                        else:
                            slide_target_path.parent.mkdir(parents=True, exist_ok=True)
                            atomic_replace_file(staged_path, slide_target_path)
                            created_paths.append(slide_target_path)
                except Exception as exc:  # noqa: BLE001 - normalize backend failures for the UI
                    host._show_project_warning("保存项目", f"写入数字化切片失败: {slide_target_path}\n{exc}")
                    return _failed_asset_result(project_to_persist, created_paths, str(exc))
                document.path = revised_relative
                output_meta = copy.deepcopy(document.metadata)
                output_slide_meta = (
                    dict(output_meta.get("digital_slide", {}))
                    if isinstance(output_meta.get("digital_slide"), dict)
                    else {}
                )
                output_slide_meta["working_path"] = str(slide_target_path)
                output_meta["digital_slide"] = output_slide_meta
                document.metadata = output_meta
                continue

            image = host._project_asset_image_for_save(source_document)
            if image is None or image.isNull():
                host._show_project_warning(
                    "保存项目",
                    f"无法找到项目内图片数据: {host._document_display_name(source_document)}",
                )
                return _failed_asset_result(
                    project_to_persist,
                    created_paths,
                    "项目内图片数据不可用。",
                )
            original_target = project_assets_root(target_path) / document.path
            output_path = original_target
            original_target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with staged_path_for(original_target, suffix=".png") as staged_path:
                    if not image.save(str(staged_path), "PNG") or staged_path.stat().st_size <= 0:
                        raise OSError("QImage 未生成有效 PNG")
                    digest = _file_sha256(staged_path)
                    revised_relative = _revisioned_asset_path(document.path, digest)
                    output_path = project_assets_root(target_path) / revised_relative
                    if output_path.exists():
                        if _file_sha256(output_path) != digest:
                            raise OSError(
                                f"修订资产哈希冲突或既有文件已损坏: {output_path}"
                            )
                        staged_path.unlink(missing_ok=True)
                    else:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_replace_file(staged_path, output_path)
                        created_paths.append(output_path)
                    document.path = revised_relative
            except Exception as exc:  # noqa: BLE001 - storage failures share one UI contract
                host._show_project_warning("保存项目", f"写入项目内图片失败: {output_path}\n{exc}")
                return _failed_asset_result(project_to_persist, created_paths, str(exc))
        return ProjectAssetPersistResult(True, project_to_persist, created_paths)

    def _cleanup_unreferenced_revision_assets(self, target_path: Path, project: ProjectState) -> None:
        asset_root = project_assets_root(target_path)
        if not asset_root.exists():
            return
        referenced = {
            (asset_root / document.path).resolve()
            for document in project.documents
            if document.is_project_asset()
        }
        for candidate in asset_root.rglob("*"):
            if not candidate.is_file() or candidate.resolve() in referenced:
                continue
            if not re.search(
                r"\.rev-[0-9a-f]{8,64}(?:\.[^.]+)?$",
                candidate.name,
                flags=re.IGNORECASE,
            ):
                continue
            try:
                candidate.unlink()
            except OSError:
                continue

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failed_asset_result(
    project: ProjectState,
    created_paths: list[Path],
    message: str,
) -> ProjectAssetPersistResult:
    for created_path in created_paths:
        try:
            created_path.unlink(missing_ok=True)
        except OSError:
            pass
    return ProjectAssetPersistResult(False, project, [], message)


def _revisioned_asset_path(path_token: str, digest: str) -> str:
    path = Path(path_token)
    stem = re.sub(r"\.rev-[0-9a-f]{8,64}$", "", path.stem, flags=re.IGNORECASE)
    return (path.parent / f"{stem}.rev-{digest[:12]}{path.suffix}").as_posix()


def _recalculate_loaded_area_measurements(project: ProjectState) -> int:
    changed = 0
    for document in project.documents:
        for measurement in document.measurements:
            if measurement.measurement_kind != "area" or measurement.exact_area_px is not None:
                continue
            previous = measurement.area_px
            measurement.recalculate(document.calibration)
            if previous is None or not math.isclose(
                float(previous),
                float(measurement.area_px or 0.0),
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                changed += 1
    return changed
