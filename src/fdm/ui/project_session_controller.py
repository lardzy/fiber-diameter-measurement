from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import copy
import hashlib
import math
import re
from typing import Any, Protocol

from fdm import __version__
from fdm.analysis_artifacts import AnalysisArtifact, AnalysisAssetReference
from fdm.atomic_io import atomic_replace_file, staged_path_for
from fdm.models import ImageDocument, ProjectState, project_assets_root
from fdm.lifecycle import TransitionIntent
from fdm.project_io import ProjectIO, resolve_document_load_path
from fdm.services.analysis_asset_io import (
    copy_verified_analysis_asset,
    validate_analysis_asset_reference,
)
from fdm.services.digital_slide_store import copy_slide_file
from fdm.services.raster_io import (
    qimage_to_raster_plane,
    recommended_native_asset_suffix,
    write_native_raster_asset,
)
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
class DocumentPersistenceIdentity:
    """Lightweight identity for an ordered document persisted by the project."""

    document_id: str
    source_type: str
    document_kind: str
    path: str
    absolute_path: str


@dataclass(frozen=True, slots=True)
class UnresolvedPersistenceIdentity:
    """Lightweight unresolved state which affects project persistence."""

    document_id: str
    original_index: int
    attempted_path: str
    reason: str
    original_path_token: str
    original_absolute_path_token: str


@dataclass(frozen=True, slots=True)
class ProjectPersistenceSnapshot:
    """Persistence-sensitive state without measurement geometry or runtime data."""

    documents: tuple[DocumentPersistenceIdentity, ...]
    unresolved: tuple[UnresolvedPersistenceIdentity, ...]


@dataclass(frozen=True, slots=True)
class ProjectDirtySnapshot:
    """Small immutable project state used exclusively for dirty comparisons."""

    project_default_calibration: tuple[str, float, str, str] | None
    project_default_document_ids: tuple[str, ...]
    project_asset_documents: tuple[tuple[str, str], ...]
    project_group_templates: tuple[tuple[str, str], ...]
    project_extension_state_id: int
    document_persistence: ProjectPersistenceSnapshot


@dataclass(frozen=True, slots=True)
class ProjectSaveResult:
    success: bool
    path: Path | None = None
    cancelled: bool = False
    message: str = ""
    unresolved_count: int = 0

    def __bool__(self) -> bool:
        return self.success


@dataclass(frozen=True, slots=True)
class ProjectLoadResult:
    success: bool
    path: Path | None = None
    cancelled: bool = False
    already_open: bool = False
    message: str = ""

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


@dataclass(frozen=True, slots=True)
class PersistentDocumentSnapshot:
    """One ordered document and its runtime-free persistence payload."""

    document_id: str
    source_document: ImageDocument
    payload: dict[str, Any]
    unresolved: bool = False


@dataclass(frozen=True, slots=True)
class DocumentSaveOverride:
    """Asset fields published to JSON and live state only after staging."""

    document_id: str
    path: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AnalysisAssetSaveOverride:
    """One verified analysis-asset reference published after JSON commit."""

    artifact_id: str
    asset_index: int
    reference: AnalysisAssetReference


@dataclass(frozen=True, slots=True)
class AssetWriteOperation:
    """A project asset write derived from an immutable save plan."""

    snapshot: PersistentDocumentSnapshot
    source_document: ImageDocument


@dataclass(frozen=True, slots=True)
class ProjectSavePlan:
    """Runtime-free project payload plus ordered source identities."""

    payload: dict[str, Any]
    documents: tuple[PersistentDocumentSnapshot, ...]
    preserve_path_document_ids: frozenset[str]

    def payload_with_overrides(
        self,
        overrides: tuple[DocumentSaveOverride, ...] | list[DocumentSaveOverride],
        analysis_asset_overrides: (
            tuple[AnalysisAssetSaveOverride, ...]
            | list[AnalysisAssetSaveOverride]
        ) = (),
    ) -> dict[str, Any]:
        override_by_id = {item.document_id: item for item in overrides}
        documents: list[dict[str, Any]] = []
        for snapshot in self.documents:
            document_payload = dict(snapshot.payload)
            override = override_by_id.get(snapshot.document_id)
            if override is not None:
                document_payload["path"] = override.path
                document_payload["metadata"] = copy.deepcopy(override.metadata)
                document_payload.pop("absolute_path", None)
            documents.append(document_payload)
        output = dict(self.payload)
        output["documents"] = documents
        asset_override_by_key = {
            (item.artifact_id, item.asset_index): item.reference
            for item in analysis_asset_overrides
        }
        raw_artifacts = self.payload.get("analysis_artifacts", [])
        if raw_artifacts:
            artifacts: list[dict[str, Any]] = []
            for raw_artifact in raw_artifacts:
                if not isinstance(raw_artifact, dict):
                    raise ValueError("分析结果持久化 payload 无效")
                artifact_payload = copy.deepcopy(raw_artifact)
                artifact_id = str(artifact_payload.get("id", ""))
                raw_assets = artifact_payload.get("assets", [])
                if not isinstance(raw_assets, list):
                    raise ValueError(f"分析结果 {artifact_id} 的资产列表无效")
                assets: list[dict[str, object]] = []
                for asset_index, raw_asset in enumerate(raw_assets):
                    reference = asset_override_by_key.get(
                        (artifact_id, asset_index)
                    )
                    if reference is not None:
                        assets.append(reference.to_dict())
                    elif isinstance(raw_asset, dict):
                        assets.append(copy.deepcopy(raw_asset))
                    else:
                        raise ValueError(
                            f"分析结果 {artifact_id} 的资产引用无效"
                        )
                artifact_payload["assets"] = assets
                artifacts.append(artifact_payload)
            output["analysis_artifacts"] = artifacts
        return output


@dataclass(frozen=True, slots=True)
class ProjectAssetStageResult:
    success: bool
    overrides: tuple[DocumentSaveOverride, ...] = ()
    analysis_asset_overrides: tuple[AnalysisAssetSaveOverride, ...] = ()
    created_paths: tuple[Path, ...] = ()
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
    def _project_asset_raster_for_save(self, document: ImageDocument): ...
    def _analysis_asset_source_for_save(
        self,
        reference: AnalysisAssetReference,
    ) -> Path | None: ...
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

    def _ordered_persisted_document_sources(self) -> tuple[ImageDocument, ...]:
        """Return live/unresolved document references in their persisted order.

        This is deliberately the single ordering implementation shared by dirty
        checks and the save-only deep-copy boundary.  Callers must not mutate the
        returned documents while iterating them.
        """

        live_by_id = {document.id: document for document in self._host.project.documents}
        ordered: list[ImageDocument] = []
        seen: set[str] = set()
        for document_id in self._project_document_order:
            document = live_by_id.get(document_id)
            if document is None:
                unresolved = self._unresolved_documents.get(document_id)
                document = unresolved.document if unresolved is not None else None
            if document is not None and document.id not in seen:
                ordered.append(document)
                seen.add(document.id)
        for document in self._host.project.documents:
            if document.id not in seen:
                ordered.append(document)
                seen.add(document.id)
        return tuple(ordered)

    def persistence_snapshot(self) -> ProjectPersistenceSnapshot:
        """Return ordered document identity and unresolved state used by save."""

        documents = tuple(
            DocumentPersistenceIdentity(
                document_id=document.id,
                source_type=document.source_type,
                document_kind=document.document_kind,
                path=str(document.path),
                absolute_path=str(document.absolute_path or ""),
            )
            for document in self._ordered_persisted_document_sources()
        )
        unresolved = tuple(
            UnresolvedPersistenceIdentity(
                document_id=item.document.id,
                original_index=item.original_index,
                attempted_path=item.attempted_path,
                reason=item.reason,
                original_path_token=item.original_path_token,
                original_absolute_path_token=item.original_absolute_path_token or "",
            )
            for item in self.unresolved_documents()
        )
        return ProjectPersistenceSnapshot(documents=documents, unresolved=unresolved)

    def _begin_project_load(self, documents: list[ImageDocument]) -> None:
        self._unresolved_documents.clear()
        self._project_document_order = [document.id for document in documents]

    def _build_project_save_plan(
        self,
        *,
        project: ProjectState | None = None,
        version: str | None = None,
    ) -> ProjectSavePlan:
        """Snapshot persistence fields without copying ImageDocument runtime state."""

        source_project = project or self._host.project
        ordered_sources = (
            tuple(source_project.documents)
            if project is not None
            else self._ordered_persisted_document_sources()
        )
        payload = ProjectIO.persistent_payload(
            source_project,
            documents=ordered_sources,
            version=version,
        )
        payload_documents = payload.get("documents", [])
        if not isinstance(payload_documents, list) or len(payload_documents) != len(ordered_sources):
            raise ValueError("项目持久化 payload 与文档顺序不一致")

        live_ids = {document.id for document in self._host.project.documents}
        snapshots: list[PersistentDocumentSnapshot] = []
        preserved_ids: set[str] = set()
        for source_document, raw_payload in zip(ordered_sources, payload_documents):
            if not isinstance(raw_payload, dict):
                raise ValueError(f"文档 {source_document.id} 的持久化 payload 无效")
            document_payload = raw_payload
            unresolved = source_document.id not in live_ids and source_document.id in self._unresolved_documents
            if unresolved:
                unresolved_record = self._unresolved_documents[source_document.id]
                document_payload["path"] = unresolved_record.original_path_token
                if unresolved_record.original_absolute_path_token:
                    document_payload["absolute_path"] = unresolved_record.original_absolute_path_token
                else:
                    document_payload.pop("absolute_path", None)
                preserved_ids.add(source_document.id)
            snapshots.append(
                PersistentDocumentSnapshot(
                    document_id=source_document.id,
                    source_document=source_document,
                    payload=document_payload,
                    unresolved=unresolved,
                )
            )
        payload["documents"] = [snapshot.payload for snapshot in snapshots]
        return ProjectSavePlan(
            payload=payload,
            documents=tuple(snapshots),
            preserve_path_document_ids=frozenset(preserved_ids),
        )

    def _project_for_save(self) -> ProjectState:
        """Compatibility view for callers which still expect a ProjectState.

        The normal save path uses :class:`ProjectSavePlan` directly.  This
        compatibility method reconstructs only serialized fields and never
        traverses history, clean snapshots or display caches through deepcopy.
        """

        plan = self._build_project_save_plan()
        project = ProjectState.from_dict(plan.payload_with_overrides(()))
        # Calibration presets are a legacy in-memory compatibility field and
        # are intentionally not part of the project JSON.  The former
        # deepcopy-based compatibility view nevertheless retained them, so do
        # the same without copying any document geometry/runtime state.
        project.calibration_presets = copy.deepcopy(
            self._host.project.calibration_presets
        )
        project.load_issues = copy.deepcopy(self._host.project.load_issues)
        return project

    def save_project(self, path: str | None = None) -> ProjectSaveResult:
        host = self._host
        try:
            save_plan = self._build_project_save_plan(version=__version__)
        except Exception as exc:  # noqa: BLE001 - normalize snapshot failures for the UI
            host._show_project_warning("保存项目", f"无法构造项目保存计划，当前项目未改变：\n{exc}")
            return ProjectSaveResult(False, message=str(exc))
        if not save_plan.documents:
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
        asset_result = self._stage_project_assets(target_path, save_plan)
        if not asset_result:
            return ProjectSaveResult(
                False,
                path=target_path,
                message=asset_result.message or "项目资产写入失败。",
            )
        committed_payload = save_plan.payload_with_overrides(
            asset_result.overrides,
            asset_result.analysis_asset_overrides,
        )
        try:
            # The project JSON is the commit point and is always published last.
            ProjectIO.save_payload(
                committed_payload,
                target_path,
                document_sources=tuple(item.source_document for item in save_plan.documents),
                preserve_path_document_ids=set(save_plan.preserve_path_document_ids),
            )
        except Exception as exc:  # noqa: BLE001 - preserve the previous project on any storage failure
            for created_path in asset_result.created_paths:
                try:
                    created_path.unlink(missing_ok=True)
                except OSError:
                    pass
            host._show_project_warning("保存项目", f"项目文件写入失败，旧项目保持不变：\n{exc}")
            return ProjectSaveResult(False, path=target_path, message=str(exc))

        # Publish save-only overrides to live state strictly after the project
        # JSON commit.  Any asset or JSON failure above leaves live paths,
        # metadata, version and dirty savepoints untouched.
        override_by_id = {item.document_id: item for item in asset_result.overrides}
        for document in host.project.documents:
            override = override_by_id.get(document.id)
            if override is None:
                continue
            document.path = override.path
            document.metadata = copy.deepcopy(override.metadata)
        committed_artifacts = committed_payload.get("analysis_artifacts", [])
        if isinstance(committed_artifacts, list):
            host.project.analysis_artifacts = [
                AnalysisArtifact.from_dict(item)
                for item in committed_artifacts
                if isinstance(item, dict)
            ]
        host.project.version = __version__
        # The JSON replacement above is the save commit point.  Removing old
        # revision assets is deliberately post-commit housekeeping: a denied
        # directory traversal, disappearing mount, or antivirus race must not
        # turn an already committed project into a reported/dirty half-save.
        try:
            self._cleanup_unreferenced_revision_assets_payload(target_path, committed_payload)
        except Exception:  # noqa: BLE001 - post-commit cleanup is strictly best-effort
            pass
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

    def load_project(self) -> ProjectLoadResult:
        selected_path = self._host._select_project_open_path()
        if not selected_path:
            return ProjectLoadResult(False, cancelled=True, message="用户取消打开项目。")
        return self.load_project_from_path(Path(selected_path))

    def load_project_from_path(self, path: str | Path) -> ProjectLoadResult:
        host = self._host
        project_path = Path(path).expanduser().resolve(strict=False)
        if project_path.suffix.lower() != ".fdmproj":
            message = f"所选文件不是 .fdmproj 项目文件：\n{project_path}"
            host._show_project_warning("打开项目", message)
            return ProjectLoadResult(False, path=project_path, message=message)
        if not project_path.is_file():
            message = f"项目文件不存在或无法访问：\n{project_path}"
            host._show_project_warning("打开项目", message)
            return ProjectLoadResult(False, path=project_path, message=message)
        current_project_path = getattr(host, "_project_path", None)
        if current_project_path is not None:
            try:
                same_project = Path(current_project_path).resolve(strict=False) == project_path
            except OSError:
                same_project = False
            if same_project:
                message = f"项目已经打开：{project_path}"
                host._show_status_message(message, 4000)
                return ProjectLoadResult(
                    True,
                    path=project_path,
                    already_open=True,
                    message=message,
                )
        try:
            project = ProjectIO.load(project_path)
        except Exception as exc:  # noqa: BLE001 - external project files are an untrusted input boundary
            message = f"无法读取项目文件，当前工作区未改变：\n{project_path}\n\n{exc}"
            host._show_project_warning("打开项目", message)
            return ProjectLoadResult(False, path=project_path, message=message)

        prepare_transition = getattr(host, "_prepare_transition", None)
        if callable(prepare_transition):
            transition = prepare_transition(TransitionIntent.OPEN_PROJECT)
            if not bool(getattr(transition, "completed", False)):
                reason = str(getattr(transition, "reason", "") or "资源尚未安全退出，已取消打开项目。")
                host._show_project_information("打开项目", reason)
                return ProjectLoadResult(
                    False,
                    path=project_path,
                    cancelled=bool(getattr(transition, "cancelled", False)),
                    message=reason,
                )
        else:
            host.stop_live_preview()
        if host.is_image_loading():
            message = "图片加载任务尚未安全退出，已阻止项目切换。"
            host._show_project_information("打开项目", message)
            return ProjectLoadResult(False, path=project_path, message=message)
        if not host._confirm_close_documents(host.project.documents):
            return ProjectLoadResult(
                False,
                path=project_path,
                cancelled=True,
                message="用户取消切换项目。",
            )
        recalculated_area_count = _recalculate_loaded_area_measurements(project)
        missing_paths: list[str] = []
        try:
            host._reset_workspace()
        except RuntimeError as exc:
            host._show_project_information(
                "打开项目",
                f"资源尚未安全退出，已取消打开项目。\n\n{exc}",
            )
            return ProjectLoadResult(False, path=project_path, message=str(exc))
        imported_count = host._merge_legacy_calibration_presets(project.calibration_presets)
        self._begin_project_load(project.documents)
        host._project_path = project_path
        host.project = ProjectState(
            version=project.version,
            documents=[],
            project_default_calibration=project.project_default_calibration,
            project_group_templates=list(project.project_group_templates),
            project_rois=list(project.project_rois),
            analysis_artifacts=list(project.analysis_artifacts),
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
        return ProjectLoadResult(True, path=project_path, message=message)

    def persist_project_assets(
        self,
        target_path: Path,
        *,
        project: ProjectState | None = None,
    ) -> ProjectAssetPersistResult:
        """Compatibility wrapper around the non-mutating asset staging plan."""

        plan = self._build_project_save_plan(project=project)
        result = self._stage_project_assets(target_path, plan)
        payload = plan.payload_with_overrides(
            result.overrides,
            result.analysis_asset_overrides,
        )
        compatibility_project = ProjectState.from_dict(payload)
        compatibility_project.calibration_presets = copy.deepcopy(
            getattr(project or self._host.project, "calibration_presets", [])
        )
        compatibility_project.load_issues = copy.deepcopy(
            getattr(project or self._host.project, "load_issues", [])
        )
        return ProjectAssetPersistResult(
            result.success,
            compatibility_project,
            list(result.created_paths),
            result.message,
        )

    def _stage_project_assets(
        self,
        target_path: Path,
        plan: ProjectSavePlan,
    ) -> ProjectAssetStageResult:
        host = self._host
        live_by_id = {document.id: document for document in host.project.documents}
        operations = tuple(
            AssetWriteOperation(
                snapshot=snapshot,
                source_document=live_by_id.get(snapshot.document_id, snapshot.source_document),
            )
            for snapshot in plan.documents
            if snapshot.payload.get("source_type") == "project_asset" and not snapshot.unresolved
        )
        created_paths: list[Path] = []
        overrides: list[DocumentSaveOverride] = []
        for operation in operations:
            snapshot = operation.snapshot
            source_document = operation.source_document
            try:
                document_path = _validated_project_asset_path_token(
                    snapshot.payload.get("path", "")
                )
            except ValueError as exc:
                host._show_project_warning(
                    "保存项目",
                    f"项目资产路径无效: "
                    f"{host._document_display_name(source_document)}\n{exc}",
                )
                return _failed_asset_stage_result(
                    created_paths,
                    str(exc),
                )
            document_kind = str(snapshot.payload.get("document_kind", "image"))
            document_metadata = (
                copy.deepcopy(snapshot.payload.get("metadata", {}))
                if isinstance(snapshot.payload.get("metadata"), dict)
                else {}
            )
            if document_kind == "digital_slide":
                source_meta = source_document.metadata.get("digital_slide", {})
                source_token = (
                    str(source_meta.get("working_path", "")).strip()
                    if isinstance(source_meta, dict)
                    else ""
                )
                original_target = project_assets_root(target_path) / document_path
                source_path = Path(source_token).expanduser() if source_token else original_target
                if not source_path.exists():
                    host._show_project_warning(
                        "保存项目",
                        f"无法找到项目内数字化切片数据: {host._document_display_name(source_document)}",
                    )
                    return _failed_asset_stage_result(
                        created_paths,
                        "数字化切片源文件不存在。",
                    )
                slide_target_path = original_target
                try:
                    with staged_path_for(original_target, suffix=".fdmslide") as staged_path:
                        copy_slide_file(source_path, staged_path)
                        digest = _file_sha256(staged_path)
                        revised_relative = _revisioned_asset_path(document_path, digest)
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
                    return _failed_asset_stage_result(created_paths, str(exc))
                output_meta = document_metadata
                output_slide_meta = (
                    dict(output_meta.get("digital_slide", {}))
                    if isinstance(output_meta.get("digital_slide"), dict)
                    else {}
                )
                output_slide_meta["working_path"] = str(slide_target_path)
                output_meta["digital_slide"] = output_slide_meta
                overrides.append(
                    DocumentSaveOverride(
                        document_id=snapshot.document_id,
                        path=revised_relative,
                        metadata=output_meta,
                    )
                )
                continue

            raster_provider = getattr(
                host,
                "_project_asset_raster_for_save",
                None,
            )
            image = host._project_asset_image_for_save(source_document)
            try:
                raster_asset = (
                    raster_provider(source_document)
                    if callable(raster_provider)
                    else None
                )
            except Exception as exc:  # noqa: BLE001 - normalize provider failures
                host._show_project_warning(
                    "保存项目",
                    f"读取项目内图片像素失败: "
                    f"{host._document_display_name(source_document)}\n{exc}",
                )
                return _failed_asset_stage_result(created_paths, str(exc))
            if raster_asset is None and (image is None or image.isNull()):
                host._show_project_warning(
                    "保存项目",
                    f"无法找到项目内图片数据: {host._document_display_name(source_document)}",
                )
                return _failed_asset_stage_result(
                    created_paths,
                    "项目内图片数据不可用。",
                )
            if raster_asset is None:
                plane = qimage_to_raster_plane(image)
                raster_metadata = None
            else:
                plane, raster_metadata = raster_asset
            expected_pixel_type = source_document.raster_pixel_type
            if (
                expected_pixel_type is not None
                and plane.pixel_type is not expected_pixel_type
            ):
                reason = (
                    "项目文档声明的像素类型与待保存像素不一致："
                    f"期望 {expected_pixel_type.value}，实际 "
                    f"{plane.pixel_type.value}。"
                )
                host._show_project_warning("保存项目", reason)
                return _failed_asset_stage_result(created_paths, reason)
            expected_size = tuple(source_document.image_size)
            actual_size = (plane.width, plane.height)
            if expected_size != actual_size:
                reason = (
                    "项目文档声明的图片尺寸与待保存像素不一致："
                    f"期望 {expected_size[0]}×{expected_size[1]}，实际 "
                    f"{actual_size[0]}×{actual_size[1]}。"
                )
                host._show_project_warning("保存项目", reason)
                return _failed_asset_stage_result(created_paths, reason)
            required_suffix = recommended_native_asset_suffix(
                plane.pixel_type
            )
            current_suffix = Path(document_path).suffix.casefold()
            if plane.pixel_type.value == "gray32_float":
                allowed_suffixes = {".tif", ".tiff"}
            elif plane.pixel_type.value == "gray16":
                allowed_suffixes = {".png", ".tif", ".tiff"}
            else:
                allowed_suffixes = {".png"}
            if current_suffix not in allowed_suffixes:
                document_path = str(
                    Path(document_path).with_suffix(required_suffix)
                ).replace("\\", "/")
            original_target = project_assets_root(target_path) / document_path
            output_path = original_target
            original_target.parent.mkdir(parents=True, exist_ok=True)
            try:
                staged_suffix = (
                    original_target.suffix
                    if original_target.suffix.casefold()
                    in {".png", ".tif", ".tiff"}
                    else ".png"
                )
                with staged_path_for(
                    original_target,
                    suffix=staged_suffix,
                ) as staged_path:
                    encoded = write_native_raster_asset(
                        plane,
                        staged_path,
                        metadata=raster_metadata,
                    )
                    if not encoded:
                        failure = encoded.failure
                        detail = (
                            f"{failure.message}: {failure.detail}"
                            if failure is not None and failure.detail
                            else (
                                failure.message
                                if failure is not None
                                else "未知编码错误"
                            )
                        )
                        raise OSError(detail)
                    digest = _file_sha256(staged_path)
                    revised_relative = _revisioned_asset_path(document_path, digest)
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
            except Exception as exc:  # noqa: BLE001 - storage failures share one UI contract
                host._show_project_warning("保存项目", f"写入项目内图片失败: {output_path}\n{exc}")
                return _failed_asset_stage_result(created_paths, str(exc))
            overrides.append(
                DocumentSaveOverride(
                    document_id=snapshot.document_id,
                    path=revised_relative,
                    metadata=document_metadata,
                )
            )
        try:
            analysis_asset_overrides = self._stage_analysis_assets(
                target_path,
                plan,
                created_paths=created_paths,
            )
        except Exception as exc:  # noqa: BLE001 - one storage contract for all assets
            host._show_project_warning(
                "保存项目",
                f"写入分析结果资产失败，旧项目保持不变：\n{exc}",
            )
            return _failed_asset_stage_result(created_paths, str(exc))
        return ProjectAssetStageResult(
            True,
            overrides=tuple(overrides),
            analysis_asset_overrides=analysis_asset_overrides,
            created_paths=tuple(created_paths),
        )

    def _stage_analysis_assets(
        self,
        target_path: Path,
        plan: ProjectSavePlan,
        *,
        created_paths: list[Path],
    ) -> tuple[AnalysisAssetSaveOverride, ...]:
        """Verify and stage every external array referenced by analysis data."""

        raw_artifacts = plan.payload.get("analysis_artifacts", [])
        if not raw_artifacts:
            return ()
        if not isinstance(raw_artifacts, list):
            raise ValueError("项目中的分析结果列表无效")
        host = self._host
        provider = getattr(host, "_analysis_asset_source_for_save", None)
        old_project_path = getattr(host, "_project_path", None)
        target_root = project_assets_root(target_path)
        overrides: list[AnalysisAssetSaveOverride] = []
        for raw_artifact in raw_artifacts:
            if not isinstance(raw_artifact, dict):
                raise ValueError("项目中的分析结果记录无效")
            artifact = AnalysisArtifact.from_dict(raw_artifact)
            for asset_index, reference in enumerate(artifact.assets):
                source_path: Path | None = None
                if callable(provider):
                    provided = provider(reference)
                    if provided is not None:
                        source_path = Path(provided)
                if source_path is None and old_project_path is not None:
                    candidate = (
                        project_assets_root(old_project_path) / reference.path
                    )
                    if candidate.is_file():
                        source_path = candidate
                if source_path is None:
                    candidate = target_root / reference.path
                    if candidate.is_file():
                        source_path = candidate
                if source_path is None or not source_path.is_file():
                    raise FileNotFoundError(
                        f"分析结果 {artifact.id} 的资产不存在："
                        f"{reference.path}"
                    )
                validate_analysis_asset_reference(source_path, reference)

                original_token = _validated_project_asset_path_token(
                    reference.path
                )
                original_parts = PurePosixPath(original_token).parts
                if not original_parts or original_parts[0].casefold() != "analysis":
                    original_token = (
                        PurePosixPath("analysis")
                        / artifact.id
                        / PurePosixPath(original_token).name
                    ).as_posix()
                revised_relative = _revisioned_asset_path(
                    original_token,
                    reference.sha256,
                )
                output_path = target_root / revised_relative
                if output_path.exists():
                    validate_analysis_asset_reference(output_path, reference)
                else:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        copy_verified_analysis_asset(
                            source_path,
                            output_path,
                            reference,
                        )
                    except Exception:
                        try:
                            output_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                        raise
                    created_paths.append(output_path)
                overrides.append(
                    AnalysisAssetSaveOverride(
                        artifact_id=artifact.id,
                        asset_index=asset_index,
                        reference=AnalysisAssetReference(
                            kind=reference.kind,
                            path=revised_relative,
                            sha256=reference.sha256,
                            media_type=reference.media_type,
                            metadata=reference.metadata,
                        ),
                    )
                )
        return tuple(overrides)

    def _cleanup_unreferenced_revision_assets(self, target_path: Path, project: ProjectState) -> None:
        self._cleanup_unreferenced_revision_assets_payload(target_path, project.to_dict())

    def _cleanup_unreferenced_revision_assets_payload(
        self,
        target_path: Path,
        payload: dict[str, Any],
    ) -> None:
        asset_root = project_assets_root(target_path)
        if not asset_root.exists():
            return
        referenced = {
            (asset_root / str(document.get("path", ""))).resolve()
            for document in payload.get("documents", [])
            if isinstance(document, dict) and document.get("source_type") == "project_asset"
        }
        for artifact in payload.get("analysis_artifacts", []):
            if not isinstance(artifact, dict):
                continue
            for asset in artifact.get("assets", []):
                if isinstance(asset, dict) and asset.get("path"):
                    referenced.add(
                        (asset_root / str(asset["path"])).resolve()
                    )
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


def _failed_asset_stage_result(
    created_paths: list[Path],
    message: str,
) -> ProjectAssetStageResult:
    for created_path in created_paths:
        try:
            created_path.unlink(missing_ok=True)
        except OSError:
            pass
    return ProjectAssetStageResult(False, message=message)


def _validated_project_asset_path_token(value: object) -> str:
    token = str(value or "").replace("\\", "/").strip()
    path = PurePosixPath(token)
    if (
        not token
        or path.is_absolute()
        or ".." in path.parts
        or any(part.endswith(":") for part in path.parts)
    ):
        raise ValueError("项目资产路径必须是资产目录内的安全相对路径。")
    normalized = path.as_posix()
    if normalized in {".", ""}:
        raise ValueError("项目资产路径不能为空。")
    return normalized


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
