from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path, PureWindowsPath
import copy
import json
import math

from fdm.atomic_io import atomic_copy_file, atomic_write_json
from fdm.models import (
    PROJECT_SCHEMA_VERSION,
    Calibration,
    CalibrationPreset,
    ImageDocument,
    ProjectCompatibilityState,
    ProjectState,
)


_NONFINITE_RAW_VALUE_KEY = "__fdm_nonfinite_float_v1__"


def _encode_raw_payload_value(value: object) -> object:
    """Encode legacy non-finite numbers without emitting invalid JSON tokens."""

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "+inf"
        else:
            label = "-inf"
        return {_NONFINITE_RAW_VALUE_KEY: label}
    if isinstance(value, dict):
        return {str(key): _encode_raw_payload_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_raw_payload_value(item) for item in value]
    return value


def _decode_raw_payload_value(value: object) -> object:
    if isinstance(value, dict):
        if set(value) == {_NONFINITE_RAW_VALUE_KEY}:
            label = value.get(_NONFINITE_RAW_VALUE_KEY)
            if label == "nan":
                return float("nan")
            if label == "+inf":
                return float("inf")
            if label == "-inf":
                return float("-inf")
        return {str(key): _decode_raw_payload_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_raw_payload_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class DocumentPathResolution:
    path: Path
    source: str
    repaired_from_missing_absolute: bool = False


class ProjectCompatibilityError(ValueError):
    """Raised when a compatibility decision forbids an in-place save."""


@dataclass(frozen=True, slots=True)
class ProjectUpgradeBackupResult:
    source_path: Path
    backup_path: Path | None
    source_schema_version: int
    created: bool

    @property
    def required(self) -> bool:
        return self.source_schema_version < PROJECT_SCHEMA_VERSION


def _save_filesystem_token(token: str, project_dir: Path) -> Path:
    image_path = Path(token).expanduser()
    if image_path.is_absolute():
        return image_path
    return project_dir / image_path


def _project_relative_path(path: Path, project_dir: Path) -> str | None:
    try:
        return path.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return None


def _is_foreign_absolute_path_token(token: str) -> bool:
    return PureWindowsPath(token).is_absolute() and not Path(token).expanduser().is_absolute()


def _path_token_filename(token: str) -> str:
    if "\\" in token or PureWindowsPath(token).is_absolute():
        return PureWindowsPath(token).name
    return Path(token).expanduser().name


def _relative_path_candidate(token: str, project_dir: Path) -> Path | None:
    if _is_foreign_absolute_path_token(token):
        return None
    if "\\" in token:
        windows_path = PureWindowsPath(token)
        if windows_path.drive or windows_path.root:
            return None
        return project_dir.joinpath(*windows_path.parts).resolve()
    image_path = Path(token).expanduser()
    if image_path.is_absolute():
        return None
    return (project_dir / image_path).resolve()


def _apply_document_save_path(payload: dict, document: ImageDocument, project_dir: Path) -> None:
    if document.source_type != "filesystem":
        payload.pop("absolute_path", None)
        return
    token = str(document.path or "").strip()
    if not token:
        payload.pop("absolute_path", None)
        return
    absolute_path = _save_filesystem_token(token, project_dir)
    relative_path = _project_relative_path(absolute_path, project_dir)
    if relative_path is None:
        payload["path"] = str(absolute_path)
        payload.pop("absolute_path", None)
        return
    payload["path"] = relative_path
    payload["absolute_path"] = str(absolute_path)


def resolve_document_load_path(
    document: ImageDocument,
    project_path: str | Path,
) -> DocumentPathResolution | None:
    project_file = Path(project_path).expanduser().resolve()
    project_dir = project_file.parent
    if document.is_project_asset():
        candidate = document.resolved_path(project_file)
        return DocumentPathResolution(candidate, "project_asset") if candidate.exists() else None

    backup_token = str(document.absolute_path or "").strip()
    absolute_path_missing = False
    if backup_token:
        if _is_foreign_absolute_path_token(backup_token):
            absolute_path_missing = True
        else:
            candidate = Path(backup_token).expanduser().resolve()
            if candidate.exists():
                return DocumentPathResolution(candidate, "absolute_path")
            absolute_path_missing = True

    token = str(document.path or "").strip()
    if token and _is_foreign_absolute_path_token(token):
        absolute_path_missing = True
    token_path = Path(token).expanduser() if token and not _is_foreign_absolute_path_token(token) else None
    if token_path is not None and token_path.is_absolute():
        candidate = token_path.resolve()
        if candidate.exists():
            return DocumentPathResolution(candidate, "path")
        absolute_path_missing = True

    if token:
        candidate = _relative_path_candidate(token, project_dir)
        if candidate is not None and candidate.exists():
            return DocumentPathResolution(
                candidate,
                "relative_path",
                repaired_from_missing_absolute=absolute_path_missing,
            )

    filename = _path_token_filename(token) if token else ""
    if not filename and backup_token:
        filename = _path_token_filename(backup_token)
    if not filename:
        return None

    direct_candidate = (project_dir / filename).resolve()
    if direct_candidate.exists() and direct_candidate.is_file():
        return DocumentPathResolution(
            direct_candidate,
            "project_dir_filename",
            repaired_from_missing_absolute=absolute_path_missing,
        )

    return None


class ProjectIO:
    """Read and write lightweight project files."""

    @staticmethod
    def persistent_payload(
        project: ProjectState,
        *,
        documents: tuple[ImageDocument, ...] | list[ImageDocument] | None = None,
        version: str | None = None,
    ) -> dict:
        """Build a project payload containing persistence fields only.

        ``ImageDocument.to_dict()`` deliberately excludes runtime state such as
        history, dirty trackers, decoded images and display caches.  Accepting an
        explicit ordered document sequence lets the project-session controller
        preserve unresolved placeholders without deep-copying live documents.
        """

        ordered_documents = list(getattr(project, "documents", []) if documents is None else documents)
        projected = ProjectState(
            version=str(getattr(project, "version", "") if version is None else version),
            documents=ordered_documents,
            calibration_presets=list(getattr(project, "calibration_presets", [])),
            project_default_calibration=getattr(project, "project_default_calibration", None),
            project_group_templates=list(getattr(project, "project_group_templates", [])),
            project_rois=list(getattr(project, "project_rois", [])),
            analysis_artifacts=list(getattr(project, "analysis_artifacts", [])),
            metadata=getattr(project, "metadata", {}),
            load_issues=list(getattr(project, "load_issues", [])),
            project_schema_version=getattr(
                project,
                "project_schema_version",
                PROJECT_SCHEMA_VERSION,
            ),
            min_reader_version=getattr(project, "min_reader_version", 1),
            required_features=tuple(
                getattr(project, "required_features", ())
            ),
            compatibility=getattr(
                project,
                "compatibility",
                ProjectCompatibilityState(),
            ),
        )
        payload = projected.to_dict()
        # Model serializers already allocate independent geometry/list payloads,
        # but metadata is intentionally returned by reference.  Snapshot only
        # these comparatively small persistent branches so asset staging cannot
        # mutate the live project through an alias.
        payload["metadata"] = copy.deepcopy(getattr(project, "metadata", {}))
        for document_payload, document in zip(payload.get("documents", []), ordered_documents):
            if isinstance(document_payload, dict):
                document_payload["metadata"] = copy.deepcopy(document.metadata)
        serialized_issues = _serialize_load_issues(projected)
        if serialized_issues:
            payload["load_issues"] = serialized_issues
        return payload

    @staticmethod
    def save_payload(
        payload: dict,
        path: str | Path,
        *,
        document_sources: tuple[ImageDocument, ...] | list[ImageDocument] | None = None,
        preserve_path_document_ids: set[str] | None = None,
    ) -> Path:
        """Atomically publish an already materialized persistent payload.

        Filesystem path normalization is kept at this final boundary.  Only the
        document dictionaries whose paths need normalization are copied; area
        coordinate arrays and all other immutable plan data remain shared.
        """

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_payload = dict(payload)
        payload_documents = payload.get("documents", [])
        if not isinstance(payload_documents, list):
            raise ValueError("项目 payload 的 documents 必须是列表")
        output_documents = list(payload_documents)
        sources = list(document_sources or [])
        if sources and len(sources) != len(output_documents):
            raise ValueError("项目 payload 与文档源数量不一致")
        preserved_ids = set(preserve_path_document_ids or set())
        project_dir = output_path.expanduser().resolve().parent
        for index, source in enumerate(sources):
            document_payload = output_documents[index]
            if not isinstance(document_payload, dict) or source.id in preserved_ids:
                continue
            normalized_payload = dict(document_payload)
            _apply_document_save_path(normalized_payload, source, project_dir)
            output_documents[index] = normalized_payload
        output_payload["documents"] = output_documents
        atomic_write_json(output_path, output_payload, ensure_ascii=False, indent=2)
        return output_path

    @staticmethod
    def save(
        project: ProjectState,
        path: str | Path,
        *,
        preserve_path_document_ids: set[str] | None = None,
    ) -> Path:
        if not project.compatibility.can_overwrite(path):
            features = "、".join(
                project.compatibility.unknown_required_features
            )
            raise ProjectCompatibilityError(
                "项目包含当前程序不支持的必需功能，不能覆盖原文件"
                + (f": {features}" if features else "")
            )
        payload = ProjectIO.persistent_payload(project)
        return ProjectIO.save_payload(
            payload,
            path,
            document_sources=project.documents,
            preserve_path_document_ids=preserve_path_document_ids,
        )

    @staticmethod
    def load(path: str | Path) -> ProjectState:
        input_path = Path(path).expanduser().resolve()
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        sanitized_payload, issues = _sanitize_invalid_calibration_payloads(payload)
        project = ProjectState.from_dict(sanitized_payload)
        project.compatibility = replace(
            project.compatibility,
            source_path=str(input_path),
        )
        project.load_issues = issues
        issues_by_document = {
            str(issue.get("document_id")): str(issue.get("message", "标尺无效"))
            for issue in issues
            if issue.get("kind") == "document_calibration"
        }
        for document in project.documents:
            document.calibration_load_error = issues_by_document.get(document.id)
            matching_issue = next(
                (
                    issue
                    for issue in issues
                    if issue.get("kind") == "document_calibration"
                    and str(issue.get("document_id", "")) == document.id
                ),
                None,
            )
            document.calibration_load_payload = (
                copy.deepcopy(matching_issue.get("raw_payload"))
                if isinstance(matching_issue, dict) and isinstance(matching_issue.get("raw_payload"), dict)
                else None
            )
        return project

    @staticmethod
    def create_pre_upgrade_backup(
        path: str | Path,
        *,
        project: ProjectState | None = None,
    ) -> ProjectUpgradeBackupResult:
        """Create an idempotent backup before a legacy project is upgraded.

        Current-schema files need no backup.  A stable ``.pre-v2.bak`` name is
        used so repeated save attempts do not create unbounded backup chains.
        """

        source_path = Path(path).expanduser().resolve()
        if project is None:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("项目文件根节点必须是 JSON 对象")
            raw_schema_version = payload.get("project_schema_version", 1)
            if (
                isinstance(raw_schema_version, bool)
                or not isinstance(raw_schema_version, int)
                or raw_schema_version < 1
            ):
                raise ValueError("project_schema_version 无效")
            source_schema_version = raw_schema_version
        else:
            source_schema_version = project.compatibility.source_schema_version

        if source_schema_version >= PROJECT_SCHEMA_VERSION:
            return ProjectUpgradeBackupResult(
                source_path=source_path,
                backup_path=None,
                source_schema_version=source_schema_version,
                created=False,
            )

        backup_path = source_path.with_name(
            f"{source_path.name}.pre-v{PROJECT_SCHEMA_VERSION}.bak"
        )
        if backup_path.exists():
            return ProjectUpgradeBackupResult(
                source_path=source_path,
                backup_path=backup_path,
                source_schema_version=source_schema_version,
                created=False,
            )
        atomic_copy_file(source_path, backup_path)
        return ProjectUpgradeBackupResult(
            source_path=source_path,
            backup_path=backup_path,
            source_schema_version=source_schema_version,
            created=True,
        )


def _sanitize_invalid_calibration_payloads(payload: object) -> tuple[dict, list[dict]]:
    if not isinstance(payload, dict):
        raise ValueError("项目文件根节点必须是 JSON 对象")
    # Calibration quarantine is intentionally copy-on-write.  Project payloads can
    # contain millions of geometry coordinates, so deep-copying the root before
    # model construction doubles both the peak memory and the amount of Python
    # object traversal during load.  The sanitizer only ever replaces calibration
    # branches; all unrelated payload branches therefore remain safe to share.
    sanitized = dict(payload)
    issues = _deserialize_load_issues(payload.get("load_issues", []))
    sanitized.pop("load_issues", None)

    project_default = sanitized.get("project_default_calibration")
    if isinstance(project_default, dict):
        try:
            Calibration.from_dict(project_default)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                {
                    "kind": "project_default_calibration",
                    "message": str(exc),
                    "raw_payload": copy.deepcopy(project_default),
                }
            )
            sanitized["project_default_calibration"] = None

    documents = payload.get("documents", [])
    if isinstance(documents, list):
        sanitized_documents: list[object] | None = None
        for index, document in enumerate(documents):
            if not isinstance(document, dict) or not isinstance(document.get("calibration"), dict):
                continue
            raw_calibration = document["calibration"]
            try:
                Calibration.from_dict(raw_calibration)
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(
                    {
                        "kind": "document_calibration",
                        "document_id": str(document.get("id", "")),
                        "message": str(exc),
                        "raw_payload": copy.deepcopy(raw_calibration),
                    }
                )
                if sanitized_documents is None:
                    sanitized_documents = list(documents)
                sanitized_document = dict(document)
                sanitized_document["calibration"] = None
                sanitized_documents[index] = sanitized_document
        if sanitized_documents is not None:
            sanitized["documents"] = sanitized_documents

    presets = payload.get("calibration_presets", [])
    if not isinstance(presets, list):
        issues.append(
            {
                "kind": "calibration_presets",
                "message": "calibration_presets 必须是列表，已忽略",
                "raw_payload": copy.deepcopy(presets),
            }
        )
        sanitized["calibration_presets"] = []
    else:
        valid_presets: list[object] | None = None
        for index, preset in enumerate(presets):
            if not isinstance(preset, dict):
                if valid_presets is None:
                    valid_presets = list(presets[:index])
                issues.append(
                    {
                        "kind": "calibration_preset",
                        "index": index,
                        "message": "标定预设必须是 JSON 对象，已忽略",
                        "raw_payload": copy.deepcopy(preset),
                    }
                )
                continue
            try:
                CalibrationPreset.from_dict(preset)
            except (KeyError, TypeError, ValueError) as exc:
                if valid_presets is None:
                    valid_presets = list(presets[:index])
                issues.append(
                    {
                        "kind": "calibration_preset",
                        "index": index,
                        "message": str(exc),
                        "raw_payload": copy.deepcopy(preset),
                    }
                )
            else:
                if valid_presets is not None:
                    valid_presets.append(preset)
        if valid_presets is not None:
            sanitized["calibration_presets"] = valid_presets
    return sanitized, issues


def _serialize_load_issues(project: ProjectState) -> list[dict]:
    serialized: list[dict] = []
    for issue in project.load_issues:
        kind = str(issue.get("kind", ""))
        if kind in {"document_calibration", "sidecar_calibration"}:
            document = project.get_document(str(issue.get("document_id", "")))
            if document is not None and document.calibration is not None:
                continue
        if kind == "project_default_calibration" and project.project_default_calibration is not None:
            continue
        item = {
            str(key): copy.deepcopy(value)
            for key, value in issue.items()
            if key != "raw_payload"
        }
        if "raw_payload" in issue:
            item["raw_payload_json"] = json.dumps(
                _encode_raw_payload_value(issue["raw_payload"]),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        serialized.append(item)
    return serialized


def _deserialize_load_issues(payload: object) -> list[dict]:
    if not isinstance(payload, list):
        return []
    issues: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        issue = dict(item)
        if "raw_payload" in issue:
            # Legacy issue registries may already contain the decoded calibration
            # payload.  Keep that small branch independent without recursively
            # copying unrelated issue metadata.
            issue["raw_payload"] = copy.deepcopy(issue["raw_payload"])
        raw_json = issue.pop("raw_payload_json", None)
        if isinstance(raw_json, str):
            try:
                issue["raw_payload"] = _decode_raw_payload_value(json.loads(raw_json))
            except (TypeError, ValueError):
                issue["raw_payload_text"] = raw_json
        issues.append(issue)
    return issues
