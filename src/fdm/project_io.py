from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import copy
import json
import math

from fdm.atomic_io import atomic_write_json
from fdm.models import Calibration, CalibrationPreset, ImageDocument, ProjectState


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
    def save(
        project: ProjectState,
        path: str | Path,
        *,
        preserve_path_document_ids: set[str] | None = None,
    ) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = project.to_dict()
        project_dir = output_path.expanduser().resolve().parent
        preserved_ids = set(preserve_path_document_ids or set())
        for document_payload, document in zip(payload.get("documents", []), project.documents):
            if isinstance(document_payload, dict):
                if document.id in preserved_ids:
                    continue
                _apply_document_save_path(document_payload, document, project_dir)
        serialized_issues = _serialize_load_issues(project)
        if serialized_issues:
            payload["load_issues"] = serialized_issues
        atomic_write_json(output_path, payload, ensure_ascii=False, indent=2)
        return output_path

    @staticmethod
    def load(path: str | Path) -> ProjectState:
        input_path = Path(path)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        sanitized_payload, issues = _sanitize_invalid_calibration_payloads(payload)
        project = ProjectState.from_dict(sanitized_payload)
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


def _sanitize_invalid_calibration_payloads(payload: object) -> tuple[dict, list[dict]]:
    if not isinstance(payload, dict):
        raise ValueError("项目文件根节点必须是 JSON 对象")
    sanitized = copy.deepcopy(payload)
    issues = _deserialize_load_issues(sanitized.pop("load_issues", []))

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

    documents = sanitized.get("documents", [])
    if isinstance(documents, list):
        for document in documents:
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
                document["calibration"] = None

    presets = sanitized.get("calibration_presets", [])
    if isinstance(presets, list):
        valid_presets: list[object] = []
        for index, preset in enumerate(presets):
            if not isinstance(preset, dict):
                valid_presets.append(preset)
                continue
            try:
                CalibrationPreset.from_dict(preset)
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(
                    {
                        "kind": "calibration_preset",
                        "index": index,
                        "message": str(exc),
                        "raw_payload": copy.deepcopy(preset),
                    }
                )
            else:
                valid_presets.append(preset)
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
        issue = copy.deepcopy(item)
        raw_json = issue.pop("raw_payload_json", None)
        if isinstance(raw_json, str):
            try:
                issue["raw_payload"] = _decode_raw_payload_value(json.loads(raw_json))
            except (TypeError, ValueError):
                issue["raw_payload_text"] = raw_json
        issues.append(issue)
    return issues
