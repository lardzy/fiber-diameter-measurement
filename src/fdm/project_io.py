from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import json

from fdm.models import ImageDocument, ProjectState


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
    def save(project: ProjectState, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = project.to_dict()
        project_dir = output_path.expanduser().resolve().parent
        for document_payload, document in zip(payload.get("documents", []), project.documents):
            if isinstance(document_payload, dict):
                _apply_document_save_path(document_payload, document, project_dir)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def load(path: str | Path) -> ProjectState:
        input_path = Path(path)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        return ProjectState.from_dict(payload)
