"""Verified evidence assets inside the existing project asset directory."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath

from fdm.atomic_io import atomic_copy_file


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _persistent_failure(evidence: dict, root: Path) -> dict:
    """A failed image must not prevent saving its measurement and diagnostics."""
    failure = deepcopy(evidence)
    failure["state"] = "failed"
    failure.setdefault("error", "试验原图保存失败")
    raw_path = str(failure.pop("path", ""))
    if raw_path:
        path = Path(raw_path)
        if path.is_absolute():
            try:
                token = PurePosixPath(path.resolve().relative_to(root.resolve()).as_posix())
            except ValueError:
                token = None  # A temporary session path is not a durable asset.
        elif PureWindowsPath(raw_path).is_absolute():
            token = None
        else:
            token = PurePosixPath(raw_path)
        if token and not token.is_absolute() and ".." not in token.parts and token.parts[0] == "experiments":
            failure["path"] = token.as_posix()
    return failure


def stage_experiment_assets(payload: dict, root: Path, created: list[Path]) -> dict:
    if not payload.get("live_experiments"):
        return payload
    output = dict(payload)
    experiments = deepcopy(payload["live_experiments"])
    output["live_experiments"] = experiments
    copied = {}
    for session in experiments:
        records = session.get("measurements", [])
        if isinstance(records, dict):
            records = records.values()
        for record in records:
            evidence = record.get("evidence")
            if not evidence or evidence.get("state") == "disabled":
                continue
            if evidence.get("state") == "failed":
                record["evidence"] = _persistent_failure(evidence, root)
                continue
            if evidence.get("state") != "saved":
                raise ValueError("试验原图尚未保存成功，请先完成写入或处理失败项。")
            path = Path(evidence["path"])
            if not path.is_absolute():
                path = root / path
            digest = str(evidence.get("sha256") or "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("试验原图摘要无效。")
            relative = f"experiments/{digest}.png"
            destination = root / relative
            if digest not in copied:
                if not path.is_file() or _file_sha256(path) != digest:
                    raise ValueError("试验原图缺失或摘要不匹配。")
                if destination.exists():
                    if _file_sha256(destination) != digest:
                        raise ValueError("项目内试验原图摘要不匹配。")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    atomic_copy_file(path, destination)
                    created.append(destination)
                copied[digest] = relative
            evidence["path"] = relative
    return output


def rebase_experiment_assets(experiments, saved_payload: dict, root: Path) -> None:
    """Publish paths only after a verified project save has committed.

    This is a persistence-only update: no new undo command or content revision.
    Retain newer image status, metadata, or content. A previous queued save may
    already have changed only the path, so compare all other evidence fields.
    The override also follows records temporarily absent through undo/delete.
    """
    sessions = {session.id: session for session in experiments}
    root = root.resolve()
    for snapshot in saved_payload.get("live_experiments", []):
        session = sessions.get(snapshot["id"])
        if session is None:
            continue
        records = snapshot.get("measurements", [])
        if isinstance(records, dict):
            records = records.values()
        for saved_record in records:
            original = saved_record.get("evidence")
            if not original or original.get("state") != "saved":
                continue
            record_id = saved_record["id"]
            if record_id not in session._known_measurement_ids:
                continue
            record = session.measurements.get(record_id)
            current = (record.evidence if record is not None else
                       session._evidence_overrides.get(record_id, original))
            if (current is None or
                    {key: value for key, value in current.items() if key != "path"} !=
                    {key: value for key, value in original.items() if key != "path"}):
                continue
            digest = original.get("sha256")
            # The committed writer has verified this digest and destination.
            # Do not reread/hash potentially large PNGs on the GUI thread.
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                continue
            evidence = {**deepcopy(current), "path": str(root / "experiments" / f"{digest}.png")}
            session._evidence_overrides[record_id] = evidence
            if record is not None:
                session.measurements[record_id] = replace(record, evidence=evidence)


def hydrate_experiment_assets(experiments, root: Path) -> None:
    root = root.resolve()
    verified = {}
    for session in experiments:
        for record_id, record in list(session.measurements.items()):
            evidence = record.evidence
            if not evidence or evidence.get("state") != "saved":
                continue
            evidence = dict(evidence)
            token = PurePosixPath(str(evidence.get("path", "")))
            if token.is_absolute() or ".." in token.parts or not token.parts or token.parts[0] != "experiments":
                evidence.update(state="failed", error="试验原图路径无效")
            else:
                path = (root / str(token)).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    evidence.update(state="failed", error="试验原图缺失")
                else:
                    if path not in verified:
                        verified[path] = _file_sha256(path)
                    # Cache the actual digest, not success for another record's
                    # expected digest. A conflicting reference must still fail.
                    if verified[path] != evidence.get("sha256"):
                        evidence.update(state="failed", error="试验原图摘要不匹配")
                evidence["path"] = str(path)
            session.measurements[record_id] = replace(record, evidence=evidence)
