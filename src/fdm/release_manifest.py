from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from typing import Any


RELEASE_MANIFEST_FILENAME = "release-manifest.json"
BUILD_ID_FILENAME = "build-id.txt"
DEVELOPMENT_FEATURES = frozenset(
    {"measurement", "capture", "digital-slide", "area-inference", "magic-segmentation"}
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_root() -> Path:
    override = os.environ.get("FDM_SELF_CHECK_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _safe_manifest_path(root: Path, token: object) -> Path | None:
    normalized = str(token or "").replace("\\", "/").strip()
    pure_path = PurePosixPath(normalized)
    if not normalized or pure_path.is_absolute() or ".." in pure_path.parts:
        return None
    candidate = root.joinpath(*pure_path.parts)
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return candidate


def verify_release_manifest(
    app_root: str | Path,
    *,
    expected_profile: str | None = None,
    expected_version: str | None = None,
    expected_commit: str | None = None,
    require_clean_source: bool = False,
    require_clean_build: bool = False,
    reject_extra_files: bool = False,
) -> dict[str, Any]:
    root = Path(app_root)
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = root / RELEASE_MANIFEST_FILENAME
    result: dict[str, Any] = {
        "ok": False,
        "root": str(root),
        "manifest": str(manifest_path),
        "profile": "",
        "version": "",
        "build_id": "",
        "source_commit": "",
        "features": [],
        "dependency_versions": {},
        "checked_files": 0,
        "errors": errors,
        "warnings": warnings,
    }
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing {RELEASE_MANIFEST_FILENAME}")
        return result
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {RELEASE_MANIFEST_FILENAME}: {exc}")
        return result
    if not isinstance(payload, dict):
        errors.append("release manifest root must be a JSON object")
        return result

    schema_version = payload.get("schema_version")
    if schema_version != 1:
        errors.append(f"unsupported release manifest schema: {schema_version!r}")
    profile = str(payload.get("profile", ""))
    version = str(payload.get("version", ""))
    build_id = str(payload.get("build_id", ""))
    source_commit = str(payload.get("source_commit", ""))
    features_payload = payload.get("features", [])
    if not isinstance(features_payload, list) or any(not isinstance(item, str) for item in features_payload):
        errors.append("release manifest features must be a list of strings")
        features_payload = []
    features = [str(item).strip() for item in features_payload if str(item).strip()]
    dependency_versions = payload.get("dependency_versions", {})
    if not isinstance(dependency_versions, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in dependency_versions.items()
    ):
        errors.append("release manifest dependency_versions must be a string mapping")
        dependency_versions = {}
    result.update(
        profile=profile,
        version=version,
        build_id=build_id,
        source_commit=source_commit,
        features=features,
        dependency_versions=dependency_versions,
    )
    if profile == "full" and not {"area-inference", "magic-segmentation"}.issubset(features):
        errors.append("full profile feature manifest is incomplete")
    if expected_profile is not None and profile != expected_profile:
        errors.append(f"profile mismatch: expected {expected_profile}, found {profile or '<empty>'}")
    if expected_version is not None and version != expected_version:
        errors.append(f"version mismatch: expected {expected_version}, found {version or '<empty>'}")
    if expected_commit is not None and source_commit != expected_commit:
        errors.append(f"stale dist: expected commit {expected_commit}, found {source_commit or '<empty>'}")
    if require_clean_source and bool(payload.get("source_dirty", True)):
        errors.append("release manifest records a dirty source tree")
    if require_clean_build and not bool(payload.get("clean_build", False)):
        errors.append("release manifest was produced by a non-clean build")

    marker_path = root / BUILD_ID_FILENAME
    try:
        marker = marker_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        errors.append(f"missing or unreadable {BUILD_ID_FILENAME}: {exc}")
    else:
        if not build_id or marker != build_id:
            errors.append("build-id mismatch between manifest and build-id.txt")

    entries = payload.get("files", [])
    if not isinstance(entries, list):
        errors.append("release manifest files must be a list")
        entries = []
    listed_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("release manifest contains a non-object file entry")
            continue
        token = str(entry.get("path", "")).replace("\\", "/")
        if token in listed_paths:
            errors.append(f"duplicate manifest file entry: {token}")
            continue
        listed_paths.add(token)
        file_path = _safe_manifest_path(root, token)
        if file_path is None:
            errors.append(f"unsafe manifest file path: {token!r}")
            continue
        if not file_path.is_file():
            errors.append(f"missing packaged file: {token}")
            continue
        expected_size = entry.get("size")
        if not isinstance(expected_size, int) or file_path.stat().st_size != expected_size:
            errors.append(f"size mismatch: {token}")
            continue
        expected_hash = str(entry.get("sha256", ""))
        if len(expected_hash) != 64 or sha256_file(file_path) != expected_hash:
            errors.append(f"sha256 mismatch: {token}")
            continue
        result["checked_files"] = int(result["checked_files"]) + 1

    required_files = payload.get("required_runtime_files", [])
    if not isinstance(required_files, list):
        errors.append("required_runtime_files must be a list")
        required_files = []
    for required in required_files:
        token = str(required).replace("\\", "/")
        if token not in listed_paths:
            errors.append(f"required runtime asset is absent from manifest inventory: {token}")
        required_path = _safe_manifest_path(root, token)
        if required_path is None or not required_path.is_file():
            errors.append(f"missing required runtime asset: {token}")

    if reject_extra_files and root.is_dir():
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != RELEASE_MANIFEST_FILENAME
        }
        extras = sorted(actual_paths - listed_paths)
        missing_inventory = sorted(listed_paths - actual_paths)
        if extras:
            errors.append("unmanifested packaged files: " + ", ".join(extras))
        if missing_inventory:
            errors.append("manifest inventory files missing from package: " + ", ".join(missing_inventory))

    result["ok"] = not errors
    return result


def run_release_self_check(app_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(app_root or release_root())
    report = verify_release_manifest(root)
    functional_checks: dict[str, Any] = {}
    report["functional_checks"] = functional_checks
    if not report.get("ok"):
        return report

    errors = report.setdefault("errors", [])
    warnings = report.setdefault("warnings", [])
    for executable_name in ("FiberDiameterMeasurement.exe", "FiberAreaWorker.exe"):
        executable_path = root / executable_name
        valid, reason = _validate_pe_executable(executable_path)
        functional_checks[f"pe:{executable_name}"] = valid
        if not valid:
            errors.append(f"invalid Windows executable {executable_name}: {reason}")

    try:
        from fdm.models import Calibration

        probe = Calibration(
            mode="self_check",
            pixels_per_unit=5.0,
            unit="um",
            source_label="release-self-check",
        )
        measurement_ok = abs(probe.px_to_unit(25.0) - 5.0) <= 1e-12
    except Exception as exc:  # noqa: BLE001
        measurement_ok = False
        errors.append(f"core measurement self-check failed: {exc}")
    functional_checks["core_measurement"] = measurement_ok
    if not measurement_ok and not any("core measurement" in str(item) for item in errors):
        errors.append("core measurement self-check returned an unexpected value")

    try:
        from fdm.application_launch import SingleInstanceCoordinator
        from PySide6.QtNetwork import QLocalServer, QLocalSocket

        ipc_ok = all((SingleInstanceCoordinator, QLocalServer, QLocalSocket))
    except Exception as exc:  # noqa: BLE001
        ipc_ok = False
        errors.append(f"Qt local IPC self-check failed: {exc}")
    functional_checks["qt_local_ipc"] = ipc_ok
    if not ipc_ok and not any("Qt local IPC" in str(item) for item in errors):
        errors.append("Qt local IPC self-check is unavailable")

    if report.get("profile") == "full":
        versions = report.get("dependency_versions", {})
        for distribution in ("Pillow", "torch", "torchvision"):
            version = str(versions.get(distribution, "")).strip()
            ok = bool(version and version != "not-installed")
            functional_checks[f"dependency:{distribution}"] = ok
            if not ok:
                errors.append(f"full profile dependency is unavailable: {distribution}")

    execute_runtime_probe = bool(
        (sys.platform.startswith("win") and getattr(sys, "frozen", False))
        or os.environ.get("FDM_SELF_CHECK_EXECUTE", "").strip() == "1"
    )
    if report.get("profile") == "full" and execute_runtime_probe and not errors:
        try:
            probe_result = _probe_area_worker(root)
        except Exception as exc:  # noqa: BLE001
            functional_checks["area_worker"] = False
            errors.append(f"area worker functional self-check failed: {exc}")
        else:
            functional_checks["area_worker"] = True
            functional_checks["area_worker_result"] = probe_result
    elif report.get("profile") == "full":
        warnings.append("Windows frozen area-worker execution probe skipped on this host")
        functional_checks["area_worker"] = "skipped_non_windows"

    report["ok"] = not errors
    return report


def _validate_pe_executable(path: Path) -> tuple[bool, str]:
    try:
        with path.open("rb") as stream:
            dos_header = stream.read(64)
            if len(dos_header) < 64 or dos_header[:2] != b"MZ":
                return False, "missing DOS MZ header"
            pe_offset = int.from_bytes(dos_header[60:64], "little")
            if pe_offset < 64 or pe_offset > 16 * 1024 * 1024:
                return False, "invalid PE header offset"
            stream.seek(pe_offset)
            if stream.read(4) != b"PE\x00\x00":
                return False, "missing PE signature"
    except OSError as exc:
        return False, str(exc)
    return True, ""


def _probe_area_worker(root: Path) -> dict[str, Any]:
    worker = root / "FiberAreaWorker.exe"
    weights_dir = root / "runtime" / "area-models"
    vendor_root = root / "runtime" / "area-infer" / "vendor" / "yolact"
    model_files = sorted(weights_dir.glob("*.pth"))
    if not model_files:
        raise RuntimeError("no packaged area model")
    model_file = model_files[0]
    request_id = "self-check-infer"
    with tempfile.TemporaryDirectory(prefix="fdm-self-check-") as tmpdir:
        from PIL import Image

        image_path = Path(tmpdir) / "probe.png"
        Image.new("RGB", (64, 64), color=(255, 255, 255)).save(image_path, format="PNG")
        requests = [
            {
                "protocol": "fdm-area-worker",
                "version": 1,
                "request_id": "self-check-hello",
                "op": "hello",
            },
            {
                "protocol": "fdm-area-worker",
                "version": 1,
                "request_id": request_id,
                "op": "infer",
                "image": {"path": str(image_path)},
                "model": {"name": model_file.stem, "file": model_file.name},
                "runtime": {
                    "weights_dir": str(weights_dir),
                    "vendor_root": str(vendor_root),
                    "device": "cpu",
                    "require_trusted_weights": True,
                    "verify_trusted_weights": True,
                },
                "options": {
                    "include_overlay": False,
                    "inference": {"top_k": 5, "nms_top_k": 20},
                },
            },
        ]
        payload = "".join(
            json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
            for item in requests
        )
        completed = subprocess.run(
            [str(worker), "--persistent"],
            input=payload,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=300,
            check=False,
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or len(lines) != 2:
        raise RuntimeError(
            f"worker protocol failed rc={completed.returncode}, lines={len(lines)}, "
            f"stderr={completed.stderr[-1000:]}"
        )
    responses = [json.loads(line) for line in lines]
    if any(response.get("protocol") != "fdm-area-worker" for response in responses):
        raise RuntimeError("worker returned an invalid protocol envelope")
    if responses[0].get("request_id") != "self-check-hello" or responses[0].get("ok") is not True:
        raise RuntimeError("worker persistent hello failed")
    if responses[1].get("request_id") != request_id or responses[1].get("ok") is not True:
        raise RuntimeError(f"worker CPU inference failed: {responses[1].get('error')}")
    result = responses[1].get("result")
    if not isinstance(result, dict) or not isinstance(result.get("instances"), list):
        raise RuntimeError("worker inference result schema is invalid")
    return {
        "mode": "persistent",
        "device": "cpu",
        "model_file": model_file.name,
        "instance_count": len(result["instances"]),
    }


def packaged_runtime_features(app_root: str | Path | None = None) -> frozenset[str]:
    """Return the explicit packaged feature manifest; source runs expose all features."""

    root = Path(app_root) if app_root is not None else release_root()
    manifest_path = root / RELEASE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        features = {"measurement", "capture", "digital-slide"}
        runtime_root = root / "runtime"
        if (runtime_root / "area-infer").is_dir() and (runtime_root / "area-models").is_dir():
            features.add("area-inference")
        if (runtime_root / "segment-anything").is_dir():
            features.add("magic-segmentation")
        return frozenset(features)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    features = payload.get("features", []) if isinstance(payload, dict) else []
    if not isinstance(features, list):
        return frozenset()
    return frozenset(str(item).strip() for item in features if str(item).strip())


def runtime_capability_hint(app_root: str | Path | None = None) -> str:
    root = Path(app_root) if app_root is not None else release_root()
    manifest_path = root / RELEASE_MANIFEST_FILENAME
    features = packaged_runtime_features(root)
    missing = []
    if "area-inference" not in features:
        missing.append("面积推理")
    if "magic-segmentation" not in features:
        missing.append("智能分割")
    if not missing:
        return ""
    if manifest_path.is_file():
        return f"当前发布 profile 未包含：{'、'.join(missing)}。相关入口已隐藏。"
    return (
        f"当前开发 wheel 未包含完整运行时资产：{'、'.join(missing)}。"
        "正式终端用户交付请使用 Windows full 安装包。"
    )


def format_self_check_report(report: dict[str, Any]) -> str:
    lines = [
        "Release self-check: " + ("PASS" if report.get("ok") else "FAIL"),
        f"Root: {report.get('root', '')}",
        f"Version: {report.get('version', '')}",
        f"Profile: {report.get('profile', '')}",
        f"Build ID: {report.get('build_id', '')}",
        f"Checked files: {report.get('checked_files', 0)}",
    ]
    for error in report.get("errors", []):
        lines.append(f"ERROR: {error}")
    for warning in report.get("warnings", []):
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines)
