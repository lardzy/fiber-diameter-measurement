from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import zipfile

from fdm.settings import OfflineSegmentationEnginePack


ENGINE_MANIFEST_KIND = "fdm.offline_segmentation_engine"
ENGINE_MANIFEST_VERSION = 1
SUPPORTED_ENGINE_IDS = {"sam3", "micro_sam"}


@dataclass(frozen=True, slots=True)
class EnginePackInspection:
    record: OfflineSegmentationEnginePack
    manifest_path: Path
    python_path: Path
    diagnostic_arguments: tuple[str, ...]
    resource_count: int
    total_resource_bytes: int


@dataclass(frozen=True, slots=True)
class EngineDiagnosticResult:
    ok: bool
    message: str
    details: dict[str, object]
    stdout: str = ""
    stderr: str = ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_path(root: Path, relative: object) -> Path:
    token = PurePosixPath(str(relative or "").replace("\\", "/"))
    if token.is_absolute() or ".." in token.parts:
        raise ValueError(f"引擎包包含不安全的相对路径：{relative}")
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*token.parts)).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"引擎包路径越出根目录：{relative}")
    return candidate


class OfflineSegmentationEngineService:
    """Validate and diagnose opt-in local engine packs.

    The service deliberately has no inference method.  Installing a pack here
    does not register a new magic-wand tool or alter the current EdgeSAM path.
    """

    def __init__(self, managed_root: str | Path) -> None:
        self.managed_root = Path(managed_root).expanduser()

    @staticmethod
    def manifest_path(pack_root: str | Path) -> Path:
        root = Path(pack_root).expanduser()
        for filename in ("engine.json", "manifest.json"):
            candidate = root / filename
            if candidate.is_file():
                return candidate
        raise ValueError("离线引擎目录缺少 engine.json 或 manifest.json。")

    def inspect(self, pack_root: str | Path, *, managed: bool | None = None) -> EnginePackInspection:
        root = Path(pack_root).expanduser().resolve()
        manifest_path = self.manifest_path(root)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取离线引擎 manifest：{exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("离线引擎 manifest 必须是 JSON 对象。")
        if payload.get("kind") != ENGINE_MANIFEST_KIND:
            raise ValueError(f"离线引擎 kind 必须为 {ENGINE_MANIFEST_KIND}。")
        if int(payload.get("schema_version", 0) or 0) != ENGINE_MANIFEST_VERSION:
            raise ValueError("离线引擎 manifest 版本不受支持。")
        engine_id = str(payload.get("engine_id", "")).strip().lower()
        if engine_id not in SUPPORTED_ENGINE_IDS:
            raise ValueError("离线引擎仅支持 sam3 或 micro_sam。")
        if str(payload.get("device", "cpu")).strip().lower() != "cpu":
            raise ValueError("当前离线引擎管理仅接受具备纯 CPU 路径的包。")
        python_path = _contained_path(root, payload.get("python", "python"))
        if not python_path.is_file():
            raise ValueError(f"引擎 Python 不存在：{python_path}")
        resources = payload.get("resources", [])
        if not isinstance(resources, list):
            raise ValueError("resources 必须是数组。")
        resource_count = 0
        total_bytes = 0
        for item in resources:
            if not isinstance(item, dict):
                raise ValueError("resources 中存在无效条目。")
            required = bool(item.get("required", True))
            path = _contained_path(root, item.get("path", ""))
            if not path.is_file():
                if required:
                    raise ValueError(f"引擎必需资源不存在：{path.relative_to(root)}")
                continue
            expected = str(item.get("sha256", "")).strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(
                    f"引擎资源缺少有效 SHA-256：{path.relative_to(root)}"
                )
            if _sha256_file(path) != expected:
                raise ValueError(f"引擎资源校验失败：{path.relative_to(root)}")
            resource_count += 1
            total_bytes += path.stat().st_size
        diagnostic = payload.get("diagnostic", [])
        if not isinstance(diagnostic, list) or not all(isinstance(item, str) for item in diagnostic):
            raise ValueError("diagnostic 必须是字符串参数数组。")
        manifest_digest = _sha256_file(manifest_path)
        is_managed = (
            bool(managed)
            if managed is not None
            else self.managed_root.resolve() in root.parents
        )
        record = OfflineSegmentationEnginePack(
            engine_id=engine_id,
            display_name=str(payload.get("display_name", engine_id)).strip() or engine_id,
            version=str(payload.get("version", "unknown")).strip() or "unknown",
            path=str(root),
            manifest_sha256=manifest_digest,
            device="cpu",
            managed=is_managed,
        ).normalized_copy()
        return EnginePackInspection(
            record=record,
            manifest_path=manifest_path,
            python_path=python_path,
            diagnostic_arguments=tuple(diagnostic),
            resource_count=resource_count,
            total_resource_bytes=total_bytes,
        )

    def import_package(self, source: str | Path) -> EnginePackInspection:
        source_path = Path(source).expanduser().resolve()
        self.managed_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".engine-import-", dir=self.managed_root))
        try:
            if source_path.is_dir():
                shutil.copytree(source_path, staging / "pack", dirs_exist_ok=True)
                unpacked = staging / "pack"
            elif source_path.is_file() and source_path.suffix.casefold() == ".zip":
                unpacked = staging / "pack"
                unpacked.mkdir()
                with zipfile.ZipFile(source_path) as archive:
                    for info in archive.infolist():
                        _contained_path(unpacked, info.filename)
                    archive.extractall(unpacked)
                roots = [path for path in unpacked.iterdir() if path.is_dir()]
                if not any((unpacked / name).is_file() for name in ("engine.json", "manifest.json")) and len(roots) == 1:
                    unpacked = roots[0]
            else:
                raise ValueError("请选择离线引擎目录或 ZIP 包。")
            inspection = self.inspect(unpacked, managed=True)
            destination_name = (
                f"{inspection.record.engine_id}-{inspection.record.version}-"
                f"{inspection.record.manifest_sha256[:10]}"
            )
            destination = self.managed_root / _safe_folder_name(destination_name)
            if destination.exists():
                shutil.rmtree(staging, ignore_errors=True)
                return self.inspect(destination, managed=True)
            unpacked.rename(destination)
            shutil.rmtree(staging, ignore_errors=True)
            return self.inspect(destination, managed=True)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def diagnose(
        self,
        record: OfflineSegmentationEnginePack,
        *,
        timeout_seconds: float = 90.0,
    ) -> EngineDiagnosticResult:
        inspection = self.inspect(record.path, managed=record.managed)
        if inspection.record.manifest_sha256 != record.manifest_sha256:
            return EngineDiagnosticResult(
                ok=False,
                message="引擎 manifest 已变化，请重新导入或重新关联。",
                details={"code": "manifest_changed"},
            )
        if not inspection.diagnostic_arguments:
            return EngineDiagnosticResult(
                ok=True,
                message="文件、校验和与 CPU 配置检查通过；该包未提供运行诊断命令。",
                details={
                    "resource_count": inspection.resource_count,
                    "resource_bytes": inspection.total_resource_bytes,
                    "runtime_executed": False,
                },
            )
        root = Path(record.path)
        arguments = [
            str(_contained_path(root, item[1:])) if item.startswith("@") else item
            for item in inspection.diagnostic_arguments
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "FDM_ENGINE_DEVICE": "cpu",
            }
        )
        try:
            completed = subprocess.run(
                [str(inspection.python_path), *arguments],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=max(1.0, float(timeout_seconds)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return EngineDiagnosticResult(
                ok=False,
                message="引擎 CPU 诊断超时。",
                details={"code": "timeout", "timeout_seconds": timeout_seconds},
                stdout=str(exc.stdout or "")[-4000:],
                stderr=str(exc.stderr or "")[-4000:],
            )
        stdout = completed.stdout[-8000:]
        stderr = completed.stderr[-8000:]
        details: dict[str, object] = {
            "return_code": completed.returncode,
            "resource_count": inspection.resource_count,
            "resource_bytes": inspection.total_resource_bytes,
            "runtime_executed": True,
        }
        try:
            parsed = json.loads(completed.stdout)
            if isinstance(parsed, dict):
                details.update(parsed)
        except json.JSONDecodeError:
            pass
        ok = completed.returncode == 0 and bool(details.get("ok", True))
        return EngineDiagnosticResult(
            ok=ok,
            message=("CPU 诊断通过。" if ok else "CPU 诊断失败，请查看详情。"),
            details=details,
            stdout=stdout,
            stderr=stderr,
        )

    def remove_managed_pack(self, record: OfflineSegmentationEnginePack) -> bool:
        root = Path(record.path).expanduser().resolve()
        managed_root = self.managed_root.resolve()
        if not record.managed or managed_root not in root.parents:
            return False
        if root == managed_root:
            raise ValueError("拒绝删除离线引擎根目录。")
        shutil.rmtree(root)
        return True


def _safe_folder_name(value: str) -> str:
    token = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return token.strip("._-")[:160] or "engine-pack"
