"""Safe, atomic persistence for large analysis arrays.

Analysis arrays are stored as compressed NPZ without pickle.  A canonical
UTF-8 JSON manifest is embedded as a uint8 member so the archive can be
validated independently of the project JSON before it is accepted or copied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np
from numpy.typing import NDArray

from fdm.analysis_artifacts import AnalysisAssetReference
from fdm.atomic_io import atomic_copy_file, atomic_replace_file, staged_path_for


ANALYSIS_NPZ_FORMAT = "fdm.analysis-npz.v1"
ANALYSIS_NPZ_MANIFEST_MEMBER = "__fdm_analysis_manifest__"
MAX_ANALYSIS_ASSET_UNCOMPRESSED_BYTES = 1 << 30
_MAX_ARCHIVE_CONTAINER_BYTES = MAX_ANALYSIS_ASSET_UNCOMPRESSED_BYTES + (16 << 20)
_MAX_MANIFEST_BYTES = 1 << 20
_MEMBER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SCHEMA_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class AnalysisAssetFileInfo:
    path: Path
    sha256: str
    byte_count: int
    uncompressed_byte_count: int
    schema: str
    members: tuple[tuple[str, str, tuple[int, ...]], ...]


def write_safe_analysis_npz(
    target: str | Path,
    *,
    schema: str,
    arrays: Mapping[str, NDArray[Any]],
    metadata: Mapping[str, object] | None = None,
) -> AnalysisAssetFileInfo:
    """Atomically write one pickle-free analysis archive."""

    target_path = Path(target)
    if target_path.suffix.casefold() != ".npz":
        raise ValueError("分析数组资产必须使用 .npz 扩展名")
    frozen_arrays = _normalize_arrays(arrays)
    manifest = _build_manifest(
        schema=schema,
        arrays=frozen_arrays,
        metadata=metadata,
    )
    manifest_bytes = _canonical_json(manifest).encode("utf-8")
    archive_arrays = dict(frozen_arrays)
    archive_arrays[ANALYSIS_NPZ_MANIFEST_MEMBER] = np.frombuffer(
        manifest_bytes,
        dtype=np.uint8,
    )
    with staged_path_for(target_path, suffix=".npz") as staged_path:
        with staged_path.open("wb") as stream:
            np.savez_compressed(stream, **archive_arrays)
            stream.flush()
            os.fsync(stream.fileno())
        staged_info = inspect_safe_analysis_npz(staged_path)
        if staged_info.schema != str(schema):
            raise OSError("分析资产写入后的 schema 校验失败")
        atomic_replace_file(staged_path, target_path)
    return inspect_safe_analysis_npz(target_path)


def inspect_safe_analysis_npz(path: str | Path) -> AnalysisAssetFileInfo:
    """Validate an NPZ archive without enabling pickle deserialization."""

    source = Path(path)
    if source.suffix.casefold() != ".npz":
        raise ValueError("分析数组资产必须使用 .npz 扩展名")
    if not source.is_file():
        raise FileNotFoundError(f"分析资产不存在：{source}")
    if source.stat().st_size > _MAX_ARCHIVE_CONTAINER_BYTES:
        raise ValueError("分析资产文件超过 1 GiB 安全上限")
    try:
        with zipfile.ZipFile(source, "r") as container:
            entries = container.infolist()
            if len(entries) != len({entry.filename for entry in entries}):
                raise ValueError("分析资产 ZIP 包含重复成员")
            if any(entry.flag_bits & 0x1 for entry in entries):
                raise ValueError("分析资产 ZIP 不允许加密成员")
            manifest_entries = [
                entry
                for entry in entries
                if entry.filename
                == f"{ANALYSIS_NPZ_MANIFEST_MEMBER}.npy"
            ]
            if len(manifest_entries) != 1:
                raise ValueError("分析资产缺少唯一的安全 manifest")
            if manifest_entries[0].file_size > _MAX_MANIFEST_BYTES:
                raise ValueError("分析资产 manifest 超过 1 MiB 安全上限")
            declared_uncompressed = sum(int(entry.file_size) for entry in entries)
            if declared_uncompressed > _MAX_ARCHIVE_CONTAINER_BYTES:
                raise ValueError("分析资产 ZIP 解压后超过 1 GiB 安全上限")
    except zipfile.BadZipFile as exc:
        raise ValueError("分析资产不是合法 NPZ/ZIP 文件") from exc
    try:
        with np.load(source, allow_pickle=False) as archive:
            names = tuple(archive.files)
            if names.count(ANALYSIS_NPZ_MANIFEST_MEMBER) != 1:
                raise ValueError("分析资产缺少唯一的安全 manifest")
            manifest_raw = np.asarray(archive[ANALYSIS_NPZ_MANIFEST_MEMBER])
            if manifest_raw.dtype != np.uint8 or manifest_raw.ndim != 1:
                raise ValueError("分析资产 manifest 必须是 uint8 一维数组")
            try:
                manifest = json.loads(manifest_raw.tobytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("分析资产 manifest 不是合法 UTF-8 JSON") from exc
            normalized_manifest = _validate_manifest(manifest)
            expected_members = normalized_manifest["members"]
            actual_data_names = tuple(
                name
                for name in names
                if name != ANALYSIS_NPZ_MANIFEST_MEMBER
            )
            if set(actual_data_names) != set(expected_members):
                raise ValueError("分析资产成员与 manifest 不一致")
            members: list[tuple[str, str, tuple[int, ...]]] = []
            total_bytes = 0
            for name in sorted(actual_data_names):
                array = np.asarray(archive[name])
                if array.dtype.hasobject:
                    raise ValueError("分析资产禁止 object dtype")
                if array.dtype.kind not in "biufc":
                    raise ValueError(f"分析资产成员 {name} 使用了不安全 dtype")
                expected = expected_members[name]
                expected_dtype = str(expected["dtype"])
                expected_shape = tuple(int(value) for value in expected["shape"])
                if str(array.dtype) != expected_dtype or array.shape != expected_shape:
                    raise ValueError(f"分析资产成员 {name} 的 dtype 或 shape 与 manifest 不一致")
                total_bytes += int(array.nbytes)
                if total_bytes > MAX_ANALYSIS_ASSET_UNCOMPRESSED_BYTES:
                    raise ValueError("分析资产解压后超过 1 GiB 安全上限")
                members.append((name, str(array.dtype), tuple(array.shape)))
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"无法读取安全分析资产：{exc}") from exc
    return AnalysisAssetFileInfo(
        path=source,
        sha256=file_sha256(source),
        byte_count=source.stat().st_size,
        uncompressed_byte_count=total_bytes,
        schema=str(normalized_manifest["schema"]),
        members=tuple(members),
    )


def validate_analysis_asset_reference(
    source: str | Path,
    reference: AnalysisAssetReference,
) -> AnalysisAssetFileInfo:
    """Validate path-independent archive content against a project reference."""

    if not isinstance(reference, AnalysisAssetReference):
        raise TypeError("reference 必须是 AnalysisAssetReference")
    info = inspect_safe_analysis_npz(source)
    if info.sha256 != reference.sha256:
        raise ValueError(
            f"分析资产 SHA256 不一致：期望 {reference.sha256}，实际 {info.sha256}"
        )
    metadata = reference.metadata
    expected_schema = str(metadata.get("schema", "")).strip()
    if not expected_schema or expected_schema != info.schema:
        raise ValueError("分析资产 schema 与项目引用不一致")
    expected_members = metadata.get("members")
    if isinstance(expected_members, Mapping):
        actual = {
            name: {"dtype": dtype, "shape": list(shape)}
            for name, dtype, shape in info.members
        }
        normalized_expected = {
            str(name): {
                "dtype": str(descriptor.get("dtype", "")),
                "shape": list(descriptor.get("shape", ())),
            }
            for name, descriptor in expected_members.items()
            if isinstance(descriptor, Mapping)
        }
        if normalized_expected != actual:
            raise ValueError("分析资产成员描述与项目引用不一致")
    return info


def copy_verified_analysis_asset(
    source: str | Path,
    target: str | Path,
    reference: AnalysisAssetReference,
) -> AnalysisAssetFileInfo:
    """Verify, atomically copy, and re-verify one referenced analysis asset."""

    source_path = Path(source)
    target_path = Path(target)
    source_info = validate_analysis_asset_reference(source_path, reference)
    try:
        same_path = source_path.resolve() == target_path.resolve()
    except OSError:
        same_path = False
    if not same_path:
        atomic_copy_file(source_path, target_path)
    output_info = validate_analysis_asset_reference(target_path, reference)
    if output_info.sha256 != source_info.sha256:
        raise OSError("分析资产复制后哈希发生变化")
    return output_info


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_arrays(
    arrays: Mapping[str, NDArray[Any]],
) -> dict[str, NDArray[np.generic]]:
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("分析资产至少需要一个数组")
    result: dict[str, NDArray[np.generic]] = {}
    total_bytes = 0
    for raw_name, value in arrays.items():
        name = str(raw_name)
        if (
            name == ANALYSIS_NPZ_MANIFEST_MEMBER
            or not _MEMBER_PATTERN.fullmatch(name)
        ):
            raise ValueError(f"分析资产成员名称不合法：{name!r}")
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind not in "biufc":
            raise TypeError(f"分析资产成员 {name} 使用了不安全 dtype")
        if any(
            isinstance(dimension, bool)
            or int(dimension) != dimension
            or int(dimension) < 0
            for dimension in array.shape
        ):
            raise ValueError(f"分析资产成员 {name} 的 shape 不合法")
        total_bytes += int(array.nbytes)
        if total_bytes > MAX_ANALYSIS_ASSET_UNCOMPRESSED_BYTES:
            raise ValueError("分析资产解压后超过 1 GiB 安全上限")
        frozen = np.ascontiguousarray(array).copy()
        frozen.setflags(write=False)
        result[name] = frozen
    return result


def _build_manifest(
    *,
    schema: str,
    arrays: Mapping[str, NDArray[np.generic]],
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    schema_token = str(schema or "").strip()
    if not _SCHEMA_PATTERN.fullmatch(schema_token):
        raise ValueError("分析资产 schema 不合法")
    extra_metadata = dict(metadata or {})
    extra_metadata.pop("schema", None)
    extra_metadata.pop("members", None)
    extra_metadata.pop("allow_pickle", None)
    manifest = {
        "format": ANALYSIS_NPZ_FORMAT,
        "schema": schema_token,
        "allow_pickle": False,
        "members": {
            name: {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
            for name, array in sorted(arrays.items())
        },
        "metadata": extra_metadata,
    }
    _canonical_json(manifest)
    return manifest


def _validate_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("分析资产 manifest 必须是 JSON 对象")
    if set(value) != {
        "format",
        "schema",
        "allow_pickle",
        "members",
        "metadata",
    }:
        raise ValueError("分析资产 manifest 字段不完整或包含未知字段")
    if value.get("format") != ANALYSIS_NPZ_FORMAT:
        raise ValueError("不支持的分析资产格式版本")
    if value.get("allow_pickle") is not False:
        raise ValueError("分析资产禁止启用 pickle")
    schema = str(value.get("schema", "")).strip()
    if not _SCHEMA_PATTERN.fullmatch(schema):
        raise ValueError("分析资产 schema 不合法")
    members = value.get("members")
    if not isinstance(members, dict) or not members:
        raise ValueError("分析资产 manifest 未声明数组成员")
    normalized_members: dict[str, dict[str, object]] = {}
    total_bytes = 0
    for raw_name, raw_descriptor in members.items():
        name = str(raw_name)
        if not _MEMBER_PATTERN.fullmatch(name):
            raise ValueError(f"分析资产成员名称不合法：{name!r}")
        if not isinstance(raw_descriptor, dict) or set(raw_descriptor) != {
            "dtype",
            "shape",
        }:
            raise ValueError(f"分析资产成员 {name} 描述不合法")
        try:
            dtype = np.dtype(str(raw_descriptor["dtype"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"分析资产成员 {name} 的 dtype 不合法") from exc
        if dtype.hasobject or dtype.kind not in "biufc":
            raise ValueError(f"分析资产成员 {name} 使用了不安全 dtype")
        shape = raw_descriptor["shape"]
        if not isinstance(shape, list) or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in shape
        ):
            raise ValueError(f"分析资产成员 {name} 的 shape 不合法")
        element_count = math.prod(shape)
        total_bytes += element_count * dtype.itemsize
        if total_bytes > MAX_ANALYSIS_ASSET_UNCOMPRESSED_BYTES:
            raise ValueError("分析资产 manifest 声明的数据超过 1 GiB 安全上限")
        normalized_members[name] = {
            "dtype": str(dtype),
            "shape": shape,
        }
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("分析资产 manifest.metadata 必须是 JSON 对象")
    _canonical_json(metadata)
    return {
        "format": ANALYSIS_NPZ_FORMAT,
        "schema": schema,
        "allow_pickle": False,
        "members": normalized_members,
        "metadata": metadata,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
