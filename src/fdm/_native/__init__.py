"""Internal native-kernel selection; Python references remain usable from source.

Only an absent extension permits automatic fallback. A broken or stale binary
must be repaired instead of silently changing the backend used for measurement.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
from importlib.machinery import EXTENSION_SUFFIXES
import os
from pathlib import Path
import sys
import sysconfig
from types import ModuleType

API_VERSION = 1
EXTENSION_NAME = "fdm._native._kernels"


def native_source_sha256(root: Path) -> str:
    """Match setup.py's fingerprint, including paths and file boundaries."""
    digest = hashlib.sha256()
    sources = sorted(
        path for path in (root / "native").rglob("*")
        if path.is_file() and path.suffix in {".cpp", ".hpp"}
    )
    if not sources:
        raise RuntimeError("未找到原生内核源码，无法检查构建版本。")
    for path in sources:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_root() -> Path | None:
    if getattr(sys, "frozen", False):
        return None
    root = Path(__file__).resolve().parents[3]
    return root if (root / "native").is_dir() and (root / "pyproject.toml").is_file() else None


def _validate_source(module: ModuleType) -> None:
    root = _source_root()
    if root is not None and module.build_info().get("source_sha256") != native_source_sha256(root):
        raise RuntimeError("原生内核与当前源码不一致，请重新构建 fdm._native._kernels。")


@lru_cache(maxsize=1)
def _load_native() -> ModuleType:
    module = importlib.import_module(EXTENSION_NAME)
    filename = str(getattr(module, "__file__", ""))
    if not any(filename.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        raise RuntimeError("原生内核必须由已编译的扩展模块提供。")
    info = module.build_info()
    if info.get("api_version") != API_VERSION:
        raise RuntimeError("原生内核接口版本不匹配，请重新构建。")
    abi = sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag
    if info.get("python_abi") != abi:
        raise RuntimeError("原生内核 Python ABI 不匹配，请使用当前 Python 重新构建。")
    _validate_source(module)
    return module


def require_native(*, validate_source: bool = False) -> ModuleType:
    """Return a verified binary regardless of the source backend preference."""
    module = _load_native()
    if validate_source:
        _validate_source(module)
    return module


def build_info() -> dict[str, object]:
    return dict(require_native().build_info())


def selected_backend() -> ModuleType | None:
    """Read the preference at each service call to support paired diagnostics."""
    mode = os.environ.get("FDM_NATIVE_BACKEND", "auto")
    if getattr(sys, "frozen", False):
        mode = "native"
    if mode not in {"auto", "python", "native"}:
        raise ValueError("FDM_NATIVE_BACKEND 必须为 auto、python 或 native。")
    if mode == "python":
        return None
    try:
        return require_native()
    except ModuleNotFoundError as exc:
        if mode == "auto" and exc.name == EXTENSION_NAME:
            return None
        raise


def array_snapshot(value, *, ndim: int, dtypes: tuple[str, ...]):
    """Own writable/strided inputs before releasing the GIL.

    Already immutable contiguous task snapshots can be borrowed for the call;
    their owners must not mutate them through another alias concurrently.
    No dtype coercion is allowed at this numerical boundary.
    """
    import numpy as np

    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"原生内核输入必须是 {ndim} 维数组。")
    if array.dtype not in tuple(np.dtype(dtype) for dtype in dtypes):
        raise TypeError(f"原生内核输入类型必须为 {', '.join(dtypes)}。")
    if array.flags.writeable or not array.flags.c_contiguous or not array.flags.aligned:
        array = np.array(array, order="C", copy=True)
        array.setflags(write=False)
    return array
