"""Backend failures must not silently change production measurement execution."""
from __future__ import annotations

import importlib.machinery
from pathlib import Path
import sys
import sysconfig
from types import SimpleNamespace

import numpy as np
import pytest

from fdm import _native


@pytest.fixture(autouse=True)
def reset_native_loader():
    _native._load_native.cache_clear()
    yield
    _native._load_native.cache_clear()


def missing(name):
    return ModuleNotFoundError(f"No module named {name}", name=name)


def test_auto_falls_back_only_for_absent_extension(monkeypatch):
    monkeypatch.setenv("FDM_NATIVE_BACKEND", "auto")
    def absent(*args, **kwargs):
        raise missing(_native.EXTENSION_NAME)
    monkeypatch.setattr(_native, "require_native", absent)
    assert _native.selected_backend() is None
    monkeypatch.setenv("FDM_NATIVE_BACKEND", "native")
    with pytest.raises(ModuleNotFoundError):
        _native.selected_backend()


@pytest.mark.parametrize("error", [missing("another_dependency"), ImportError("bad ABI"), OSError("bad DLL"), RuntimeError("bad source")])
def test_broken_native_never_falls_back(monkeypatch, error):
    monkeypatch.setenv("FDM_NATIVE_BACKEND", "auto")
    def broken(*args, **kwargs):
        raise error
    monkeypatch.setattr(_native, "require_native", broken)
    with pytest.raises(type(error), match=str(error)):
        _native.selected_backend()


def test_source_python_and_frozen_native_selection(monkeypatch):
    binary = object()
    monkeypatch.setattr(_native, "require_native", lambda: binary)
    monkeypatch.setenv("FDM_NATIVE_BACKEND", "python")
    assert _native.selected_backend() is None
    monkeypatch.setenv("FDM_NATIVE_BACKEND", "native")
    assert _native.selected_backend() is binary
    monkeypatch.setenv("FDM_NATIVE_BACKEND", "bad")
    with pytest.raises(ValueError, match="FDM_NATIVE_BACKEND"):
        _native.selected_backend()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("FDM_NATIVE_BACKEND", "python")
    assert _native.selected_backend() is binary


@pytest.mark.parametrize("field,value,message", [
    ("api_version", 0, "接口版本"),
    ("python_abi", "different-python", "ABI"),
    ("source_sha256", "old-binary", "源码"),
])
def test_binary_identity_checked(monkeypatch, field, value, message):
    root = Path(__file__).resolve().parents[1]
    info = {"api_version": 1, "python_abi": sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag,
            "source_sha256": _native.native_source_sha256(root)}
    info[field] = value
    fake = SimpleNamespace(__file__="_kernels" + importlib.machinery.EXTENSION_SUFFIXES[0], build_info=lambda: info)
    monkeypatch.setattr(_native.importlib, "import_module", lambda name: fake)
    with pytest.raises(RuntimeError, match=message):
        _native.require_native(validate_source=True)


def test_python_file_cannot_impersonate_binary(monkeypatch):
    monkeypatch.setattr(_native.importlib, "import_module", lambda name: SimpleNamespace(__file__="_kernels.py"))
    with pytest.raises(RuntimeError, match="已编译"):
        _native.require_native()


def test_source_hash_includes_names_and_headers(tmp_path):
    directory = tmp_path / "native"
    directory.mkdir()
    (directory / "a.cpp").write_text("ab")
    original = _native.native_source_sha256(tmp_path)
    (directory / "a.cpp").rename(directory / "b.cpp")
    assert _native.native_source_sha256(tmp_path) != original
    (directory / "b.cpp").rename(directory / "a.cpp")
    assert _native.native_source_sha256(tmp_path) == original
    (directory / "a.hpp").write_text("header")
    assert _native.native_source_sha256(tmp_path) != original


def test_snapshot_owns_writable_and_strided_inputs():
    source = np.arange(30, dtype=np.float64).reshape(5, 6)
    for view in (source, source[:, ::-1], source[::2, ::2]):
        snapshot = _native.array_snapshot(view, ndim=2, dtypes=("float64",))
        assert not snapshot.flags.writeable and snapshot.flags.c_contiguous
        assert not np.shares_memory(snapshot, source)
        np.testing.assert_array_equal(snapshot, view)
    source.setflags(write=False)
    assert _native.array_snapshot(source, ndim=2, dtypes=("float64",)) is source
    with pytest.raises(TypeError):
        _native.array_snapshot(source.astype(np.float32), ndim=2, dtypes=("float64",))
    with pytest.raises(ValueError):
        _native.array_snapshot(source, ndim=1, dtypes=("float64",))
