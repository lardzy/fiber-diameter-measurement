"""Build the small internal kernels while keeping setuptools package metadata."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import sysconfig

from setuptools import setup


ROOT = Path(__file__).resolve().parent
native_mode = os.environ.get("FDM_BUILD_NATIVE", "1")
if native_mode not in {"0", "1"}:
    raise RuntimeError("FDM_BUILD_NATIVE must be 0 (reference development) or 1 (native build)")

extensions = []
commands = {}
if native_mode == "1":
    # Match the interpreter's deployment floor. pybind11 otherwise adds 10.14
    # for C++17 even when this ARM64 Python was built for a newer macOS.
    if sys.platform == "darwin" and "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
        deployment_target = sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET")
        if deployment_target:
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = str(deployment_target)
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    native_files = sorted(
        path for path in (ROOT / "native").rglob("*")
        if path.is_file() and path.suffix in {".cpp", ".hpp"}
    )
    digest = hashlib.sha256()
    for path in native_files:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    # Targets are explicit: the Windows capture module has a separate module
    # initializer and OS libraries, and must never enter portable _kernels.
    sources = ["native/kernels.cpp", "native/cu_manual.cpp", "native/cu_automatic.cpp"]
    if not sources:
        raise RuntimeError("The native source files are missing; rebuild from a complete source distribution")
    abi = sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag
    extensions = [Pybind11Extension(
        "fdm._native._kernels",
        sources,
        include_dirs=[str(ROOT / "native")],
        depends=[str(path) for path in native_files if path.suffix == ".hpp"],
        cxx_std=17,
        define_macros=[
            ("FDM_NATIVE_API_VERSION", "1"),
            ("FDM_NATIVE_SOURCE_SHA256", f'"{digest.hexdigest()}"'),
            ("FDM_NATIVE_PYTHON_ABI", f'"{abi}"'),
            ("FDM_NATIVE_BUILD_TYPE", '"release"'),
        ],
        extra_compile_args=["/fp:strict", "/utf-8"] if sys.platform == "win32" else ["-fno-fast-math", "-ffp-contract=off"],
    )]
    if sys.platform == "win32":
        extensions.append(Pybind11Extension(
            "fdm._native._windows_capture", ["native/windows_capture.cpp"],
            cxx_std=17,
            define_macros=[("FDM_NATIVE_SOURCE_SHA256", f'"{digest.hexdigest()}"'),
                           ("FDM_NATIVE_PYTHON_ABI", f'"{abi}"')],
            libraries=["d3d11", "dxgi", "windowsapp", "dwmapi", "user32", "runtimeobject"],
            extra_compile_args=["/EHsc", "/utf-8", "/fp:strict"],
        ))
    commands = {"build_ext": build_ext}

setup(ext_modules=extensions, cmdclass=commands)
