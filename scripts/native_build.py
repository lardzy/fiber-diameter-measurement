"""Prepare the source-tree extension without changing installed runtime packages."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib


# Never load a .pyd into the packaging process: Windows keeps imported DLLs
# locked, and CPython cannot reliably unload/reload a rebuilt extension.
_PROBE = r"""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "src"))
try:
    import fdm._native as native
    if Path(native.__file__).resolve() != root / "src/fdm/_native/__init__.py":
        raise RuntimeError("native wrapper was imported from another checkout")
    directory = Path(sys.argv[2]).resolve()
    native.__path__ = [str(directory)]
    from fdm.services.native_kernels_self_check import native_probe_passed, run_native_kernels_self_check
    report = run_native_kernels_self_check(validate_source=True)
    module = native.require_native(validate_source=True)
    if Path(module.__file__).resolve().parent != directory:
        raise RuntimeError("native binary was imported from another checkout")
    from fdm.services.cu_live_self_check import run_cu_live_self_check, cu_live_probe_passed
    live = run_cu_live_self_check()
    if sys.platform == "win32":
        from fdm._native import _microview
        if Path(_microview.__file__).resolve().parent != directory:
            raise RuntimeError("capture binary was imported from another checkout")
        capture_info = dict(_microview.build_info())
        kernel_info = dict(module.build_info())
        from fdm.services.microview_transport import bridge_build_matches
        if not bridge_build_matches(capture_info, kernel_info):
            raise RuntimeError("capture API, image-info ABI, or build mismatch")
    passed = native_probe_passed(report) and cu_live_probe_passed(live)
    print(json.dumps({"ok": passed, "native_kernels": report, "cu_live_measurement": live}))
    sys.exit(0 if passed else 1)
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
    sys.exit(1)
"""

_CHECK_BUILD_TOOLS = r"""
from importlib.metadata import version
import sys
from packaging.requirements import Requirement

for text in sys.argv[1:]:
    requirement = Requirement(text)
    if requirement.marker is None or requirement.marker.evaluate():
        if version(requirement.name) not in requirement.specifier:
            raise RuntimeError(f"build requirement not satisfied: {text}")
"""


def check_native_build_dependencies(
    *, root: Path | None = None, extension_dir: Path | None = None,
) -> list[str]:
    """Check ABI, source fingerprint and actual computations in a fresh process."""
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    extension_dir = extension_dir or root / "src/fdm/_native"
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", _PROBE, str(root), str(extension_dir)],
            cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"native kernel build probe failed: {exc}"]
    try:
        report = json.loads(completed.stdout)
    except ValueError:
        return [f"native kernel build probe returned invalid JSON (exit {completed.returncode}); "
                f"stdout={completed.stdout[-1000:]}; stderr={completed.stderr[-2000:]}"]
    if completed.returncode == 0 and isinstance(report, dict) and report.get("ok") is True:
        return []
    detail = (report.get("error") or report) if isinstance(report, dict) else report
    return [f"native kernel build probe failed (exit {completed.returncode}): {detail}; "
            f"{completed.stderr[-2000:]}".rstrip()]


def _build_python(root: Path, staging: Path, env: dict[str, str], log) -> str:
    """Reuse installed build tools, or provision them in a disposable same-ABI venv."""
    with (root / "pyproject.toml").open("rb") as stream:
        requirements = tomllib.load(stream)["build-system"]["requires"]
    if not isinstance(requirements, list) or not requirements or not all(isinstance(r, str) for r in requirements):
        raise ValueError("pyproject.toml has invalid build-system.requires")
    available = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-c", _CHECK_BUILD_TOOLS, *requirements],
        cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=False,
    )
    if available.returncode == 0:
        return sys.executable
    print("Preparing isolated native build dependencies...")
    environment = staging / "build-env"
    subprocess.run(
        [sys.executable, "-m", "venv", str(environment)],
        cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True,
    )
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", *requirements],
        cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True,
    )
    return str(python)


def _replace_extension(source: Path, destination: Path) -> None:
    """Publish a complete file; a locked Windows DLL leaves the old file intact."""
    descriptor, temporary = tempfile.mkstemp(prefix=".kernel-", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _publish_extensions(candidates: dict[Path, Path], *, root: Path, backup_dir: Path) -> None:
    """Publish the verified set, rolling every changed file back on any failure.

    Each replacement is atomic. Both extension hashes are checked as one set
    before and after publication; the build never accepts a mixed generation.
    """
    backup_dir.mkdir()
    previous: dict[Path, Path | None] = {}
    for destination in candidates:
        previous[destination] = None
        if destination.is_file():
            backup = backup_dir / destination.name
            shutil.copy2(destination, backup)
            previous[destination] = backup
    published: list[Path] = []
    try:
        for destination, candidate in candidates.items():
            _replace_extension(candidate, destination)
            published.append(destination)
        errors = check_native_build_dependencies(root=root)
        if errors:
            raise RuntimeError("installed extensions failed validation: " + "; ".join(errors))
    except BaseException as failure:
        rollback_errors = []
        for destination in reversed(published):
            try:
                backup = previous[destination]
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    _replace_extension(backup, destination)
            except OSError as exc:
                rollback_errors.append(f"{destination}: {exc}")
        if rollback_errors:
            # A concurrently loaded Windows DLL can also prevent rollback.
            # Keep the known previous files beyond TemporaryDirectory cleanup.
            recovery = Path(tempfile.mkdtemp(prefix="recovery-", dir=root / "build/native-kernels"))
            for backup in previous.values():
                if backup is not None:
                    shutil.copy2(backup, recovery / backup.name)
            raise RuntimeError(f"{failure}; extension rollback failed: {'; '.join(rollback_errors)}; previous binaries retained at {recovery}") from failure
        raise


def _extension_names() -> tuple[str, ...]:
    return ("_kernels", "_microview") if sys.platform == "win32" else ("_kernels",)


def ensure_native_kernels(root: Path) -> list[str]:
    """Reuse a valid extension, otherwise rebuild once and require a fresh probe."""
    root = root.resolve()
    errors = check_native_build_dependencies(root=root)
    if not errors:
        print("Native kernels are current; skipping compilation.")
        return []
    print("Native kernels need rebuilding:\n  " + "\n  ".join(errors))
    log_path = None
    try:
        if not (root / "setup.py").is_file():
            raise FileNotFoundError("setup.py is missing; use a complete source checkout")
        build_root = root / "build/native-kernels"
        build_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="rebuild-", suffix=".log", dir=build_root, delete=False) as log:
            log_path = Path(log.name)
            print(f"Native build log: {log_path}")
            with tempfile.TemporaryDirectory(prefix="work-", dir=build_root) as temporary:
                staging = Path(temporary)
                env = os.environ.copy()
                env["FDM_BUILD_NATIVE"] = "1"
                env["PYTHONUTF8"] = "1"
                python = _build_python(root, staging, env, log)
                library = staging / "lib"
                subprocess.run(
                    [python, "-I", "-X", "utf8", str(root / "setup.py"), "build_ext", "--force",
                     "--build-lib", str(library), "--build-temp", str(staging / "objects")],
                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True,
                )
                suffix = sysconfig.get_config_var("EXT_SUFFIX")
                names = _extension_names()
                candidates = {}
                for name in names:
                    filename = name + suffix
                    candidate = library / "fdm/_native" / filename
                    if not candidate.is_file():
                        raise RuntimeError(f"compiler did not produce the current Python extension: {filename}")
                    candidates[root / "src/fdm/_native" / filename] = candidate
                errors = check_native_build_dependencies(root=root, extension_dir=library / "fdm/_native")
                if errors:
                    raise RuntimeError("rebuilt extensions failed validation: " + "; ".join(errors))
                _publish_extensions(candidates, root=root, backup_dir=staging / "previous")
        print("Native kernels rebuilt and verified; continuing packaging.")
        return []
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, RuntimeError) as exc:
        return [
            f"Native extension rebuild failed: {exc}",
            f"Build log: {log_path}" if log_path else "No native build log was created.",
            "Windows builds require MSVC C++ Build Tools and the Windows SDK. "
            "Close applications using this checkout if the extension is locked. "
            "Missing Python build tools are installed in a temporary environment; "
            "its package installation requires access to the configured package index or cache.",
        ]
