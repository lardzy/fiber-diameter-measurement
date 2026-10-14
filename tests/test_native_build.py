from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import native_build


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project with spaces"
    (root / "src/fdm/_native").mkdir(parents=True)
    (root / "setup.py").write_text("# build fixture\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=68", "wheel", "pybind11==3.0.4"]\n',
        encoding="utf-8",
    )
    return root


def extension(root):
    return root / "src/fdm/_native" / ("_kernels" + sysconfig.get_config_var("EXT_SUFFIX"))


def fake_compiler(command, **kwargs):
    library = Path(command[command.index("--build-lib") + 1])
    destination = library / "fdm/_native" / extension(Path()).name
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"new extension")
    kwargs["stdout"].write(b"compiler output\n")
    return subprocess.CompletedProcess(command, 0)


def test_valid_binary_skips_compiler_and_dependency_install(project, monkeypatch):
    probe = Mock(return_value=[])
    run = Mock()
    monkeypatch.setattr(native_build, "check_native_build_dependencies", probe)
    monkeypatch.setattr(native_build.subprocess, "run", run)
    assert native_build.ensure_native_kernels(project) == []
    probe.assert_called_once_with(root=project)
    run.assert_not_called()
    assert not (project / "build").exists()


@pytest.mark.parametrize("reason", ["extension missing", "stale source", "wrong ABI", "bad DLL"])
def test_rebuild_validates_before_and_after_atomic_publish(project, monkeypatch, reason):
    egg_info = project / "src/local.egg-info"
    egg_info.mkdir()
    metadata = egg_info / "PKG-INFO"
    metadata.write_bytes(b"user metadata\r\n")
    old_abi = extension(project).with_name("_kernels.cp999-win_amd64.pyd")
    old_abi.write_bytes(b"unrelated old ABI")
    if reason != "extension missing":
        extension(project).write_bytes(b"old extension")
    probe = Mock(side_effect=[[reason], [], []])
    run = Mock(side_effect=fake_compiler)
    monkeypatch.setattr(native_build, "check_native_build_dependencies", probe)
    monkeypatch.setattr(native_build, "_build_python", Mock(return_value=sys.executable))
    monkeypatch.setattr(native_build.subprocess, "run", run)
    monkeypatch.setenv("FDM_BUILD_NATIVE", "0")
    assert native_build.ensure_native_kernels(project) == []
    assert extension(project).read_bytes() == b"new extension"
    assert old_abi.read_bytes() == b"unrelated old ABI"
    assert metadata.read_bytes() == b"user metadata\r\n"
    assert run.call_args.args[0][0] == sys.executable
    assert "--force" in run.call_args.args[0]
    assert run.call_args.kwargs["env"]["FDM_BUILD_NATIVE"] == "1"
    assert os.environ["FDM_BUILD_NATIVE"] == "0"
    assert probe.call_count == 3
    assert "extension_dir" in probe.call_args_list[1].kwargs
    assert probe.call_args_list[2].kwargs == {"root": project}


@pytest.mark.parametrize("failure", [
    subprocess.CalledProcessError(1, ["compiler"]),
    subprocess.TimeoutExpired(["compiler"], 600),
    OSError("compiler missing"),
])
def test_compile_failure_preserves_binary_and_log(project, monkeypatch, failure):
    extension(project).write_bytes(b"old extension")
    probe = Mock(return_value=["stale"])
    monkeypatch.setattr(native_build, "check_native_build_dependencies", probe)
    monkeypatch.setattr(native_build, "_build_python", Mock(return_value=sys.executable))
    monkeypatch.setattr(native_build.subprocess, "run", Mock(side_effect=failure))
    errors = native_build.ensure_native_kernels(project)
    assert errors and "rebuild failed" in errors[0]
    assert "MSVC" in errors[-1]
    assert extension(project).read_bytes() == b"old extension"
    assert list((project / "build/native-kernels").glob("*.log"))
    assert probe.call_count == 1


@pytest.mark.parametrize("post_install", [False, True])
def test_failed_recheck_cannot_publish_unverified_binary(project, monkeypatch, post_install):
    extension(project).write_bytes(b"old extension")
    results = [["stale"], [], ["installed probe failed"]] if post_install else [["stale"], ["bad result"]]
    monkeypatch.setattr(native_build, "check_native_build_dependencies", Mock(side_effect=results))
    monkeypatch.setattr(native_build, "_build_python", Mock(return_value=sys.executable))
    monkeypatch.setattr(native_build.subprocess, "run", Mock(side_effect=fake_compiler))
    assert native_build.ensure_native_kernels(project)
    assert extension(project).read_bytes() == b"old extension"


def test_locked_destination_stops_without_removing_old_binary(project, monkeypatch):
    extension(project).write_bytes(b"locked old extension")
    monkeypatch.setattr(native_build, "check_native_build_dependencies", Mock(side_effect=[["stale"], []]))
    monkeypatch.setattr(native_build, "_build_python", Mock(return_value=sys.executable))
    monkeypatch.setattr(native_build.subprocess, "run", Mock(side_effect=fake_compiler))
    monkeypatch.setattr(native_build.os, "replace", Mock(side_effect=PermissionError("DLL locked")))
    errors = native_build.ensure_native_kernels(project)
    assert "DLL locked" in errors[0]
    assert extension(project).read_bytes() == b"locked old extension"
    assert not list(extension(project).parent.glob(".kernel-*.tmp"))


def test_build_tools_reuse_current_python_or_install_only_in_temporary_venv(project, monkeypatch, tmp_path):
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(native_build.subprocess, "run", run)
    env = {"FDM_BUILD_NATIVE": "1"}
    with (tmp_path / "build.log").open("wb") as log:
        assert native_build._build_python(project, tmp_path, env, log) == sys.executable
        assert run.call_count == 1
        assert "pybind11==3.0.4" in run.call_args.args[0]
        run.reset_mock()
        run.side_effect = [subprocess.CompletedProcess([], 1), subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 0)]
        python = native_build._build_python(project, tmp_path, env, log)
    assert Path(python).is_relative_to(tmp_path / "build-env")
    assert run.call_args_list[1].args[0][:3] == [sys.executable, "-m", "venv"]
    install = run.call_args_list[2].args[0]
    assert install[0] == python
    assert install[-3:] == ["setuptools>=68", "wheel", "pybind11==3.0.4"]
    assert not any("-e" in call.args[0] for call in run.call_args_list)


def test_probe_always_uses_fresh_current_interpreter(project, monkeypatch):
    run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps({"ok": True}), stderr=""))
    monkeypatch.setattr(native_build.subprocess, "run", run)
    assert native_build.check_native_build_dependencies(root=project) == []
    command = run.call_args.args[0]
    assert command[:5] == [sys.executable, "-I", "-X", "utf8", "-c"]
    assert command[-2:] == [str(project), str(project / "src/fdm/_native")]


@pytest.mark.parametrize("result", [
    subprocess.CompletedProcess([], 1, stdout='{"ok": false, "error": "bad DLL"}', stderr=""),
    subprocess.CompletedProcess([], 0, stdout='{"ok": false}', stderr=""),
    subprocess.CompletedProcess([], 0, stdout='not JSON', stderr="load error"),
    subprocess.CompletedProcess([], -11, stdout='', stderr="native crash"),
])
def test_failed_probe_never_accepts_nonzero_false_or_invalid_output(project, monkeypatch, result):
    monkeypatch.setattr(native_build.subprocess, "run", Mock(return_value=result))
    assert native_build.check_native_build_dependencies(root=project)


def test_probe_does_not_accept_an_extension_from_another_checkout(project):
    # This source root has no Python wrapper, while the test environment has an
    # editable FDM installation. Its otherwise valid extension must not pass.
    assert native_build.check_native_build_dependencies(root=project)
