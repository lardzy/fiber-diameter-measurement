"""PowerShell launch, publication and UI failure contracts without real Excel."""
from concurrent.futures import Future
import base64
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from fdm.live_experiment import excel_template as module


def bootstrap(command):
    return base64.b64decode(command[command.index("-EncodedCommand") + 1]).decode("utf-16le")


def variables(command):
    text = bootstrap(command)
    return {name: Path(next(line for line in text.splitlines() if line.startswith(f"${name} = "))
                       .split(" = ", 1)[1][1:-1].replace("''", "'"))
            for name in ("requestPath", "resultPath", "outputPath", "environmentPath")}


@pytest.fixture
def excel_request(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32", version=sys.version))
    template = tmp_path / "中文 模板's.xlt"
    template.write_bytes(b"original template")
    return {"operation": "inspect", "template": str(template), "profile": {}}


def install_worker(monkeypatch, *, result=None, returncode=0, stderr=b"", callback=None, owner=False):
    commands, processes = [], []

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.returncode = returncode
            self.killed = False
            self.paths = variables(command)
            self.excel_request = json.loads(self.paths["requestPath"].read_text(encoding="utf-8-sig"))
            self.paths["environmentPath"].write_text(json.dumps({"powershell_version": "5.1.test",
                "effective_policy": "Bypass", "original_console_codepage": 936, "output_codepage": 65001,
                "policies": [{"scope": "MachinePolicy", "policy": "Undefined"}]}), encoding="utf-8-sig")
            Path(str(self.paths["resultPath"]) + ".progress.json").write_text('{"stage":"写入试验记录"}', encoding="utf-8")
            if owner:
                Path(str(self.paths["resultPath"]) + ".owner.json").write_text('{"id":123,"created":"456"}')
            if result is not None:
                self.paths["resultPath"].write_text(json.dumps(result), encoding="utf-8-sig")
            if callback:
                callback(self)
            processes.append(self)

        def communicate(self, timeout=None):
            return b"", stderr

        def kill(self):
            self.killed = True
            self.returncode = -9

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    return commands, processes


@pytest.mark.parametrize("cleanup", [False, True])
def test_launcher_is_process_scoped_and_calls_file_without_evaluating_contents(tmp_path, cleanup):
    command = module._powershell_command(tmp_path / "测试's.ps1", tmp_path / "request.json", tmp_path / "result.json", cleanup=cleanup)
    assert command[command.index("-ExecutionPolicy") + 1] == "Bypass"
    assert {"-NoProfile", "-NonInteractive", "-STA", "-EncodedCommand"} <= set(command)
    script = bootstrap(command)
    assert "测试''s.ps1" in script
    assert "& $worker -RequestPath $requestPath -ResultPath $resultPath" in script
    assert (" -Cleanup" in script) is cleanup
    assert "Set-ExecutionPolicy" not in script
    assert "Invoke-Expression" not in script and "Get-Content" not in script
    assert script.index("Get-ExecutionPolicy -List") < script.index("& $worker")
    assert script.index("[Console]::OutputEncoding =") < script.index("& $worker")
    assert "phase='script_start'" in script
    assert variables(command)["outputPath"].name == ("result.json.cleanup.json" if cleanup else "result.json")


def test_worker_success_returns_policy_encoding_version_and_stage(excel_request, monkeypatch):
    install_worker(monkeypatch, result={"ok": True, "methods": []})
    result = module.run_excel_request(excel_request)
    diagnostic = result["diagnostics"]
    assert diagnostic["environment"]["effective_policy"] == "Bypass"
    assert diagnostic["environment"]["powershell_version"] == "5.1.test"
    assert diagnostic["stage"] == "写入试验记录"
    assert diagnostic["app_version"]
    assert diagnostic["exit_code"] == 0


@pytest.mark.parametrize("structured", [False, True])
def test_policy_block_has_readable_message_and_lossless_chinese_diagnostics(excel_request, monkeypatch, structured):
    error = "无法加载文件，系统禁止运行脚本。SecurityError PSSecurityException about_Execution_Policies"
    result = {"ok": False, "error": error, "error_type": "System.Management.Automation.PSSecurityException"} if structured else None
    def group_policy(process):
        path = process.paths["environmentPath"]
        environment = json.loads(path.read_text(encoding="utf-8-sig"))
        environment.update(effective_policy="AllSigned", policies=[
            {"scope": "MachinePolicy", "policy": "AllSigned"}, {"scope": "Process", "policy": "Bypass"}])
        path.write_text(json.dumps(environment), encoding="utf-8-sig")
    install_worker(monkeypatch, result=result, returncode=1, stderr=error.encode("gbk"), callback=group_policy)
    with pytest.raises(module.ExcelTemplateError) as caught:
        module.run_excel_request(excel_request)
    assert caught.value.code == "execution_policy"
    assert "系统执行策略" in str(caught.value)
    assert caught.value.diagnostics["stderr"] == error
    assert caught.value.diagnostics["stderr_encoding"] == "cp936"
    assert caught.value.diagnostics["environment"]["effective_policy"] == "AllSigned"
    assert "\ufffd" not in str(caught.value.diagnostics)


@pytest.mark.parametrize("result,code", [(None, "missing_result"),
    ({"ok": False, "code": "invalid_request", "error": "invalid JSON"}, "invalid_request"),
    ({"ok": False, "error": "80040154 REGDB_E_CLASSNOTREG"}, "excel_unavailable")])
def test_worker_failure_classification(excel_request, monkeypatch, result, code):
    install_worker(monkeypatch, result=result, returncode=1)
    with pytest.raises(module.ExcelTemplateError) as caught:
        module.run_excel_request(excel_request)
    assert caught.value.code == code


def test_launch_error_is_diagnostic_without_cleanup(excel_request, monkeypatch):
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("powershell.exe")))
    with pytest.raises(module.ExcelTemplateError) as caught:
        module.run_excel_request(excel_request)
    assert caught.value.code == "launch_failed"
    assert "powershell.exe" in caught.value.diagnostics["launch_error"]


@pytest.mark.parametrize("reason", ["timeout", "cancelled"])
@pytest.mark.parametrize("cleanup_failure", ["nonzero", "timeout", "launch"])
def test_cleanup_failure_never_masks_timeout_or_cancel(excel_request, monkeypatch, reason, cleanup_failure):
    event = threading.Event()
    commands, processes = install_worker(monkeypatch, owner=True,
        callback=lambda process: event.set() if reason == "cancelled" else None)
    cleanup_commands = []

    def cleanup(command, **kwargs):
        cleanup_commands.append(command)
        if cleanup_failure == "timeout":
            raise subprocess.TimeoutExpired(command, 15)
        if cleanup_failure == "launch":
            raise OSError("cleanup unavailable")
        return SimpleNamespace(returncode=1, stderr=b"SecurityError")

    monkeypatch.setattr(module.subprocess, "run", cleanup)
    expected = module.ExcelTemplateTimeout if reason == "timeout" else module.ExcelTemplateCancelled
    with pytest.raises(expected) as caught:
        module.run_excel_request(excel_request, timeout=0 if reason == "timeout" else 90, cancel_event=event)
    assert caught.value.code == reason
    assert caught.value.diagnostics["cleanup"]["ok"] is False
    assert processes[0].killed
    assert len(cleanup_commands) == 1
    assert commands[0][:-1] == cleanup_commands[0][:-1]
    assert ("超时" in str(caught.value)) is (reason == "timeout")
    assert "写入试验记录" in caught.value.diagnostics["stage"]


def test_unexpected_worker_exit_cleans_only_via_ownership_guard(excel_request, monkeypatch):
    install_worker(monkeypatch, returncode=2, owner=True)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or SimpleNamespace(returncode=0, stderr=b""))
    with pytest.raises(module.ExcelTemplateError):
        module.run_excel_request(excel_request)
    assert len(calls) == 1 and " -Cleanup" in bootstrap(calls[0])


@pytest.mark.parametrize("handed_off", [False, True])
@pytest.mark.parametrize("cleanup_ok", [False, True])
def test_success_with_excel_close_error_keeps_result_and_recovers_only_owned_instance(excel_request, monkeypatch, handed_off, cleanup_ok):
    def close_failure(process):
        Path(str(process.paths["resultPath"]) + ".close.json").write_text(json.dumps(
            {"ok": False, "handed_off": handed_off, "errors": ["Workbook.Close: RPC failed"]}), encoding="utf-8")
    install_worker(monkeypatch, result={"ok": True, "methods": [], "handed_off": handed_off},
                   owner=True, callback=close_failure)
    calls = []
    def cleanup(command, **kwargs):
        calls.append(command)
        variables(command)["outputPath"].write_text(json.dumps({"ok": cleanup_ok}), encoding="utf-8")
        return SimpleNamespace(returncode=0 if cleanup_ok else 1, stderr=b"")
    monkeypatch.setattr(module.subprocess, "run", cleanup)
    result = module.run_excel_request(excel_request)
    assert result["ok"] is True and result["methods"] == []
    assert result["diagnostics"]["excel_close"]["errors"] == ["Workbook.Close: RPC failed"]
    assert bool(calls) is (not handed_off)
    if not handed_off:
        assert result["diagnostics"]["cleanup"]["ok"] is cleanup_ok
        assert result["warnings"]


@pytest.mark.parametrize("failure", ["worker", "publish", "cancel", "missing"])
@pytest.mark.parametrize("already_exists", [False, True])
def test_failed_export_preserves_existing_report_and_removes_staging(excel_request, monkeypatch, tmp_path, failure, already_exists):
    destination = tmp_path / "已成功报告.xls"
    if already_exists:
        destination.write_bytes(b"previous successful report")
    event = threading.Event()
    excel_request.update(operation="export", destination=str(destination), show=False)

    def write_staged(process):
        assert process.excel_request["show"] is False
        assert Path(process.excel_request["destination"]) != destination
        assert destination.read_bytes() == b"previous successful report" if already_exists else not destination.exists()
        if failure != "missing":
            Path(process.excel_request["destination"]).write_bytes(b"new workbook")

    result = {"ok": False, "error": "save failed"} if failure == "worker" else {"ok": True}
    commands, processes = install_worker(monkeypatch, result=result,
        returncode=1 if failure == "worker" else 0, callback=write_staged)
    if failure == "publish":
        monkeypatch.setattr(module, "atomic_replace_file", lambda *a: (_ for _ in ()).throw(PermissionError("file is open")))
    if failure == "cancel":
        original = module._run_excel_worker
        def cancel_after_completed(*args, **kwargs):
            value = original(*args, **kwargs)
            event.set()
            return value
        monkeypatch.setattr(module, "_run_excel_worker", cancel_after_completed)
    with pytest.raises(module.ExcelTemplateError):
        module.run_excel_request(excel_request, cancel_event=event)
    assert destination.read_bytes() == b"previous successful report" if already_exists else not destination.exists()
    assert not Path(processes[0].excel_request["destination"]).exists()
    assert Path(excel_request["template"]).read_bytes() == b"original template"


def test_successful_publication_then_open_failure_keeps_success(excel_request, monkeypatch, tmp_path):
    destination = tmp_path / "报告.xls"
    destination.write_bytes(b"old")
    excel_request.update(operation="export", destination=str(destination), show=True)
    install_worker(monkeypatch, result={"ok": True, "status": "ready", "method": "test", "warnings": []},
                   callback=lambda process: Path(process.excel_request["destination"]).write_bytes(b"complete report"))
    opened = []
    def cannot_open(path):
        opened.append(path)
        assert destination.read_bytes() == b"complete report"
        raise OSError("no registered viewer")
    monkeypatch.setattr(module.os, "startfile", cannot_open, raising=False)
    result = module.run_excel_request(excel_request)
    assert result["ok"] and result["destination"] == str(destination)
    assert result["diagnostics"]["published"] is True
    assert opened == [str(destination)]
    assert "报告已保存" in result["warnings"][0]


def test_export_cannot_replace_template_even_through_hardlink(excel_request, monkeypatch, tmp_path):
    alias = tmp_path / "alias.xls"
    alias.hardlink_to(excel_request["template"])
    excel_request.update(operation="export", destination=str(alias))
    with pytest.raises(module.ExcelTemplateError) as caught:
        module.run_excel_request(excel_request)
    assert caught.value.code == "template_overwrite"


def test_diagnostics_panel_copy_export_and_failed_retry_preserve_last_report(tmp_path, monkeypatch, desktop_application):
    from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
    from fdm.ui import trial_report_panel as ui

    monkeypatch.setattr(ui, "template_preferences", lambda: {})
    monkeypatch.setattr(ui, "working_template", lambda **kwargs: tmp_path / "sheet.xlt")
    window = QMainWindow()
    central = QWidget(window)
    window.setCentralWidget(central)
    window.report_layout = QVBoxLayout(central)
    window._guard = lambda fn: fn()
    panel = ui.TrialReportPanel(window)
    try:
        panel.future = Future()
        success = {"ok": True, "destination": str(tmp_path / "成功.xls"), "method": "方法", "status": "ready",
                   "diagnostics": {"stage": "报告已保存", "environment": {"effective_policy": "Bypass"}}}
        panel.future.set_result(success)
        panel.poll()
        previous_label = panel.last_report_label.text()
        panel.future = Future()
        diagnostic = {"stage": "启动模板组件", "environment": {"effective_policy": "AllSigned"}, "app_version": "test"}
        panel.future.set_exception(module.ExcelTemplateError("组织策略阻止运行", code="execution_policy", diagnostics=diagnostic))
        panel.poll()
        assert panel.last_report is success
        assert panel.last_report_label.text() == previous_label
        assert panel.status.text() == "组织策略阻止运行"
        assert panel.diagnostics_widget.isHidden()
        panel.diagnostics_toggle.click()
        assert not panel.diagnostics_widget.isHidden()
        panel.copy_diagnostics_button.click()
        assert json.loads(desktop_application.clipboard().text())["environment"]["effective_policy"] == "AllSigned"
        target = tmp_path / "诊断.json"
        monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *a: (str(target), ""))
        panel.export_diagnostics_button.click()
        assert json.loads(target.read_text(encoding="utf-8"))["message"] == "组织策略阻止运行"
        panel.future = Future()
        panel.cancel_button.setEnabled(True)
        panel.cancel_button.click()
        assert panel.cancel_event.is_set() and not panel.cancel_button.isEnabled()
        panel.future.set_exception(module.ExcelTemplateCancelled("任务已取消", code="cancelled"))
        panel.poll()
        assert panel.last_report is success and panel.status.text() == "任务已取消"
    finally:
        panel.close()
        window.close()
        window.deleteLater()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell integration; never launches Excel")
@pytest.mark.parametrize("inherited_policy", [None, "Restricted"])
def test_windows_invalid_json_and_cleanup_without_owner_are_structured(tmp_path, monkeypatch, inherited_policy):
    if inherited_policy is None:
        monkeypatch.delenv("PSExecutionPolicyPreference", raising=False)
    else:
        monkeypatch.setenv("PSExecutionPolicyPreference", inherited_policy)
    script = Path(module.__file__).with_suffix(".ps1")
    request_path, result_path = tmp_path / "中文 请求.json", tmp_path / "结果.json"
    request_path.write_text("invalid json", encoding="utf-8")
    completed = subprocess.run(module._powershell_command(script, request_path, result_path),
                               capture_output=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    # Corporate policy refusal is a valid environment limitation, not a request
    # parsing pass. Run this contract in the controlled Windows validation VM.
    assert completed.returncode != 0 and result["code"] == "invalid_request"
    environment = json.loads(Path(str(result_path) + ".environment.json").read_text(encoding="utf-8-sig"))
    assert environment["effective_policy"] == "Bypass" and environment["output_codepage"] == 65001
    cleanup = subprocess.run(module._powershell_command(script, request_path, result_path, cleanup=True),
                             capture_output=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    assert cleanup.returncode == 0
    assert json.loads(Path(str(result_path) + ".cleanup.json").read_text(encoding="utf-8-sig"))["status"] == "owner_absent"
    assert json.loads(result_path.read_text(encoding="utf-8-sig")) == result
    # A live process with the wrong image name or creation time must survive.
    import os
    Path(str(result_path) + ".owner.json").write_text(json.dumps({"id": os.getpid(), "created": "0", "handed_off": False}), encoding="utf-8")
    mismatch = subprocess.run(module._powershell_command(script, request_path, result_path, cleanup=True),
                              capture_output=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    assert mismatch.returncode == 0
    assert json.loads(Path(str(result_path) + ".cleanup.json").read_text(encoding="utf-8-sig"))["status"] == "not_owned_or_already_closed"
