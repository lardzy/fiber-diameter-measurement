"""Full MainWindow / desktop Excel template acceptance, for source and frozen apps.

This explicitly requested probe uses an isolated profile and disposable template
copies. It never starts a camera. File-picker answers and exit-discard confirmation
are automated; page navigation, report selection, coordinator and Excel are real.
`--require-installed` rejects source/portable runs instead of calling them installed
validation. The same entry point is included in the frozen app by its static import.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from uuid import uuid4


CUSTOM_REPORTS = ("特种毛(报出)", "麻棉(报出)", "分组加权统计", "织物纤维含量报告")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def view_differences(before, after):
    """Compare literal template presentation, independently of worker metadata."""
    differences = []
    if before.get("active_sheet") != after.get("active_sheet"):
        differences.append("active worksheet changed")
    left, right = before.get("sheets", []), after.get("sheets", [])
    if [item["name"] for item in left] != [item["name"] for item in right]:
        differences.append("worksheet names/order changed")
    right_by_name = {item["name"]: item for item in right}
    for item in left:
        other = right_by_name.get(item["name"])
        if other is None:
            continue
        for field in ("visible", "shapes", "sample"):
            if item.get(field) != other.get(field):
                differences.append(f"{item['name']}: {field} changed")
    return differences


def cleanup_failures(report):
    """A successful functional run still fails if its owned resources remain."""
    failures = []
    if report.get("close_error"):
        failures.append("Final synchronization failed: " + str(report["close_error"]))
    for index, item in enumerate(report.get("cleanup", []), 1):
        if item.get("error"):
            failures.append(f"Excel cleanup {index} failed: {item['error']}")
        elif not isinstance(item.get("remaining"), int) or item["remaining"] != 0:
            failures.append(f"Excel cleanup {index} did not confirm zero remaining workbooks: {item.get('remaining')!r}")
    return failures


def application_identity():
    frozen = bool(getattr(sys, "frozen", False))
    exe = Path(sys.executable).resolve()
    result = {"frozen": frozen, "executable": str(exe), "executable_sha256": sha256(exe),
              "installed_registry_match": False, "registry_matches": [], "manifest": None}
    if frozen:
        manifest = exe.parent / "release-manifest.json"
        if manifest.is_file():
            result["manifest"] = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if sys.platform != "win32" or not frozen:
        return result
    import winreg
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                root = winreg.OpenKey(hive, r"Software\Microsoft\Windows\CurrentVersion\Uninstall",
                                     0, winreg.KEY_READ | view)
            except OSError:
                continue
            with root:
                for index in range(winreg.QueryInfoKey(root)[0]):
                    try:
                        with winreg.OpenKey(root, winreg.EnumKey(root, index)) as entry:
                            location = winreg.QueryValueEx(entry, "InstallLocation")[0]
                            if not location or Path(location).resolve() != exe.parent:
                                continue
                            result["registry_matches"].append({"name": winreg.QueryValueEx(entry, "DisplayName")[0],
                                "location": str(location)})
                    except OSError:
                        continue
    result["installed_registry_match"] = bool(result["registry_matches"])
    return result


# No GetActiveObject / unqualified ActiveWorkbook: every live read checks the
# exact independent Excel PID, its creation time and the recorded workbook name.
PROBE_PS = r'''param([string]$RequestPath)
$ErrorActionPreference='Stop'
Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class FdmTemplateFlowWindow {
 public delegate bool Visit(IntPtr h, IntPtr p);
 [DllImport("user32.dll")] static extern bool EnumWindows(Visit v, IntPtr p);
 [DllImport("user32.dll")] static extern bool EnumChildWindows(IntPtr h, Visit v, IntPtr p);
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint p);
 [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern int GetClassName(IntPtr h,System.Text.StringBuilder s,int n);
 [DllImport("oleacc.dll")] static extern int AccessibleObjectFromWindow(IntPtr h,uint id,ref Guid g,[MarshalAs(UnmanagedType.Interface)] out object obj);
 public static object Window(uint pid) { object found=null;
  Visit children=(h,p)=> {var s=new System.Text.StringBuilder(128);GetClassName(h,s,128);if(s.ToString()!="EXCEL7")return true;
   Guid id=new Guid("00020400-0000-0000-C000-000000000046");object o;
   if(AccessibleObjectFromWindow(h,0xfffffff0,ref id,out o)==0 && o!=null){found=o;return false;}return true;};
  EnumWindows((h,p)=>{uint actual;GetWindowThreadProcessId(h,out actual);if(actual==pid)EnumChildWindows(h,children,IntPtr.Zero);return found==null;},IntPtr.Zero);return found;
 }
}
'@
function Snapshot($book) {
 $items=@(); $active=[string]$book.ActiveSheet.Name
 foreach($sheet in $book.Worksheets) {
  $sample=@()
  if([string]$sheet.Name -in @('特种毛(报出)','麻棉(报出)','分组加权统计','织物纤维含量报告')) {
   foreach($address in @('A1','B2','D5','F8','G10','K20')) {
    $cell=$sheet.Range($address)
    $sample+=@{address=$address;formula=$cell.Formula;number_format=$cell.NumberFormat;
      font_name=$cell.Font.Name;font_size=$cell.Font.Size;font_color=$cell.Font.Color;
      bold=$cell.Font.Bold;fill=$cell.Interior.Color;merge=$cell.MergeArea.Address()}
   }
  }
  $items+=@{name=[string]$sheet.Name;visible=[int]$sheet.Visible;shapes=[int]$sheet.Shapes.Count;sample=$sample}
 }
 $result=@{name=[string]$book.Name;active_sheet=$active;sheets=$items;count=[int]$book.Worksheets.Count}
 foreach($sheet in $book.Worksheets) {
  if([string]$sheet.Name -eq '原始数据') {
   $result.first=$sheet.Range('D25').Value2;$result.second=$sheet.Range('D26').Value2
  }
 }
 return $result
}
function SetValidationName($book,[string]$name,[string]$value) {
 # BuiltinDocumentProperties.Item('Comments') is locale/COM-version dependent.
 # These unique, workbook-scoped constants belong to the probe and neither
 # overwrite report cells nor add pages/controls or disturb the active sheet.
 $definition='="'+$value.Replace('"','""')+'"'
 $created=$book.Names.Add($name,$definition)
 if($null -eq $created){throw ('Could not create validation name: '+$name)}
}
function ReadValidationName($book,[string]$name) {
 foreach($item in $book.Names) {
  if([string]$item.Name -eq $name) {
   $definition=[string]$item.RefersTo
   if($definition.StartsWith('="') -and $definition.EndsWith('"')) {
    return $definition.Substring(2,$definition.Length-3).Replace('""','"')
   }
   throw ('Unexpected validation name value: '+$name)
  }
 }
 return $null
}
$r=Get-Content -LiteralPath $RequestPath -Raw -Encoding UTF8|ConvertFrom-Json
function ProbeStage([string]$name) {
 @{utc=[DateTime]::UtcNow.ToString('o');stage=$name;operation=[string]$r.operation}|
  ConvertTo-Json -Compress|Add-Content -LiteralPath ($RequestPath+'.stages.jsonl') -Encoding UTF8
}
ProbeStage 'request'
$excel=$null;$book=$null;$owned=$false
if($r.operation -in @('baseline','variant','invalid')) {
 $before=@(Get-Process EXCEL -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
 ProbeStage 'creating independent Excel'
 $excel=New-Object -ComObject Excel.Application
 [uint32]$pidValue=0;[void][FdmTemplateFlowWindow]::GetWindowThreadProcessId([IntPtr]$excel.Hwnd,[ref]$pidValue)
 if(!$pidValue -or $before -contains [int]$pidValue){throw 'Baseline Excel is not independent'}
 $owned=$true
 @{id=$pidValue;created=(Get-Process -Id $pidValue).StartTime.ToUniversalTime().Ticks.ToString();
   validation_probe=$true;operation=[string]$r.operation;preexisting_excel_ids=$before}|
   ConvertTo-Json|Set-Content -LiteralPath ($RequestPath+'.owner.json') -Encoding UTF8
 try {
  $excel.DisplayAlerts=$false
  # Use the same Office policy as production, including the selected template's
  # own Workbook_Open/Activate events (the supplied template creates buttons).
  # FDM must not introduce an extra grouping macro on top of those events.
  $excel.AutomationSecurity=2
  ProbeStage 'opening source with normal template events'
  $book=$excel.Workbooks.Open([string]$r.source,[Type]::Missing,[Type]::Missing,[Type]::Missing,[Type]::Missing,
    [Type]::Missing,[Type]::Missing,[Type]::Missing,[Type]::Missing,$false)
  ProbeStage 'reading template presentation'
  $result=Snapshot $book
  if($r.operation -ne 'baseline') {
   # This only authors disposable test inputs. Production opening, the native
   # baseline, live updates and actual report export retain their normal events.
   ProbeStage 'preparing disposable fixture without save events'
   $excel.EnableEvents=$false;$excel.DisplayAlerts=$false
   if($r.operation -eq 'variant') {SetValidationName $book '_FDM_Template_Flow_Version' ([string]$r.marker)}
   if($r.operation -eq 'invalid') {$book.Worksheets.Item('原始数据').Name='验证缺少映射输入页'}
   $book.CheckCompatibility=$false
   ProbeStage 'saving disposable template fixture'
   $book.SaveAs([string]$r.destination,17)
   ProbeStage 'fixture saved'
  }
 } finally {
  ProbeStage 'closing independent probe workbook'
  if($book){$book.Close($false)}
  ProbeStage 'quitting independent probe Excel'
  if($owned){$excel.Quit()}
 }
} else {
 ProbeStage 'checking exact existing Excel identity'
 $process=Get-Process -Id ([int]$r.excel_pid)
 if($process.ProcessName -ne 'EXCEL' -or $process.StartTime.ToUniversalTime().Ticks.ToString() -ne [string]$r.excel_created){throw 'Owned Excel identity mismatch'}
 $window=[FdmTemplateFlowWindow]::Window([uint32]$r.excel_pid)
 if(!$window){throw 'Owned Excel window unavailable'}
 $excel=$window.Application
 if($r.operation -eq 'cleanup') {
  foreach($name in $r.names){$excel.Workbooks.Item([string]$name).Close($false)}
  $remaining=[int]$excel.Workbooks.Count
  if($remaining -eq 0){$excel.Quit()}
  $result=@{remaining=$remaining}
 } else {
  ProbeStage 'reading exact existing workbook'
  $book=$excel.Workbooks.Item([string]$r.name)
  if($r.operation -eq 'note') {SetValidationName $book '_FDM_Template_Flow_Note' 'manual report note retained'}
  $result=Snapshot $book
  $result.note=ReadValidationName $book '_FDM_Template_Flow_Note'
 }
}
ProbeStage 'complete'
$result|ConvertTo-Json -Depth 15|Set-Content -LiteralPath $r.result -Encoding UTF8
'''


# A timeout can leave Excel blocked in a hidden modal dialog even after Python
# has reaped PowerShell. Only the independently created probe instance has this
# owner record. Live/product workbook reads never create it and cannot invoke
# this cleanup. Validate identity again immediately before stopping the process.
PROBE_TIMEOUT_CLEANUP_PS = r'''param([string]$OwnerPath)
$ErrorActionPreference='Stop'
$owner=Get-Content -LiteralPath $OwnerPath -Raw -Encoding UTF8|ConvertFrom-Json
if(!$owner.validation_probe -or $owner.operation -notin @('baseline','variant','invalid') -or
   !$owner.id -or !$owner.created -or @($owner.preexisting_excel_ids) -contains [int]$owner.id) {
 throw 'Refusing cleanup: no independent validation-probe ownership proof'
}
$process=Get-Process -Id ([int]$owner.id) -ErrorAction SilentlyContinue
if(!$process) {@{cleaned=$true;already_exited=$true}|ConvertTo-Json -Compress;exit 0}
if($process.ProcessName -ne 'EXCEL' -or $process.StartTime.ToUniversalTime().Ticks.ToString() -ne [string]$owner.created) {
 throw 'Refusing cleanup: Excel PID or creation-time identity does not match'
}
Stop-Process -InputObject $process -Force -ErrorAction Stop
@{cleaned=$true;forced=$true;id=$owner.id;created=$owner.created}|ConvertTo-Json -Compress
'''


class ExcelProbe:
    def __init__(self, output):
        self.output, self.sequence = Path(output), 0
        self.output.mkdir(parents=True, exist_ok=True)
        self.script = self.output / "inspect-owned-workbook.ps1"
        self.script.write_text(PROBE_PS, encoding="utf-8-sig")

    def _cleanup_owned_timeout(self, request):
        owner = Path(str(request) + ".owner.json")
        if not owner.is_file():
            return {"cleaned": False, "reason": "No independently created probe owner record; no Excel process touched"}
        cleanup_script = self.output / "cleanup-owned-probe-timeout.ps1"
        cleanup_script.write_text(PROBE_TIMEOUT_CLEANUP_PS, encoding="utf-8-sig")
        try:
            completed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass", "-File", str(cleanup_script), "-OwnerPath", str(owner)],
                capture_output=True, timeout=20)
            if completed.returncode:
                return {"cleaned": False, "error": completed.stderr.decode(errors="replace")}
            return json.loads(completed.stdout.decode("utf-8-sig"))
        except Exception as exc:
            return {"cleaned": False, "error": str(exc)}

    def call(self, operation, **values):
        self.sequence += 1
        request = self.output / f"request-{self.sequence}.json"
        result = self.output / f"result-{self.sequence}.json"
        request.write_text(json.dumps(dict(operation=operation, result=str(result), **values), ensure_ascii=False), encoding="utf-8")
        try:
            completed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-STA",
                "-ExecutionPolicy", "Bypass", "-File", str(self.script), "-RequestPath", str(request)],
                capture_output=True, timeout=90)
        except subprocess.TimeoutExpired as exc:
            cleanup = self._cleanup_owned_timeout(request)
            timeout = {"operation": operation, "request": str(request), "seconds": exc.timeout,
                       "stages": str(request) + ".stages.jsonl", "cleanup": cleanup}
            (self.output / f"timeout-{self.sequence}.json").write_text(
                json.dumps(timeout, ensure_ascii=False, indent=2), encoding="utf-8")
            for stream in ("stdout", "stderr"):
                value = getattr(exc, stream, None)
                if value:
                    (self.output / f"timeout-{self.sequence}.{stream}.txt").write_bytes(
                        value if isinstance(value, bytes) else value.encode("utf-8"))
            raise RuntimeError(f"Excel probe timed out ({operation}); diagnostics: {self.output / f'timeout-{self.sequence}.json'}") from exc
        if completed.returncode:
            (self.output / f"error-{self.sequence}.txt").write_bytes(completed.stderr)
            raise RuntimeError(f"Excel probe failed ({operation}): {completed.stderr.decode(errors='replace')}")
        return json.loads(result.read_text(encoding="utf-8-sig"))


def _wait(app, predicate, *, timeout=90, description="condition"):
    from PySide6.QtTest import QTest
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        value = predicate()
        if value:
            return value
        QTest.qWait(10)
    raise AssertionError(f"Timed out: {description}")


def run_phase(root, phase):
    """One fresh process, same isolated profile; actual MainWindow route."""
    from unittest.mock import patch
    from PySide6.QtCore import QCoreApplication, QEvent, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton
    from fdm import runtime_logging, screenshot_settings, settings
    from fdm.live_experiment import MaterialSnapshot
    from fdm.version import __version__

    config = json.loads((root / "validation-owner.json").read_text(encoding="utf-8"))
    profile = root / "profile"
    phase_output = root / phase
    phase_output.mkdir()
    selected = root / "所选含量模板.xlt"
    report = {"passed": False, "phase": phase, "version": __version__, "application": application_identity(),
              "source_only": not bool(getattr(sys, "frozen", False)), "profile": str(profile),
              "checks": [], "workbooks": [], "modal_messages": [], "slot_errors": [],
              "hardware_started": False, "picker_answer_injected": True,
              "project_close_confirmation_auto_discard": True}
    probe = ExcelProbe(phase_output / "excel-probe")
    baseline = probe.call("baseline", source=str(selected))
    report["baseline"] = baseline
    books = {}
    manager = None
    window = None

    def check(condition, text):
        if not condition:
            raise AssertionError(text)
        report["checks"].append(text)

    def synced(session, minimum_generation=1):
        def ready():
            state = manager.status(session.id)
            if state["state"] in ("error", "unsupported", "stopped"):
                raise AssertionError(state)
            if state.get("connected") and state["state"] == "synced" and not state.get("pending") and state["generation"] >= minimum_generation:
                return state
        state = _wait(app, ready, description="real Excel synchronization")
        books[state["name"]] = state
        return state

    def read(state, operation="read"):
        value = probe.call(operation, excel_pid=state["excel_pid"], excel_created=state["excel_created"], name=state["name"])
        report["workbooks"].append(value)
        return value

    def pick(panel, path):
        # Exercise the connected QPushButton, not a bare panel or mocked manager.
        view.workflow_tabs.setCurrentIndex(view.workflow_tabs.count() - 1)
        app.processEvents()
        buttons = [b for b in view.findChildren(QPushButton) if b.text() == "选择模板"]
        check(len(buttons) == 1, "Main-window report has exactly one template picker")
        with patch.object(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(path), "")):
            QTest.mouseClick(buttons[0], Qt.MouseButton.LeftButton)

    with ExitStack() as stack:
        stack.enter_context(patch.object(settings, "settings_directory", lambda: profile))
        stack.enter_context(patch.object(settings, "settings_file_path", lambda: profile / "settings.json"))
        stack.enter_context(patch.object(screenshot_settings, "screenshot_settings_file_path", lambda: profile / "screenshot-settings.json"))
        stack.enter_context(patch.object(runtime_logging, "runtime_log_path", lambda: profile / "startup.log"))
        # MainWindow imports these accessors after profile isolation is installed.
        stack.enter_context(patch("fdm.services.capture.available_capture_backends", return_value=[]))
        stack.enter_context(patch("fdm.services.motion_control.serial", None))
        stack.enter_context(patch("sys.excepthook", lambda *error: report["slot_errors"].append("".join(traceback.format_exception(*error)))))
        def modal(parent, title, text, *args, **kwargs):
            report["modal_messages"].append({"title": title, "text": text})
            return QMessageBox.StandardButton.Cancel
        for name in ("warning", "critical", "information", "question"):
            stack.enter_context(patch.object(QMessageBox, name, modal))
        app = QApplication.instance() or QApplication([])
        from fdm.ui.main_window import MainWindow
        try:
            window = MainWindow()
            stack.enter_context(patch.object(window._screenshot_agent_client, "status",
                lambda **kwargs: SimpleNamespace(running=False, error="", result={})))
            stack.enter_context(patch.object(window._screenshot_agent_client, "send",
                lambda *args, **kwargs: SimpleNamespace(result={"running": False, "offline": True, "hotkeys": []})))
            window.show()
            window.resize(1366, 768)
            # Actual menu/action routing enters the real shared acquisition shell.
            window.acquisition_workspace_action.trigger()
            app.processEvents()
            hub = window._acquisition_workbench
            QTest.mouseClick(hub.navigation, Qt.MouseButton.LeftButton, pos=hub.navigation.tabRect(1).center())
            view = _wait(app, lambda: getattr(window, "_live_experiment_workspaces", {}).get("content"), description="content page")
            check(hub.navigation.currentIndex() == 1 and hub.pages.currentWidget() is view, "Actual acquisition navigation opens content trial")
            session, manager, panel = view.session, view.live_excel, view.report_panel
            initial = synced(session)
            if phase == "select":
                check(Path(initial["template_path"]).resolve() != (profile / "templates" / "sheet.xlt").resolve(), "Legacy personal copy does not shadow packaged default")
                canonical_default = (settings.runtime_directory() / "content-templates" / "sheet.xlt").resolve()
                initial_path = Path(initial["template_path"]).resolve()
                check(initial_path == canonical_default,
                      "First trial without a custom preference opens the canonical program runtime template")
                default_source_sha256 = sha256(initial_path)
                default_baseline = probe.call("baseline", source=str(initial_path))
                default_presentation = read(initial)
                report["default_open"] = {
                    "path": str(initial_path), "canonical_default_path": str(canonical_default),
                    "template_identity": initial.get("template_identity", {}),
                    "source_sha256": default_source_sha256, "state": initial,
                    "baseline": default_baseline, "presentation": default_presentation}
                check(initial.get("template_identity", {}).get("sha256", "").lower() == default_source_sha256,
                      "Initial workbook identity matches the current bytes of the program runtime template")
                check(not view_differences(default_baseline, default_presentation),
                      "Automatic default opening preserves all source pages, visibility, active sheet, shapes and sampled formatting/formulas")
                check(sha256(initial_path) == default_source_sha256,
                      "Default opening and independent inspection leave program runtime template bytes unchanged")
                session.set_materials([MaterialSnapshot("template-flow-wool", "绵羊毛", 1.31)])
                session.add_measurement(18.25, "template-flow-wool")
                manager.update(session)
                synced(session)
                pick(panel, selected)
                state = synced(session, initial["generation"] + 1)
                actual = read(state, "note")
                check(not view_differences(baseline, actual), "Selected template preserves all worksheets, visibility, active sheet, shapes and sampled styles/formulas")
                check(actual["first"] == 18.25, "Existing measurement reaches chosen template")
                check(actual["note"] == "manual report note retained", "Independent probe can write and read a validation-owned workbook note")
                check(state.get("template_identity", {}).get("sha256", "").lower() == sha256(selected), "Worker source hash matches selected file")
                # Use the real menu command. Recalculation must restore the
                # selected report page and every sheet's visibility, even when
                # the legacy grouping macro changes those properties internally.
                recalculate = view.live_excel_controls.recalculate_action
                check(recalculate.isEnabled(), "Explicit report recalculation is available in the trial menu")
                recalculate.trigger()
                recalculated_state = synced(session, state["generation"])
                recalculated = read(recalculated_state)
                report["explicit_recalculation"] = {
                    "state": recalculated_state, "presentation": recalculated}
                check(not view_differences(baseline, recalculated), "Explicit recalculation restores initial page, all visibility, added report pages and sampled formatting")
                check(recalculated["first"] == 18.25 and recalculated["note"] == "manual report note retained",
                      "Explicit recalculation preserves raw measurement and manual workbook note")
                # Exercise the actual one-shot export button and save dialog.
                # The final OS shell-open is suppressed because it can target
                # an existing user Excel process; the saved report is instead
                # inspected in an independently owned Excel instance below.
                exported_path = phase_output / "一次性含量报告.xls"
                shell_open_requests = []
                check(view.export_button.isEnabled(), "One-shot report export is available in the content workspace")
                with patch.object(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(exported_path), "")), \
                        patch("fdm.live_experiment.excel_template.os.startfile", lambda path: shell_open_requests.append(str(path))):
                    view.export_button.click()
                    def export_finished():
                        if panel.last_report is not None:
                            return panel.last_report
                        if panel.future is None:
                            raise AssertionError({"export_status": panel.status.text(), "diagnostics": panel.diagnostics})
                        return None
                    exported = _wait(app, export_finished, timeout=120, description="actual report-panel export")
                check(exported_path.is_file() and exported_path.stat().st_size > 0, "Report-panel export produces a complete saved workbook")
                export_presentation = probe.call("baseline", source=str(exported_path))
                report["one_shot_export"] = {
                    "destination": str(exported_path), "sha256": sha256(exported_path),
                    "result": exported, "presentation": export_presentation,
                    "shell_open_suppressed": True, "shell_open_requests": shell_open_requests}
                check(not view_differences(baseline, export_presentation), "Saved one-shot report preserves initial page, all sheets, visibility, shapes and sampled formatting/formulas")
                check(export_presentation["first"] == 18.25, "Saved one-shot report contains the actual trial measurement")
                check(exported.get("template_identity", {}).get("sha256", "").lower() == sha256(selected),
                      "One-shot report uses the same chosen source bytes as live Excel")
                check(read(state)["note"] == "manual report note retained", "One-shot report leaves live workbook manual note intact")
                # Leaving and re-entering uses the same associated workbook.
                window.measure_workspace_action.trigger()
                app.processEvents()
                window.acquisition_workspace_action.trigger()
                app.processEvents()
                again = synced(session)
                check(again["name"] == state["name"] and again["generation"] == state["generation"], "Returning to content trial reuses current selected workbook")
                # A content replacement under the same name must be re-openable.
                variant = root / "same-name-next-version.xlt"
                probe.call("variant", source=str(selected), destination=str(variant), marker=uuid4().hex)
                shutil.copyfile(variant, selected)
                check(sha256(selected) != config["source_sha256"], "Same-path replacement has a distinct content hash")
                pick(panel, selected)
                newer = synced(session, state["generation"] + 1)
                check(newer.get("template_identity", {}).get("sha256", "").lower() == sha256(selected), "Same-path selection uses replacement file bytes")
                check(not view_differences(baseline, read(newer)), "Same-path replacement preserves custom report pages")
                check(read(state)["note"] == "manual report note retained", "Previous workbook remains open with manual report note")
                before_missing = manager.status(session.id)
                pick(panel, root / "does-not-exist.xlt")
                app.processEvents()
                after_missing = manager.status(session.id)
                check(after_missing.get("name") == before_missing.get("name") and
                      after_missing.get("generation") == before_missing.get("generation"),
                      "Missing selected source does not silently open another template")
                invalid = root / "invalid-mapping-template.xlt"
                probe.call("invalid", source=str(selected), destination=str(invalid))
                # Replace the *same chosen path*, not merely select another
                # filename: the active workbook must retain its old identity
                # and keep accepting data while the failed new bytes remain
                # at that path. Everything here is a validation-owned copy.
                valid_backup = root / "last-valid-selected-template.xlt"
                shutil.copyfile(selected, valid_backup)
                shutil.copyfile(invalid, selected)
                check(sha256(selected) != newer["template_identity"]["sha256"],
                      "Invalid replacement changes bytes at the exact active template path")
                pick(panel, selected)
                def replacement_failed():
                    status = manager.status(session.id)
                    return status if (status.get("template_replacement_failed") or status.get("error_code")) and not status.get("template_replacement_pending") else None
                failed = _wait(app, replacement_failed, description="invalid replacement rejected")
                # Production deliberately leaves a failed, user-visible
                # candidate open. Its returned exact name belongs to this
                # acceptance run and may be closed with the same PID+creation
                # guard; unrecorded books are never swept up by cleanup.
                if failed.get("candidate_name"):
                    books[failed["candidate_name"]] = {
                        **failed, "name": failed["candidate_name"]}
                    report["failed_candidate"] = {
                        key: failed.get(key) for key in ("candidate_name", "candidate_template_identity",
                                                        "excel_pid", "excel_created", "error_code")}
                check(failed.get("connected") and failed.get("name") == newer["name"], "Failed candidate preserves the successful workbook association")
                check(read(newer)["first"] == 18.25, "Failed candidate cannot alter successful workbook data")
                session.add_measurement(19.75, "template-flow-wool")
                manager.update(session)
                continued = synced(session, newer["generation"])
                continued_book = read(continued)
                failure_reason = str(continued.get("template_replacement_message") or "").strip()
                failure_label = panel.live_template_label.text()
                report["failed_replacement_continued"] = {
                    "state": continued, "workbook": continued_book, "panel_label": failure_label,
                    "selected_file_sha256": sha256(selected), "valid_backup_sha256": sha256(valid_backup)}
                check(continued["name"] == newer["name"] and continued["generation"] == newer["generation"],
                      "Measurement after same-path replacement failure continues in the old workbook and generation")
                check(continued_book["first"] == 18.25 and continued_book["second"] == 19.75,
                      "Old workbook receives the next real measurement in D26 after failed replacement")
                check(continued.get("template_replacement_failed") is True and bool(failure_reason)
                      and Path(continued.get("template_replacement_path", "")).resolve() == selected.resolve(),
                      "Successful old-workbook synchronization retains the independent same-path replacement failure")
                check("更换未完成" in failure_label and failure_reason in failure_label,
                      "Actual report panel still shows unfinished replacement and its specific cause after measurement")
                window.grab().save(str(phase_output / "same-path-failed-continued.png"))
                shutil.copyfile(valid_backup, selected)
                pick(panel, selected)
                final = synced(session, newer["generation"] + 1)
                final_book = read(final)
                check(not view_differences(baseline, final_book), "Valid selection recovers after failed replacement")
                check(final_book["first"] == 18.25 and final_book["second"] == 19.75,
                      "Recovered workbook receives both measurements accumulated across the failed replacement")
                check(not final.get("template_replacement_failed") and not final.get("template_replacement_message")
                      and "更换未完成" not in panel.live_template_label.text(),
                      "Successful valid replacement clears its failure flag and visible failure notice")
                window.grab().save(str(phase_output / "main-window-selected.png"))
            else:
                check(Path(initial["template_path"]).resolve() == selected.resolve(), "Fresh application process reads persisted chosen template")
                check(initial.get("template_identity", {}).get("sha256", "").lower() == sha256(selected), "Restart source hash matches persisted file")
                check(not view_differences(baseline, read(initial)), "Restart preserves custom pages and initial active sheet")
                identity_pair = [initial["excel_pid"], initial["excel_created"]]
                check(identity_pair not in config.get("previous_excel_identities", []),
                      "Restart creates a distinct Excel process identity")
                window.grab().save(str(phase_output / "main-window-restarted.png"))
            check(not report["slot_errors"], "No uncaught Qt slot exception")
            check(sha256(config["source_template"]) == config["source_sha256"], "Original user template bytes unchanged")
            report["passed"] = True
        except Exception as exc:
            report["error"] = str(exc)
            report["traceback"] = traceback.format_exc()
            if window is not None:
                window.grab().save(str(phase_output / "failure.png"))
        finally:
            if manager is not None:
                try:
                    _wait(app, manager.close, timeout=30, description="manager final synchronization")
                except Exception as exc:
                    report["close_error"] = str(exc)
            if window is not None:
                window._confirm_close_documents_for_transition = lambda *args: True
                window.close()
                window.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                app.processEvents()
            grouped = {}
            for name, state in books.items():
                grouped.setdefault((state["excel_pid"], state["excel_created"]), []).append(name)
            report["cleanup"] = []
            for (pid, created), names in grouped.items():
                try:
                    report["cleanup"].append(probe.call("cleanup", excel_pid=pid, excel_created=created, names=names))
                except Exception as exc:
                    report["cleanup"].append({"error": str(exc), "pid": pid, "names": names})
            report["owned_workbook_names"] = list(books)
            report["excel_identities"] = [[pid, created] for pid, created in grouped]
            report["cleanup_failures"] = cleanup_failures(report)
            if report["cleanup_failures"]:
                report["passed"] = False
            (phase_output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-template", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--phase", choices=("select", "restart"), help=argparse.SUPPRESS)
    parser.add_argument("--owner-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        parser.error("Windows desktop Excel is required")
    identity = application_identity()
    if args.require_installed and not (identity["frozen"] and identity["installed_registry_match"]):
        parser.error("Installed validation requires the frozen EXE at its registered installation path")
    root = args.output.resolve()
    if args.phase:
        config = json.loads((root / "validation-owner.json").read_text(encoding="utf-8"))
        if not args.owner_token or config["owner_token"] != args.owner_token:
            parser.error("Missing validation output ownership token")
        return run_phase(root, args.phase)
    if args.source_template is None or not args.source_template.is_file():
        parser.error("--source-template must name an existing template")
    source = args.source_template.resolve()
    root.mkdir(parents=True, exist_ok=False)
    selected = root / "所选含量模板.xlt"
    shutil.copyfile(source, selected)
    profile = root / "profile"
    (profile / "templates").mkdir(parents=True)
    # A historical personal file is deliberately present: current defaults must
    # not read it merely because an older application seeded this path.
    (profile / "templates" / "sheet.xlt").write_bytes(b"obsolete personal template fixture")
    config = {"owner_token": uuid4().hex, "source_template": str(source), "source_sha256": sha256(source)}
    (root / "validation-owner.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"passed": False, "created_utc": datetime.now(timezone.utc).isoformat(), "application": identity,
               "installed_validation": bool(identity["frozen"] and identity["installed_registry_match"]),
               "source_only": not identity["frozen"], "phases": []}
    try:
        for phase in ("select", "restart"):
            if getattr(sys, "frozen", False):
                command = [sys.executable, "--verify-trial-template-flow"]
            else:
                command = [sys.executable, "-m", "fdm.validation.trial_template_flow"]
            command += ["--output", str(root), "--phase", phase, "--owner-token", config["owner_token"]]
            if args.require_installed:
                command.append("--require-installed")
            environment = os.environ.copy()
            if not getattr(sys, "frozen", False):
                source_root = str(Path(__file__).resolve().parents[2])
                environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
            completed = subprocess.run(command, capture_output=True, timeout=600, env=environment)
            (root / f"{phase}.stdout.txt").write_bytes(completed.stdout)
            (root / f"{phase}.stderr.txt").write_bytes(completed.stderr)
            result = root / phase / "result.json"
            phase_result = json.loads(result.read_text(encoding="utf-8")) if result.is_file() else {"error": "phase produced no result"}
            summary["phases"].append({"phase": phase, "exit_code": completed.returncode, "result": phase_result})
            if completed.returncode or not phase_result.get("passed"):
                break
            if phase == "select":
                config["previous_excel_identities"] = phase_result.get("excel_identities", [])
                (root / "validation-owner.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["passed"] = len(summary["phases"]) == 2 and all(item["result"].get("passed") for item in summary["phases"])
    except Exception as exc:
        summary["error"] = str(exc)
    finally:
        summary["original_template_unchanged"] = sha256(source) == config["source_sha256"]
        summary["passed"] = summary["passed"] and summary["original_template_unchanged"]
        (root / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": summary["passed"], "report": str(root / "result.json"),
                      "installed_validation": summary["installed_validation"]}, ensure_ascii=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
