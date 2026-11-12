"""Windows source validation for explicit Excel recovery and presentation.

Creates private trial workbooks in independent Excel instances and uses an
isolated settings directory. Never attaches to a pre-existing user's workbook.
The deliberate close/exit probes operate only on recorded PID + creation time.
No installer or hardware is involved. Evidence (including images) stays local.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-template", type=Path, required=True)
    parser.add_argument("--diameter-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("A Windows desktop with Excel is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
    from fdm import settings
    from fdm.live_experiment import ExperimentSession, MaterialSnapshot
    from fdm.live_experiment.excel_live import PowerShellLiveTransport
    from fdm.live_experiment.excel_template import save_template_preferences
    from fdm.ui.live_excel import LiveExcelManager, LiveExcelControls
    from excel_owned_probe import ExcelProbe, PROBE_PS, view_differences

    report = {"passed": False, "source_only": True, "native_excel": True,
              "hardware": False, "checks": [], "events": [], "requests": [], "cleanup": [],
              "initial_template_comparison": {}}
    app = QApplication.instance() or QApplication([])
    owned = {}
    manager = None
    windows = []
    modal = None
    modal_window = None
    sources = {}
    source_views = {}
    probe = ExcelProbe(output / "owned-excel")
    # Extend the existing identity-checked probe, keeping its exact process
    # ownership barrier. No GetActiveObject or global ActiveWorkbook is used.
    code = PROBE_PS
    code = code.replace("if($r.operation -eq 'note') {SetValidationName $book '_FDM_Template_Flow_Note' 'manual report note retained'}",
        """if($r.operation -eq 'note') {
   SetValidationName $book '_FDM_Template_Flow_Note' 'manual report note retained'
   $book.Worksheets.Item([string]$book.ActiveSheet.Name).Range('AZ100').Value2='manual worksheet content retained'
  }""")
    code = code.replace("if($r.operation -eq 'cleanup') {", """
 if($r.operation -eq 'terminate_instance') {
  # This branch is reachable only after the PID + process birth check above;
  # it simulates a real process exit, including an automation-held instance.
  Stop-Process -InputObject $process -Force
  $process.WaitForExit(5000)|Out-Null
  $result=@{exited=$process.HasExited}
 } elseif($r.operation -eq 'quit_instance') {
  $excel.DisplayAlerts=$false
  while($excel.Workbooks.Count){$excel.Workbooks.Item(1).Close($false)}
  $excel.Quit();$result=@{exited=$true}
 } elseif($r.operation -eq 'cleanup') {""")
    code = code.replace("$result=Snapshot $book\n  $result.note", """
  if($r.operation -eq 'close_book') {
   $book.Close($false);$result=@{closed=$true}
  } elseif($r.operation -eq 'modal') {
   [void]$excel.InputBox('Finish the controlled validation operation',
       'FDM presentation validation', '', 0, 0, [Type]::Missing, [Type]::Missing, 2)
   $result=@{modal_closed=$true}
  } else {
   if($r.operation -eq 'minimize'){$book.Windows.Item(1).WindowState=-4140}
   if($r.operation -eq 'save_as'){
    # A disposable test report can contain newer template formatting. Confirm
    # its legacy XLS compatibility choice only in this validation-owned call.
    $alerts=$excel.DisplayAlerts
    try{$excel.DisplayAlerts=$false;$book.CheckCompatibility=$false;$book.SaveAs([string]$r.destination,56)}
    finally{$excel.DisplayAlerts=$alerts}
   }
   $result=Snapshot $book
   $result.window_state=[int]$book.Windows.Item(1).WindowState
   $result.report_cell=$book.Worksheets.Item([string]$book.ActiveSheet.Name).Range('AZ100').Value2
   $result.note=ReadValidationName $book '_FDM_Template_Flow_Note'
  }
  $result.note""")
    # The original note read after the injected block must not touch a closed
    # COM workbook; the inner block above already reads it when appropriate.
    code = code.replace("  $result.note=ReadValidationName $book '_FDM_Template_Flow_Note'\n }\n}", " }\n}")
    probe.script.write_text(code, encoding="utf-8-sig")

    def check(value, label):
        if not value:
            raise AssertionError(label)
        report["checks"].append(label)

    def wait(predicate, label, timeout=40):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            QTest.qWait(10)
            if predicate():
                return
        raise AssertionError(f"Timeout: {label}: {manager.status(session.id) if manager else ''}")

    def identity(session):
        state = manager.status(session.id)
        return {key: state[key] for key in ("excel_pid", "excel_created", "name")}

    def operate(operation, session, **kw):
        return probe.call(operation, **identity(session), **kw)

    def shown(session):
        state = manager.status(session.id)
        return not state.get("presentation_pending") and state.get("presentation_status") in ("shown", "attention_required")

    user32 = ctypes.windll.user32
    user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    user32.PostMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
    window_callback = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def owned_modal(pid):
        found = []

        @window_callback
        def visit(hwnd, _):
            actual = ctypes.c_uint32()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(actual))
            if actual.value == int(pid):
                title = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, title, len(title))
                if title.value == "FDM presentation validation":
                    found.append(hwnd)
            return True

        user32.EnumWindows(visit, None)
        return found[0] if found else None

    class RecordingTransport(PowerShellLiveTransport):
        def request(self, operation, sid="", generation=0, revision=0, payload=None, **kwargs):
            report["requests"].append({"operation": operation, "sid": sid,
                "generation": generation, "revision": revision, "time": time.perf_counter()})
            result = super().request(operation, sid, generation, revision, payload, **kwargs)
            if result.get("excel_pid") and result.get("excel_created"):
                key = (result["excel_pid"], result["excel_created"])
                owned.setdefault(key, set())
                if result.get("name"):
                    owned[key].add(result["name"])
            return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(settings, "settings_directory", lambda: output / "profile"))
        stack.enter_context(patch.object(settings, "settings_file_path", lambda: output / "profile/settings.json"))
        try:
            for mode, source in (("content", args.content_template), ("diameter", args.diameter_template)):
                sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
                copy = output / f"{mode}.xlt"
                shutil.copy2(source, copy)
                save_template_preferences({"template": str(copy)}, trial_mode=mode)
                source_views[mode] = probe.call("baseline", source=str(copy))
            manager = LiveExcelManager(transport=RecordingTransport(), coalesce_ms=50)
            manager.changed.connect(lambda sid: report["events"].append(
                {"time": time.perf_counter(), "sid": sid, **manager.status(sid)}))
            sessions = []
            for mode in ("content", "diameter"):
                session = ExperimentSession(workspace_options={"mode": mode}, save_images=False)
                session.add_operator("显示验证")
                session.set_materials([MaterialSnapshot("wool", "绵羊毛", 1.31)])
                session.add_measurement(18.25, "wool")
                sessions.append(session)
                window = QWidget()
                window.setWindowTitle("FDM Excel 显示验证 · " + mode)
                layout = QVBoxLayout(window)
                controls = LiveExcelControls(manager, session, window, compact=True)
                layout.addWidget(controls)
                windows.append(window)
                window.show()
                manager.ensure(session)
                # Input acknowledgement can precede the final metadata reply.
                # The ownership probe needs the completed PID/birth identity.
                wait(lambda: manager.flush(session) and manager._busy is None
                     and manager.status(session.id).get("excel_pid")
                     and manager.status(session.id).get("excel_created"), "initial synchronization and identity")
                first = operate("note", session)
                differences = view_differences(source_views[mode], first)
                report["initial_template_comparison"][mode] = {
                    "baseline": source_views[mode], "workbook": first, "differences": differences}
                check(not differences,
                      f"{mode}: initial FDM workbook preserves source template pages and active sheet")
                before = len(report["requests"])
                operate("minimize", session)
                QTest.mouseClick(controls.excel_button, Qt.MouseButton.LeftButton)
                wait(lambda: shown(session), "minimized workbook presentation")
                after = operate("read", session)
                check(after["window_state"] != -4140, f"{mode}: explicit button restores minimized workbook")
                check(not any(r["operation"] in ("open", "sync", "replace") for r in report["requests"][before:]),
                      f"{mode}: show does not rewrite measurement data")
                check(not view_differences(first, after), f"{mode}: presentation preserves template pages and styles")
                check(after["note"] == "manual report note retained", f"{mode}: presentation retains manual report note")
                check(after["report_cell"] == "manual worksheet content retained", f"{mode}: presentation retains manual worksheet cell")

                manager.resync(session, present_after=True)
                wait(lambda: manager.flush(session) and shown(session), "explicit resync and presentation")
                after = operate("read", session)
                check(after["note"] == first["note"], f"{mode}: resync retains manual report note")
                check(after["report_cell"] == "manual worksheet content retained", f"{mode}: resync does not overwrite manual worksheet cell")

                if mode == "content":
                    # Real Excel modal/busy state, not a fake COM exception.
                    # Only the recorded private instance gets this InputBox.
                    request = output / "modal-request.json"
                    request.write_text(json.dumps({"operation": "modal", **identity(session),
                        "result": str(output / "modal-result.json")}), encoding="utf-8")
                    modal = subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-STA",
                        "-ExecutionPolicy", "Bypass", "-File", str(probe.script), "-RequestPath", str(request)],
                        stdout=subprocess.DEVNULL, stderr=(output / "modal.stderr.txt").open("wb"))
                    wait(lambda: owned_modal(identity(session)["excel_pid"]), "controlled native Excel modal")
                    modal_window = owned_modal(identity(session)["excel_pid"])
                    manager.present(session)
                    wait(lambda: manager.status(session.id).get("presentation_status") in
                         ("deferred", "shown", "attention_required"), "presentation while Excel is busy", 10)
                    check(bool(manager.status(session.id).get("window_visible")), "busy Excel: verified window is visible without waiting for COM activation")
                    session.add_measurement(19.75, "wool")
                    manager.update(session)
                    QTest.qWait(200)
                    user32.PostMessageW(modal_window, 0x10, 0, 0)
                    modal_window = None
                    wait(lambda: modal.poll() is not None, "controlled modal closes")
                    check(modal.returncode == 0, "native Excel modal helper completes normally")
                    modal = None
                    wait(lambda: manager.flush(session), "pending measurement after modal")
                    check(operate("read", session)["second"] == 19.75, "busy Excel: pending measurement is written after user operation")

                # Freeze polling to exercise a click before the cached connected
                # state has noticed the user's close. One click must recover.
                manager._timer.stop()
                operate("close_book", session)
                before = len(report["requests"])
                QTest.mouseClick(controls.excel_button, Qt.MouseButton.LeftButton)
                for _ in range(5):
                    controls.excel_button.click()
                manager._timer.start()
                wait(lambda: manager.flush(session) and shown(session) and
                     any(r["operation"] == "open" for r in report["requests"][before:]), "stale binding recovery")
                check(sum(r["operation"] == "open" for r in report["requests"][before:]) == 1,
                      f"{mode}: stale binding and repeated clicks create exactly one replacement")
                after = operate("read", session)
                if mode == "content":
                    check(after["first"] == 18.25, "content: one-click recovery restores accepted input D25")
                check(after["note"] is None, f"{mode}: recovery does not fabricate discarded manual content")

                operate("note", session)
                saved = operate("save_as", session, destination=str(output / f"user-save-{mode}.xls"))
                manager.show(session.id)
                wait(lambda: manager.status(session.id).get("name") == saved["name"], "SaveAs identity tracking")
                manager.resync(session, present_after=True)
                wait(lambda: manager.flush(session) and shown(session), "saved report resync")
                check(operate("read", session)["note"] == "manual report note retained",
                      f"{mode}: SaveAs and resync retain report information")

            session = sessions[0]
            before = len(report["requests"])
            manager.present(session)
            wait(lambda: shown(session) and any(r["operation"] == "show" for r in report["requests"][before:]),
                 "switch back to first workbook")
            check(not any(r["operation"] == "open" for r in report["requests"][before:]),
                  "two trials: presenting earlier workbook does not create another workbook")

            manager._timer.stop()
            check(operate("terminate_instance", session)["exited"], "private Excel process actually exits before reconnect")
            before = len(report["requests"])
            manager.present(session)
            manager._timer.start()
            wait(lambda: manager.flush(session) and shown(session) and
                 any(r["operation"] == "open" for r in report["requests"][before:]), "whole Excel exit recovery")
            check(sum(r["operation"] == "open" for r in report["requests"][before:]) == 1,
                  "whole Excel exit: one click rebuilds exactly once")
            manager.stop(session.id)
            wait(lambda: manager.is_detached(session.id), "explicit stop")
            manager.present(session)
            wait(lambda: manager.flush(session) and shown(session), "explicit resume")
            check(manager.status(session.id).get("connected"), "explicit stopped state resumes on user request")
            for source, digest in sources.items():
                check(hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest, "source template unchanged: " + source)
            report["passed"] = True
        except Exception as exc:
            import traceback
            report["error"] = str(exc)
            report["traceback"] = traceback.format_exc()
        finally:
            if modal_window:
                user32.PostMessageW(modal_window, 0x10, 0, 0)
            if modal is not None:
                try:
                    modal.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    modal.kill()
                    modal.wait(timeout=5)
            if manager is not None:
                try:
                    wait(manager.close, "manager shutdown", 30)
                except Exception as exc:
                    report["passed"] = False
                    report["close_error"] = str(exc)
            for (pid, created), names in owned.items():
                try:
                    # Whole-instance exit is permitted only for these newly
                    # created independent test instances; the helper rechecks
                    # PID/creation time immediately before touching Excel.
                    result = probe.call("quit_instance", excel_pid=pid, excel_created=created, name=next(iter(names), ""))
                    report["cleanup"].append({"pid": pid, **result})
                except Exception as exc:
                    # The explicit exit test can already have ended this PID.
                    command = ["powershell.exe", "-NoProfile", "-Command", f"Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue | Select-Object Id"]
                    exists = subprocess.run(command, capture_output=True, timeout=10).stdout.strip()
                    report["cleanup"].append({"pid": pid, "already_exited": not bool(exists), "error": str(exc) if exists else ""})
                    if exists:
                        report["passed"] = False
            for window in windows:
                window.close()
            (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]), "output": str(output)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
