"""CU template input contract, independent of Excel and the measurement loop."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
import base64
import json
import locale
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from fdm.atomic_io import atomic_replace_file, atomic_write_text, staged_path_for
from fdm.version import __version__


@dataclass(frozen=True)
class TemplateProfile:
    data_sheet: str = "原始数据"
    standard_sheet: str = "标准选择"
    method_sheet: str = "比重"
    method_range: str = "E3:E16"
    method_index_cell: str = "D10"
    group_width_cell: str = "F10"
    date_cell: str = "A1"
    operator_cells: tuple = ("D3", "V3", "AN3")
    sample_name_cells: tuple = ("E1", "W1", "AO1")
    sample_number_cells: tuple = ("G1", "Y1", "AQ1")
    column_starts: tuple = ((4, 13), (22, 31), (40, 49))
    weight_row: int = 5
    material_row: int = 6
    density_row: int = 7
    count_row: int = 8
    measured_count_row: int = 9
    content_row: int = 21
    first_data_row: int = 25
    clear_last_row: int = 20000
    recalculate_macro: str = "分组"
    # User-declared capacity is informative, never a content fingerprint gate.
    grouped_capacity: int | None = None
    method_labels: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path=None):
        if not path:
            return cls()
        values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict) or values.keys() - cls.__dataclass_fields__.keys():
            raise ValueError("模板映射配置包含未知字段。")
        profile = cls(**values)
        if (len(profile.column_starts) != 3 or any(len(pair) != 2 for pair in profile.column_starts)
                or any(not isinstance(c, int) or not 1 <= c <= 248 for pair in profile.column_starts for c in pair)
                or any(len(getattr(profile, name)) != 3 for name in
                       ("operator_cells", "sample_name_cells", "sample_number_cells"))):
            raise ValueError("模板映射须包含三位操作者的经纬九类输入位置。")
        for name in ("weight_row", "material_row", "density_row", "count_row", "measured_count_row",
                     "content_row", "first_data_row", "clear_last_row"):
            if not isinstance(getattr(profile, name), int) or not 1 <= getattr(profile, name) <= 65536:
                raise ValueError(f"模板映射行号无效：{name}")
        if profile.first_data_row <= max(profile.material_row, profile.density_row, profile.count_row,
                                          profile.measured_count_row, profile.content_row):
            raise ValueError("原始直径输入区不能覆盖元数据和计算行。")
        return profile


@dataclass(frozen=True)
class ExportSnapshot:
    id: str
    revision: int
    created_at: str
    name: str
    number: str
    method: str
    group_width: float
    grouped: bool
    operators: tuple
    measurements: tuple
    adjustments: tuple
    local_summary: dict

    @classmethod
    def capture(cls, session):
        from .statistics import summarize
        # Records/events are immutable. Snapshot references in O(n), without
        # serializing provenance or image evidence on the GUI thread.
        return cls(session.id, session.revision, session.created_at, session.sample_name,
            session.sample_number, session.method, session.group_width, session.grouped,
            tuple(deepcopy(operator) for operator in session.operators.values()),
            tuple(session.measurements.values()), tuple(session.count_adjustments.values()),
            deepcopy(summarize(session)))


def column_name(value):
    result = ""
    while value:
        value, part = divmod(value - 1, 26)
        result = chr(65 + part) + result
    return result


def template_request(snapshot, template, destination, profile=None, *, method_label=None, show=True):
    from .statistics import resolve_method
    profile = profile or TemplateProfile()
    if len(snapshot.operators) > 3:
        raise ValueError("CU 模板只定义三位操作者，未截断额外操作者。")
    groups, counts = {}, {}
    for record in snapshot.measurements:
        groups.setdefault((record.operator_id, record.direction, record.material_id), []).append(record.diameter_um)
    for event in snapshot.adjustments:
        key = event.operator_id, event.direction, event.material_id
        counts[key] = counts.get(key, 0) + event.delta
    cells, blocks, clears, checks, warnings = [], [], [], [], []
    date = datetime.fromisoformat(snapshot.created_at).astimezone().date().isoformat()
    cells.append({"cell": profile.date_cell, "value": date})
    for slot, (warp, weft) in enumerate(profile.column_starts):
        # Clear the same input areas as CU, including absent participants, but
        # never formulas in rows 9-24 or unknown user report sheets.
        for start in (warp, weft):
            low, high = column_name(start), column_name(start + 8)
            clears.append(f"{low}{profile.first_data_row}:{high}{profile.clear_last_row}")
            for row in (profile.material_row, profile.density_row, profile.count_row):
                clears.append(f"{low}{row}:{high}{row}")
            clears.append(f"{low}{profile.weight_row}")
        for mapping in (profile.operator_cells, profile.sample_name_cells, profile.sample_number_cells):
            clears.append(mapping[slot])
        if slot >= len(snapshot.operators):
            continue
        operator = snapshot.operators[slot]
        for cell, value in ((profile.operator_cells[slot], operator.name),
                           (profile.sample_name_cells[slot], snapshot.name),
                           (profile.sample_number_cells[slot], snapshot.number)):
            cells.append({"cell": cell, "value": value})
        for direction, start, weight in (("warp", warp, operator.warp_weight), ("weft", weft, operator.weft_weight)):
            cells.append({"cell": f"{column_name(start)}{profile.weight_row}", "value": weight})
            local = snapshot.local_summary["operators"][operator.id][direction]
            local_categories = {item["id"]: item for item in local["categories"]}
            for offset, material in enumerate(operator.materials[direction]):
                col = column_name(start + offset)
                key = operator.id, direction, material.id
                values = groups.get(key, [])
                if len(values) + profile.first_data_row - 1 > 65536:
                    raise ValueError("完整原始数据超过 XLS 行数，未截断导出。")
                for row, value in ((profile.material_row, material.name), (profile.density_row, material.density),
                                   (profile.count_row, counts.get(key, 0))):
                    cells.append({"cell": f"{col}{row}", "value": value})
                if values:
                    blocks.append({"cell": f"{col}{profile.first_data_row}", "values": [[v] for v in values]})
                checks.append({"count_cell": f"{col}{profile.measured_count_row}", "expected": len(values),
                    "content_cell": f"{col}{profile.content_row}", "operator": operator.name,
                    "direction": direction, "material": material.name,
                    "local_content": local_categories[material.id].get("content")})
                if snapshot.grouped and len(values) > (profile.grouped_capacity or 1488):
                    warnings.append(f"{operator.name}／{direction}／{material.name} 共 {len(values)} 条："
                        + ("超过映射声明的分组容量。" if profile.grouped_capacity else
                           "超过原 CU 分组宏的 1488 条范围；自定义宏覆盖范围待核验。"))
    label = method_label or profile.method_labels.get(snapshot.method) or resolve_method(snapshot.method).label
    return {"operation": "export", "template": str(Path(template).resolve()), "destination": str(Path(destination).resolve()),
        "profile": asdict(profile), "cells": cells, "blocks": blocks, "clears": clears, "checks": checks,
        "method_label": label, "group_width": snapshot.group_width, "grouped": snapshot.grouped,
        "snapshot_id": snapshot.id, "snapshot_revision": snapshot.revision, "show": show, "warnings": warnings}


def template_preferences_path():
    from fdm.settings import settings_directory
    return settings_directory() / "content-template.json"


def template_preferences():
    path = template_preferences_path()
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


def save_template_preferences(values):
    path = template_preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def working_template(*, create=False):
    """Seed once; upgrades/new defaults never overwrite the user's working copy."""
    from fdm.settings import runtime_directory, settings_directory
    preferences = template_preferences()
    if preferences.get("template"):
        return Path(preferences["template"])
    source = runtime_directory() / "content-templates" / "sheet.xlt"
    target = settings_directory() / "templates" / "sheet.xlt"
    if create and not target.exists() and source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents overwriting a concurrently edited copy.
        try:
            with target.open("xb") as destination, source.open("rb") as original:
                shutil.copyfileobj(original, destination)
        except FileExistsError:
            pass
    return target


class ExcelTemplateError(RuntimeError):
    """A short user-facing message with separately inspectable diagnostics."""

    def __init__(self, message, *, code="worker_failed", diagnostics=None):
        super().__init__(message)
        self.code = code
        self.diagnostics = {**(diagnostics or {}), "error_code": code, "message": message}


class ExcelTemplateCancelled(ExcelTemplateError):
    pass


class ExcelTemplateTimeout(ExcelTemplateError, TimeoutError):
    pass


def _powershell_command(script, request_path, result_path, *, cleanup=False):
    # This bootstrap only CALLS the packaged script. It does not read/evaluate
    # its contents: MachinePolicy/UserPolicy still govern script execution.
    # Capture errors before .ps1 entry, where the script's own catch cannot run.
    def literal(value):
        return "'" + str(value).replace("'", "''") + "'"

    output_path = str(result_path) + ".cleanup.json" if cleanup else str(result_path)
    environment_path = str(result_path) + (".cleanup-environment.json" if cleanup else ".environment.json")
    bootstrap = "\n".join((
        "$ErrorActionPreference = 'Stop'",
        "$ProgressPreference = 'SilentlyContinue'",
        f"$worker = {literal(script)}",
        f"$requestPath = {literal(request_path)}",
        f"$resultPath = {literal(result_path)}",
        f"$outputPath = {literal(output_path)}",
        f"$environmentPath = {literal(environment_path)}",
        "$environment = @{powershell_version=$PSVersionTable.PSVersion.ToString(); "
        "culture=[Globalization.CultureInfo]::CurrentCulture.Name; "
        "original_console_codepage=[Console]::OutputEncoding.CodePage; "
        "ansi_codepage=[Text.Encoding]::Default.CodePage; requested_policy='Bypass'}",
        "try { [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false) } catch {}",
        "try { $environment.output_codepage=[Console]::OutputEncoding.CodePage; "
        "$environment.effective_policy=[string](Get-ExecutionPolicy); "
        "$environment.policies=@(Get-ExecutionPolicy -List | ForEach-Object { "
        "@{scope=[string]$_.Scope; policy=[string]$_.ExecutionPolicy} }); "
        "$environment | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $environmentPath -Encoding UTF8 } catch {}",
        "try {",
        "    & $worker -RequestPath $requestPath -ResultPath $resultPath" + (" -Cleanup" if cleanup else ""),
        "    if (!$?) { exit 1 }",
        "} catch {",
        "    @{ok=$false; error=$_.Exception.Message; error_type=$_.Exception.GetType().FullName; "
        "error_id=$_.FullyQualifiedErrorId; phase='script_start'} | ConvertTo-Json -Depth 5 | "
        "Set-Content -LiteralPath $outputPath -Encoding UTF8",
        "    exit 1",
        "}",
    ))
    encoded = base64.b64encode(bootstrap.encode("utf-16le")).decode("ascii")
    return ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-STA",
            "-ExecutionPolicy", "Bypass", "-OutputFormat", "Text", "-EncodedCommand", encoded]


def _read_worker_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _decode_worker_output(value, environment):
    if not value:
        return "", "empty"
    if isinstance(value, str):
        return value, "text"
    if value.startswith((b"\xff\xfe", b"\xfe\xff")):
        return value.decode("utf-16", errors="replace"), "utf-16"
    encodings = ["utf-8-sig"]
    for key in ("output_codepage", "original_console_codepage", "ansi_codepage"):
        if environment.get(key):
            encodings.append("cp" + str(environment[key]))
    # A host failure before the bootstrap has no environment sidecar. Windows'
    # ANSI/OEM defaults are explicit fallbacks, never assume a Chinese codepage.
    if sys.platform == "win32":
        try:
            import ctypes
            encodings.extend("cp" + str(getattr(ctypes.windll.kernel32, name)())
                             for name in ("GetOEMCP", "GetACP"))
        except (AttributeError, OSError):
            pass
    encodings.append(locale.getpreferredencoding(False))
    for encoding in dict.fromkeys(encodings):
        try:
            return value.decode(encoding), encoding
        except (UnicodeError, LookupError):
            continue
    # Preserve undecodable bytes in diagnostics instead of destroying them.
    return value.decode("utf-8", errors="backslashreplace"), "utf-8/backslashreplace"


def _worker_diagnostics(request, script, result_path, process=None, stdout=b"", stderr=b""):
    environment = _read_worker_json(str(result_path) + ".environment.json")
    out, out_encoding = _decode_worker_output(stdout, environment)
    err, err_encoding = _decode_worker_output(stderr, environment)
    return {"app_version": __version__, "python_version": sys.version.split()[0], "platform": sys.platform,
            "operation": request.get("operation"), "template": request.get("template"),
            "script": str(script), "execution_policy_requested": "Bypass", "environment": environment,
            "stage": _read_worker_json(str(result_path) + ".progress.json").get("stage", "启动模板组件"),
            "exit_code": getattr(process, "returncode", None),
            "stdout": out[-12000:], "stderr": err[-12000:],
            "stdout_encoding": out_encoding, "stderr_encoding": err_encoding}


def _worker_error(result, diagnostics):
    detail = " ".join(str(result.get(key, "")) for key in ("error", "error_type", "error_id", "code"))
    detail += " " + diagnostics.get("stderr", "")
    normalized = detail.lower()
    if any(token in normalized for token in ("pssecurityexception", "securityerror", "about_execution_policies")):
        return ExcelTemplateError("系统执行策略阻止了模板组件运行，报告未生成。请展开诊断查看策略；"
                                  "若由组织策略管理，请联系管理员处理脚本签名或授权。",
                                  code="execution_policy", diagnostics=diagnostics)
    if any(token in normalized for token in ("80040154", "regdb_e_classnotreg")):
        return ExcelTemplateError("未找到可用的 Microsoft Excel 自动化组件，请确认已安装并可正常打开 Excel。",
                                  code="excel_unavailable", diagnostics=diagnostics)
    if result.get("code") == "invalid_request":
        return ExcelTemplateError("模板请求无法读取或格式不正确；报告未生成，请导出诊断。",
                                  code="invalid_request", diagnostics=diagnostics)
    return ExcelTemplateError("Excel 模板任务未完成，已有报告与试验记录保持不变。请展开诊断查看原因。",
                              code="worker_failed" if result else "missing_result", diagnostics=diagnostics)


def _cleanup_excel_worker(script, request_path, result_path):
    command = _powershell_command(script, request_path, result_path, cleanup=True)
    try:
        process = subprocess.run(command, timeout=15, capture_output=True,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        environment = _read_worker_json(str(result_path) + ".cleanup-environment.json")
        stderr, encoding = _decode_worker_output(process.stderr, environment)
        result = _read_worker_json(str(result_path) + ".cleanup.json")
        return {"ok": process.returncode == 0 and result.get("ok") is True, "exit_code": process.returncode,
                "result": result,
                "environment": environment, "stderr": stderr[-12000:], "stderr_encoding": encoding}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}


def _run_excel_worker(request, *, timeout, cancel_event):
    script = Path(__file__).with_name("excel_template.ps1")
    with tempfile.TemporaryDirectory(prefix="fdm-template-") as temporary:
        root = Path(temporary)
        request_path, result_path = root / "request.json", root / "result.json"
        try:
            request_path.write_text(json.dumps(request, ensure_ascii=False, allow_nan=False), encoding="utf-8-sig")
        except (TypeError, ValueError, OSError) as exc:
            diagnostics = _worker_diagnostics(request, script, result_path)
            diagnostics["request_error"] = str(exc)
            raise ExcelTemplateError("无法准备模板请求；请检查输入数值和临时目录权限。",
                                      code="invalid_request", diagnostics=diagnostics) from exc
        command = _powershell_command(script, request_path, result_path)
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            diagnostics = _worker_diagnostics(request, script, result_path)
            diagnostics["launch_error"] = str(exc)
            raise ExcelTemplateError("无法启动 Windows PowerShell 模板组件，请展开诊断查看系统错误。",
                                      code="launch_failed", diagnostics=diagnostics) from exc
        deadline, reason, transport_error = time.monotonic() + timeout, None, None
        stdout = stderr = b""
        while True:
            if cancel_event is not None and cancel_event.is_set():
                reason = "cancelled"
                break
            if time.monotonic() >= deadline:
                reason = "timeout"
                break
            try:
                stdout, stderr = process.communicate(timeout=min(.2, max(.001, deadline-time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                continue
            except OSError as exc:
                reason, transport_error = "communication_failed", str(exc)
                break
        if reason:
            stop_error = None
            try:
                process.kill()
                stdout, stderr = process.communicate(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as exc:
                stop_error = str(exc)
            diagnostics = _worker_diagnostics(request, script, result_path, process, stdout, stderr)
            diagnostics["worker_stop_error"] = stop_error
            diagnostics["communication_error"] = transport_error
            diagnostics["cleanup"] = _cleanup_excel_worker(script, request_path, result_path)
            stage = diagnostics["stage"]
            cleanup_note = "" if diagnostics["cleanup"]["ok"] else " Excel 清理未确认完成，请查看诊断。"
            if reason == "cancelled":
                raise ExcelTemplateCancelled("模板任务已取消；已有报告与试验记录保持不变。" + cleanup_note,
                                             code=reason, diagnostics=diagnostics)
            if reason == "communication_failed":
                raise ExcelTemplateError("模板组件连接中断；已有报告与试验记录保持不变。请展开诊断。" + cleanup_note,
                                          code=reason, diagnostics=diagnostics)
            raise ExcelTemplateTimeout(f"Excel 模板任务超时（{stage}）；已有报告与试验记录保持不变。" + cleanup_note,
                                       code=reason, diagnostics=diagnostics)
        result = _read_worker_json(result_path)
        diagnostics = _worker_diagnostics(request, script, result_path, process, stdout, stderr)
        closed = _read_worker_json(str(result_path) + ".close.json")
        diagnostics["excel_close"] = closed
        diagnostics["worker_result"] = {key: result[key] for key in ("error", "error_type", "error_id", "stack", "phase", "code", "excel_version")
                                        if key in result}
        if result.get("ok") is not True or process.returncode != 0:
            # Also recover an owned instance after an unexpected process exit.
            # The worker's PID + creation-time check excludes other workbooks.
            if Path(str(result_path) + ".owner.json").is_file():
                diagnostics["cleanup"] = _cleanup_excel_worker(script, request_path, result_path)
            raise _worker_error(result, diagnostics)
        if closed.get("ok") is False and not closed.get("handed_off"):
            diagnostics["cleanup"] = _cleanup_excel_worker(script, request_path, result_path)
            warning = ("Excel 关闭时发生异常，清理检查已完成；模板结果已保留。" if diagnostics["cleanup"]["ok"] else
                       "模板结果已生成，但 Excel 清理未确认完成；请展开诊断。")
            result["warnings"] = [*result.get("warnings", []), warning]
        return {**result, "diagnostics": diagnostics}


def run_excel_request(request, *, timeout=90, cancel_event=None):
    """Run COM off the GUI, then atomically publish a complete report copy."""
    base = {"app_version": __version__, "platform": sys.platform, "operation": request.get("operation")}
    if sys.platform != "win32":
        raise ExcelTemplateError("CU 模板报告需要 Windows 和 Microsoft Excel；原始记录仍可导出 CSV。",
                                  code="unsupported_platform", diagnostics=base)
    template = Path(request["template"]).resolve()
    if not template.is_file():
        raise ExcelTemplateError("未找到所选 Excel 模板，请重新选择模板。", code="template_missing", diagnostics=base)
    request = {**request, "template": str(template)}
    if cancel_event is not None and cancel_event.is_set():
        raise ExcelTemplateCancelled("模板任务已取消。", code="cancelled", diagnostics=base)
    if request.get("operation") != "export":
        return _run_excel_worker(request, timeout=timeout, cancel_event=cancel_event)
    destination = Path(request["destination"]).resolve()
    if destination == template or (destination.exists() and os.path.samefile(destination, template)):
        raise ExcelTemplateError("报告不能覆盖模板，请选择其他保存位置。", code="template_overwrite", diagnostics=base)
    diagnostics = base
    try:
        # Excel never opens or overwrites the final destination during export.
        # Publication occurs only after the worker has saved and closed COM.
        with staged_path_for(destination, suffix=destination.suffix) as staged:
            result = _run_excel_worker({**request, "destination": str(staged), "show": False},
                                       timeout=timeout, cancel_event=cancel_event)
            diagnostics = result["diagnostics"]
            if cancel_event is not None and cancel_event.is_set():
                raise ExcelTemplateCancelled("模板任务已取消；已有报告与试验记录保持不变。",
                                             code="cancelled", diagnostics=diagnostics)
            if not staged.is_file() or staged.stat().st_size == 0:
                raise ExcelTemplateError("Excel 未生成完整报告文件；已有报告保持不变。",
                                          code="report_missing", diagnostics=diagnostics)
            atomic_replace_file(staged, destination)
    except ExcelTemplateError as exc:
        exc.diagnostics["destination"] = str(destination)
        raise
    except OSError as exc:
        raise ExcelTemplateError("报告无法保存到所选位置；已有报告保持不变。请检查文件是否被占用及目录权限。",
                                  code="publish_failed", diagnostics={**diagnostics, "publish_error": str(exc)}) from exc
    result["destination"] = str(destination)
    result["diagnostics"] = {**diagnostics, "destination": str(destination), "stage": "报告已保存", "published": True}
    if request.get("show"):
        try:
            os.startfile(str(destination))
        except OSError as exc:
            result["warnings"] = [*result.get("warnings", []), "报告已保存，但无法自动打开；请从保存位置打开报告。"]
            result["diagnostics"]["open_error"] = str(exc)
    return result
