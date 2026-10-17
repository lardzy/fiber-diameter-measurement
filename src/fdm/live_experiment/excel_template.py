"""CU template input contract, independent of Excel and the measurement loop."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from fdm.atomic_io import atomic_write_text


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


def run_excel_request(request, *, timeout=90, cancel_event=None):
    """A dedicated COM process; no Excel call or wait executes on the GUI."""
    if sys.platform != "win32":
        raise RuntimeError("CU 模板报告需要 Windows 和 Microsoft Excel；原始记录仍可导出 CSV。")
    if not Path(request["template"]).is_file():
        raise FileNotFoundError("未找到所选 Excel 模板，请在试验设置中选择 sheet.xlt。")
    request = {**request, "template": str(Path(request["template"]).resolve())}
    script = Path(__file__).with_name("excel_template.ps1")
    with tempfile.TemporaryDirectory(prefix="fdm-template-") as temporary:
        root = Path(temporary)
        request_path, result_path = root / "request.json", root / "result.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, allow_nan=False), encoding="utf-8-sig")
        command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-STA", "-File", str(script),
                   "-RequestPath", str(request_path), "-ResultPath", str(result_path)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + timeout
            while True:
                if time.monotonic() >= deadline or (cancel_event is not None and cancel_event.is_set()):
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    stdout, stderr = process.communicate(timeout=.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            # The worker records only its own newly created Excel instance.
            # Cleanup validates the creation timestamp before terminating it.
            subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(script),
                            "-RequestPath", str(request_path), "-ResultPath", str(result_path), "-Cleanup"],
                           timeout=15, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                stage = json.loads(Path(str(result_path) + ".progress.json").read_text(encoding="utf-8-sig")).get("stage", "")
            except (OSError, ValueError):
                stage = ""
            raise TimeoutError(f"Excel 模板任务超时（{stage or '等待 Excel'}），采集与试验记录未受影响。") from None
        try:
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            raise RuntimeError("Excel 模板任务未返回结果：" + stderr.decode("utf-8", errors="replace")[-800:]) from None
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Excel 模板任务失败。"))
        return result
