"""Recovered CU calculation rules, with explicit validity instead of Excel errors.

All measurements participate. Fixed worksheet row/bucket truncation and the
unassigned RoundN boundary are deliberately repaired; this is not a standards
certification implementation. Snapshot metadata identifies these rule revisions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, TYPE_CHECKING

from .models import _Aggregate

if TYPE_CHECKING:
    from .models import ExperimentSession, Operator


CALCULATION_REVISION = "cu-content-rules-1"


@dataclass(frozen=True)
class MethodDefinition:
    id: str
    label: str
    family: str
    default_group_width: float

    @property
    def requires_grouping(self) -> bool:
        return self.family in ("bast_upper", "bast_mid", "empirical")


METHODS = (
    MethodDefinition("gb16988", "GB/T 16988-2013", "variance", 2.5),
    MethodDefinition("iso17751", "ISO 17751-1:2016(E)", "variance", 2.5),
    MethodDefinition("cns2339", "CNS 2339-1 L3050-1", "variance", 2.5),
    MethodDefinition("can14", "CAN CGSB-4.2 No.14-2005", "variance", 2.5),
    MethodDefinition("jis1030", "JIS L 1030-2：2012", "variance", 2.5),
    MethodDefinition("sn1062", "SN/T 1062-2002", "mean_square", 2.5),
    MethodDefinition("fz01101", "FZ/T 01101-2008", "square_mean", 2.0),
    MethodDefinition("aatcc20a", "AATCC 20A-2018", "square_mean", 2.0),
    MethodDefinition("sn0756", "SN/T 0756-1999", "bast_upper", 2.5),
    MethodDefinition("fz32004", "FZ/T 32004-2009", "bast_mid", 2.0),
    MethodDefinition("fz30003", "FZ/T 30003-2009", "empirical", 2.0),
)


def resolve_method(value: str) -> MethodDefinition:
    for method in METHODS:
        if value in (method.id, method.label) or value.replace(":", "：") == method.label:
            return method
    raise ValueError(f"不支持的含量计算方法：{value}")


def resolve_profile(value: str) -> str:
    if value in ("normal", "cu5", "cu6"):
        return "normal"
    if value in ("decolorization_free", "cu6mt", "no_bleach", "mt"):
        return "decolorization_free"
    raise ValueError("模板版本必须为 normal 或 decolorization_free")


def round_cu(value: float, digits: int = 2) -> float:
    """CU RoundN, assigning its previously unassigned exact tolerance boundaries."""
    if not math.isfinite(value):
        raise ValueError("不能舍入非有限数值")
    factor = 10.0 ** digits
    absolute = abs(value)
    integer = math.floor(absolute * factor)
    fraction = absolute - integer / factor
    half = 0.5 / factor
    if fraction > half + 1e-10:
        integer += 1
    elif fraction >= half - 1e-10 and integer % 2:
        integer += 1
    return math.copysign(integer / factor, value)


def _mt_round(value: float) -> float:
    # The shipped 54 formulas operate on the already two-decimal totals, using
    # MOD parity for the one-decimal tie; exact rational cents avoids binary
    # representation of a displayed .05 deciding that tie inconsistently.
    cents = int(round(value * 100))
    sign = -1 if cents < 0 else 1
    quotient, remainder = divmod(abs(cents), 10)
    if remainder > 5 or (remainder == 5 and quotient % 2):
        quotient += 1
    return sign * quotient / 10.0


def _status(reasons: list[dict[str, str]], has_values: bool = True) -> str:
    if any(reason["severity"] == "invalid" for reason in reasons):
        return "invalid"
    return "not_ready" if reasons or not has_values else "ready"


def _reason(code: str, message: str, *, severity: str = "not_ready", material_id: str = "") -> dict[str, str]:
    return {"code": code, "message": message, "severity": severity, "material_id": material_id}


def _direction_summary(session: "ExperimentSession", operator: "Operator", direction: str,
                       method: MethodDefinition) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reasons: list[dict[str, str]] = []
    if method.requires_grouping and not session.grouped:
        reasons.append(_reason("grouping_required", "该棉麻方法需要分组统计", severity="invalid"))
    for material in operator.materials[direction]:
        aggregate = session._aggregates.get((operator.id, direction, material.id), _Aggregate())
        moment = aggregate.grouped if session.grouped else aggregate.raw
        m, c = moment.count, aggregate.extra_count
        row: dict[str, Any] = {"id": material.id, "name": material.name, "family": material.family,
            "density": material.density, "measured_count": m, "extra_count": c, "total_count": m + c,
            "mean": aggregate.raw.mean if m else None,
            "stddev": math.sqrt(aggregate.raw.variance) if m > 1 else None,
            "calculation_mean": moment.mean if m else None,
            "calculation_stddev": math.sqrt(moment.variance) if (m > 1 or (m == 1 and session.grouped)) else None,
            "mean_square": max(0.0, moment.square_sum / m) if m else None,
            "content": None, "weight": None, "status": "not_ready", "reasons": []}
        if m + c == 0:
            row["reasons"] = [_reason("no_measurements", "尚无记录", material_id=material.id)]
            rows.append(row)
            continue
        if m == 0:
            row["reasons"].append(_reason("diameter_required", "纯计数需要该类实测直径才能计算含量", material_id=material.id))
        elif m == 1 and method.family == "variance" and not session.grouped:
            row["reasons"].append(_reason("sample_variance_requires_two", "未分组方差法需要至少两条实测直径", material_id=material.id))
        if material.density is None:
            row["reasons"].append(_reason("density_required", "请填写并确认材料密度", material_id=material.id))
        elif material.density <= 0:
            row["reasons"].append(_reason("density_required", "当前密度为零，请确认并填写正密度", material_id=material.id))
        if method.family in ("bast_upper", "bast_mid") and material.family not in ("cotton", "flax", "hemp", "ramie", "apocynum"):
            row["reasons"].append(_reason("unsupported_material", "此方法没有该材料的棉麻修正规则", severity="invalid", material_id=material.id))
        if method.family == "empirical" and material.family not in ("cotton", "ramie", "hemp", "flax"):
            row["reasons"].append(_reason("unsupported_material", "FZ/T 30003 仅支持棉、苎麻、大麻和亚麻", severity="invalid", material_id=material.id))
        row["status"] = _status(row["reasons"])
        reasons.extend(row["reasons"])
        if row["status"] == "ready":
            n, rho, mu, q = m + c, material.density, moment.mean, row["mean_square"]
            if method.family == "variance":
                quantity = mu * mu + moment.variance
            elif method.family in ("mean_square", "empirical"):
                quantity = mu * mu
            elif method.family == "square_mean":
                quantity = q
            elif material.family == "ramie":
                quantity = (aggregate.ramie_upper_square_sum if method.family == "bast_upper"
                            else aggregate.ramie_mid_square_sum) / m
            else:
                quantity = {"cotton": 0.2939, "flax": 0.42, "hemp": 0.35, "apocynum": 0.39}[material.family] * q
            row["weight"] = quantity * n * rho
            if not math.isfinite(row["weight"]):
                reason = _reason("numeric_overflow", "数据超出含量计算范围", severity="invalid", material_id=material.id)
                row["weight"], row["status"] = None, "invalid"
                row["reasons"].append(reason)
                reasons.append(reason)
        rows.append(row)
    active = [row for row in rows if row["total_count"] > 0]
    if not active:
        reasons.append(_reason("no_measurements", "当前方向尚无实测记录"))
    if method.family == "empirical" and len(active) > 1 and active[0]["family"] != "cotton":
        reasons.append(_reason("cotton_first_required", "FZ/T 30003 混纺计算须将棉列为首个组分", severity="invalid"))
    status = _status(reasons, bool(active))
    if status == "ready":
        denominator = math.fsum(row["weight"] for row in active)
        if not math.isfinite(denominator) or denominator <= 0:
            reasons.append(_reason("invalid_denominator", "含量权重总和必须为正有限数值", severity="invalid"))
            status = "invalid"
        else:
            for row in active:
                row["content"] = row["weight"] / denominator * 100.0
            if method.family == "empirical":
                for row in active[1:]:
                    x = row["content"]
                    if row["family"] == "hemp":
                        x = 1.3402 * x - 0.0034 * x * x
                    elif row["family"] == "flax":
                        x = 1.3731 * x - 0.0037 * x * x
                    row["content"] = min(100.0, max(0.0, x))
                residual = 100.0 - math.fsum(row["content"] for row in active[1:])
                if residual < -1e-10:
                    reasons.append(_reason("negative_residual", "经验修正导致首组分为负，请复核方法与材料组合", severity="invalid"))
                    status = "invalid"
                    for row in active:
                        row["content"] = None
                else:
                    active[0]["content"] = max(0.0, residual)
    return {"status": status, "reasons": reasons, "categories": rows,
            "measured_count": sum(row["measured_count"] for row in rows),
            "extra_count": sum(row["extra_count"] for row in rows),
            "total_count": sum(row["total_count"] for row in rows)}


def _mixed(operator: "Operator", warp: dict[str, Any], weft: dict[str, Any]) -> dict[str, Any]:
    reasons: list[dict[str, str]] = []
    total = operator.warp_weight + operator.weft_weight
    if total <= 0 or not math.isfinite(total):
        reasons.append(_reason("weights_required", "织物汇总需要有效的经纬重量"))
    for weight, result in ((operator.warp_weight, warp), (operator.weft_weight, weft)):
        if weight > 0 and result["status"] != "ready":
            reasons.extend(result["reasons"])
    status = _status(reasons)
    maps = [{row["id"]: row for row in result["categories"] if row["total_count"] > 0} for result in (warp, weft)]
    ids = list(dict.fromkeys([*maps[0], *maps[1]]))
    rows = []
    for key in ids:
        example = maps[0].get(key) or maps[1][key]
        content = None
        if status == "ready":
            content = 0.0
            for weight, mapping in zip((operator.warp_weight, operator.weft_weight), maps):
                if weight > 0 and key in mapping:
                    content += mapping[key]["content"] * (weight / total)
        rows.append({"id": key, "name": example["name"], "family": example["family"], "content": content})
    return {"status": status, "reasons": reasons, "categories": rows}


def _report(results: list[dict[str, Any]], profile: str) -> dict[str, Any]:
    reasons = [reason for result in results if result["status"] != "ready" for reason in result["reasons"]]
    if not results:
        reasons.append(_reason("operators_required", "请先建立操作者并测量"))
    status = _status(reasons, bool(results))
    maps = [{row["id"]: row for row in result["categories"] if row.get("total_count", 1) > 0} for result in results]
    ids = list(dict.fromkeys(key for mapping in maps for key in mapping))
    rows = []
    for key in ids:
        example = next(mapping[key] for mapping in maps if key in mapping)
        raw = math.fsum(mapping[key]["content"] if key in mapping else 0.0 for mapping in maps) / len(results) if status == "ready" else None
        means = [mapping[key]["mean"] for mapping in maps if key in mapping and mapping[key].get("mean") is not None]
        rows.append({"id": key, "name": example["name"], "family": example["family"],
                     "raw_content": raw, "content": round_cu(raw, 2) if raw is not None else None,
                     "mean": math.fsum(means) / len(means) if means else None})
    if status == "ready" and rows:
        if profile == "decolorization_free":
            for row in rows[1:]:
                row["content"] = _mt_round(row["content"])
            # The first worksheet cell references the *final* neighbouring
            # cells. Its residual must therefore follow their one-decimal pass.
            rows[0]["content"] = _mt_round(100.0 - math.fsum(row["content"] for row in rows[1:]))
        else:
            rows[0]["content"] = 100.0 - math.fsum(row["content"] for row in rows[1:])
    return {"status": status, "reasons": reasons, "categories": rows,
            "operator_count": len(results), "decimals": 1 if profile == "decolorization_free" else 2}


def summarize(session: "ExperimentSession") -> dict[str, Any]:
    """O(operators * materials), independent of historical measurement count.

    Return an owned JSON-compatible snapshot; callers cannot corrupt the cache.
    """
    key = (session.revision, session.method, session.group_width, session.grouped, session.profile)
    if session._summary_cache is not None and session._summary_cache[0] == key:
        return deepcopy(session._summary_cache[1])
    method = resolve_method(session.method)
    operators: dict[str, dict[str, Any]] = {}
    for operator in session.operators.values():
        warp = _direction_summary(session, operator, "warp", method)
        weft = _direction_summary(session, operator, "weft", method)
        operators[operator.id] = {"id": operator.id, "name": operator.name,
            "warp": warp, "weft": weft, "mixed": _mixed(operator, warp, weft),
            "warp_weight": operator.warp_weight, "weft_weight": operator.weft_weight}
    fiber = _report([op["warp"] for op in operators.values()], session.profile)
    fabric = _report([op["mixed"] for op in operators.values()], session.profile)
    result = {"session_id": session.id, "revision": session.revision,
              "calculation_revision": CALCULATION_REVISION, "method": method.label,
              "method_id": method.id, "grouped": session.grouped, "group_width": session.group_width,
              "profile": session.profile, "operators": operators, "fiber": fiber, "fabric": fabric,
              "status": fiber["status"], "reasons": fiber["reasons"],
              "rule_repairs": ["all_records_and_dynamic_bins", "raw_variance_requires_two_grouped_single_variance_zero",
                               "explicit_missing_input_status", "inclusive_rounding_tolerance_boundary"],
              "standards_compliance_verified": False}
    session._summary_cache = (key, result)
    return deepcopy(result)
