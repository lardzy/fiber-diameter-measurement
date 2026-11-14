"""Session-only range decisions; no frame or rule is application/project preference."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass


def range_side(value, limits):
    low, high = limits.get("lower_limit_um"), limits.get("upper_limit_um")
    if low is not None and value < low:
        return "below"
    if high is not None and value > high:
        return "above"
    return None


def limits_identity(limits):
    # A rename does not change the safety condition; a changed limit does.
    return (limits.get("material_id"), limits.get("lower_limit_um"), limits.get("upper_limit_um"))


@dataclass(slots=True)
class RangeDecision:
    intent: object
    result: dict
    original_material_id: str
    direction: str | None
    source_range: dict
    batch_id: str
    requires_explicit_keep: bool = False
    resolved: bool = False
    discarded: bool = False


class RangeRuleStore:
    """Rules owned by one live session, addressed explicitly by session identity."""
    def __init__(self):
        self._rules = {}

    def list_rules(self, session_id):
        return [deepcopy(value) for (trial, _, _), value in self._rules.items() if trial == session_id]

    def remove(self, session_id, key):
        self._rules.pop((session_id, *key), None)

    def clear(self, session_id):
        for key in list(self._rules):
            if key[0] == session_id:
                del self._rules[key]

    def invalidate_materials(self, session_id, material_ids):
        affected = set(material_ids)
        removed = 0
        for key, rule in list(self._rules.items()):
            if key[0] == session_id and (rule["original_material_id"] in affected
                    or rule.get("target_material_id") in affected):
                del self._rules[key]
                removed += 1
        return removed

    def remember(self, session_id, decision, action, *, target_range=None):
        key = (decision.original_material_id, decision.direction)
        self._rules[(session_id, *key)] = dict(key=key, original_material_id=key[0], direction=key[1],
            action=action, target_material_id=(target_range or {}).get("material_id"),
            source_range=deepcopy(decision.source_range), target_range=deepcopy(target_range))

    def matching(self, session, decision):
        key = (session.id, decision.original_material_id, decision.direction)
        rule = self._rules.get(key)
        if rule is None:
            return None
        intent = decision.intent
        try:
            source = session.measurement_range(decision.original_material_id,
                operator_id=intent.operator_id, direction=intent.direction)
            valid = limits_identity(source) == limits_identity(rule["source_range"])
            target = None
            if rule["action"] == "reclassify":
                target = session.measurement_range(rule["target_material_id"],
                    operator_id=intent.operator_id, direction=intent.direction)
                valid = valid and limits_identity(target) == limits_identity(rule["target_range"])
                valid = valid and range_side(decision.result["diameter_um"], target) is None
            valid = valid and range_side(decision.result["diameter_um"], source) == decision.direction
        except ValueError:
            valid = False
        if not valid:
            del self._rules[key]
            return None
        return deepcopy(rule)
