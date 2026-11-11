"""Explicitly saved selection shortcuts, independent of trial parameter snapshots."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from fdm.atomic_io import atomic_write_json


def combinations_path():
    from fdm.settings import settings_directory
    return settings_directory() / "fiber-material-combinations.json"


class MaterialCombinations:
    def __init__(self, items=()):
        self._items = []
        for item in items:
            if not isinstance(item, dict) or set(item) != {"name", "material_ids"}:
                raise ValueError("常用组合格式无效")
            self.set(item["name"], item["material_ids"])
        if len(self._items) != len(items):
            raise ValueError("常用组合名称重复")

    def items(self):
        return deepcopy(self._items)

    def get(self, name):
        return next((deepcopy(item) for item in self._items if item["name"] == name), None)

    def set(self, name, material_ids, *, previous_name=None):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("组合名称不能为空")
        name = name.strip()
        if (not isinstance(material_ids, (list, tuple)) or not 1 <= len(material_ids) <= 9
                or not all(isinstance(key, str) and key for key in material_ids)
                or len(set(material_ids)) != len(material_ids)):
            raise ValueError("组合须包含一至九个不重复的类别")
        existing = next((i for i, item in enumerate(self._items)
                         if item["name"].casefold() == name.casefold()), None)
        previous = next((i for i, item in enumerate(self._items) if item["name"] == previous_name), None)
        if previous_name is not None and previous is None:
            raise ValueError("常用组合已不存在")
        if previous is not None and existing is not None and previous != existing:
            raise ValueError("组合名称已存在")
        index = previous if previous is not None else existing
        item = {"name": name, "material_ids": list(material_ids)}
        if index is None:
            self._items.append(item)
        else:
            self._items[index] = item

    def delete(self, name):
        self._items = [item for item in self._items if item["name"] != name]

    def save(self, path=None):
        return atomic_write_json(path or combinations_path(), {"version": 1, "combinations": self.items()})

    @classmethod
    def load(cls, path=None):
        path = Path(path or combinations_path())
        if not path.exists():
            return cls()
        value = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(value, dict) or value.get("version") != 1
                or not isinstance(value.get("combinations"), list)):
            raise ValueError("常用组合文件格式无效")
        return cls(value["combinations"])
