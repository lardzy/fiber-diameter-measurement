"""Reusable fiber definitions. Trial snapshots never retain references to this store."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from uuid import uuid4

from fdm.atomic_io import atomic_write_json
from .models import MaterialSnapshot, built_in_materials


def catalog_path() -> Path:
    from fdm.settings import settings_directory
    return settings_directory() / "fiber-materials.json"


class MaterialCatalog:
    def __init__(self, materials=None):
        items = built_in_materials() if materials is None else tuple(materials)
        if len({item.id for item in items}) != len(items):
            raise ValueError("类别库存在重复 ID")
        self._items = {item.id: deepcopy(item) for item in items}
        self._validate_names()

    def _validate_names(self):
        names = [item.name.casefold() for item in self._items.values()]
        if len(set(names)) != len(names):
            raise ValueError("类别库名称不能重复")

    def _validate_references(self):
        for item in self._items.values():
            for target in (item.lower_suggestion_id, item.upper_suggestion_id):
                if target == item.id or (target is not None and target not in self._items):
                    raise ValueError("类别库存在无效的越限建议引用，请恢复相关内置定义或修改建议")

    def materials(self) -> tuple[MaterialSnapshot, ...]:
        return tuple(deepcopy(tuple(self._items.values())))

    def get(self, material_id) -> MaterialSnapshot | None:
        return deepcopy(self._items.get(material_id))

    def upsert(self, material: MaterialSnapshot) -> None:
        if any(item.id != material.id and item.name.casefold() == material.name.casefold()
               for item in self._items.values()):
            raise ValueError("类别库名称不能重复")
        for target in (material.lower_suggestion_id, material.upper_suggestion_id):
            if target == material.id:
                raise ValueError("越限建议不能指向类别自身")
            if target is not None and target not in self._items:
                raise ValueError("越限建议必须指向类别库中已有的类别")
        self._items[material.id] = deepcopy(material)

    def duplicate(self, material_id: str, name: str) -> MaterialSnapshot:
        original = self._items[material_id]
        item = replace(original, id=uuid4().hex, name=name, aliases=(),
                       provenance={**original.provenance, "copied_from_id": original.id})
        self.upsert(item)
        return deepcopy(item)

    def references(self, material_id: str) -> tuple[str, ...]:
        return tuple(item.id for item in self._items.values()
                     if material_id in (item.lower_suggestion_id, item.upper_suggestion_id))

    def delete(self, material_id: str, *, clear_references: bool = False) -> None:
        references = self.references(material_id)
        if references and not clear_references:
            raise ValueError("此类别仍被越限建议引用，请先修改建议或明确清除引用")
        if references:
            for key in references:
                item = self._items[key]
                self._items[key] = replace(item,
                    lower_suggestion_id=None if item.lower_suggestion_id == material_id else item.lower_suggestion_id,
                    upper_suggestion_id=None if item.upper_suggestion_id == material_id else item.upper_suggestion_id)
        self._items.pop(material_id)

    def restore_builtin(self, material_id: str | None = None) -> None:
        """Restore definitions without removing user-defined IDs or trial snapshots."""
        definitions = built_in_materials()
        if material_id is not None:
            definitions = tuple(item for item in definitions if item.id == material_id)
            if not definitions:
                raise ValueError("此类别没有内置定义")
        draft = dict(self._items)
        for item in definitions:
            draft[item.id] = item
        checked = MaterialCatalog(draft.values())
        checked._validate_references()
        self._items = checked._items

    def save(self, path: str | Path | None = None) -> Path:
        self._validate_references()
        return atomic_write_json(path or catalog_path(),
            {"version": 1, "materials": [item.to_dict() for item in self._items.values()]})

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MaterialCatalog":
        path = Path(path or catalog_path())
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("materials"), list):
            raise ValueError("类别库文件格式无效")
        try:
            catalog = cls(MaterialSnapshot.from_dict(item) for item in payload["materials"])
        except (AttributeError, TypeError, KeyError) as error:
            raise ValueError("类别库定义字段无效") from error
        catalog._validate_references()
        return catalog
