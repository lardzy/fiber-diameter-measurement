"""Library defaults and trial snapshots have deliberately separate edit dialogs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from uuid import uuid4

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QVBoxLayout)

from fdm.live_experiment.material_catalog import MaterialCatalog
from fdm.live_experiment.models import DiameterRangePolicy, MaterialSnapshot
from fdm.live_experiment.trial_defaults import save_diameter_range_default


def _optional_number(editor):
    text = editor.text().strip()
    return float(text) if text else None


class MaterialEditDialog(QDialog):
    def __init__(self, material, catalog, parent=None, *, scope="类别库默认"):
        super().__init__(parent)
        self.original = deepcopy(material)
        self.setWindowTitle(f"{scope} · {material.name}")
        self.resize(470, 460)
        layout = QVBoxLayout(self)
        note = QLabel("类别库默认只影响之后选用的类别；本试验参数只修改当前操作者、当前方向。", self)
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        layout.addLayout(form)
        self.name = QLineEdit(material.name, self)
        self.aliases = QLineEdit("，".join(material.aliases), self)
        self.density = QLineEdit("" if material.density is None else str(material.density), self)
        self.lower = QLineEdit("" if material.lower_limit is None else str(material.lower_limit), self)
        self.upper = QLineEdit("" if material.upper_limit is None else str(material.upper_limit), self)
        self.lower.setPlaceholderText("不限")
        self.upper.setPlaceholderText("不限")
        form.addRow("名称", self.name)
        form.addRow("别名（逗号分隔）", self.aliases)
        form.addRow("密度（g/cm³）", self.density)
        form.addRow("直径下限（μm）", self.lower)
        form.addRow("直径上限（μm）", self.upper)
        self.below, self.above = QComboBox(self), QComboBox(self)
        for combo, value in ((self.below, material.lower_suggestion_id), (self.above, material.upper_suggestion_id)):
            combo.addItem("不建议改类", None)
            for candidate in catalog.materials():
                if candidate.id != material.id:
                    combo.addItem(candidate.name, candidate.id)
            if value is not None and combo.findData(value) < 0:
                combo.addItem(f"{value}（不在当前类别库）", value)
            combo.setCurrentIndex(max(0, combo.findData(value)))
        form.addRow("低于下限时建议", self.below)
        form.addRow("高于上限时建议", self.above)
        self.source = QLineEdit(str(material.provenance.get("parameter_source", material.provenance.get("source", ""))), self)
        form.addRow("参数来源", self.source)
        self.error = QLabel(self)
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def material(self):
        return replace(self.original, name=self.name.text(),
            aliases=tuple(self.aliases.text().replace("，", ",").split(",")),
            density=_optional_number(self.density), lower_limit=_optional_number(self.lower),
            upper_limit=_optional_number(self.upper), lower_suggestion_id=self.below.currentData(),
            upper_suggestion_id=self.above.currentData(),
            provenance={**self.original.provenance, "parameter_source": self.source.text().strip()})

    def accept(self):
        try:
            self.material()
        except (ValueError, TypeError) as error:
            self.error.setText(str(error))
            return
        super().accept()


class MaterialCatalogDialog(QDialog):
    def __init__(self, parent=None, *, path=None):
        super().__init__(parent)
        self.path = path
        self.catalog = MaterialCatalog.load(path)
        self.setWindowTitle("纤维类别库 · 类别库默认")
        self.resize(650, 540)
        layout = QVBoxLayout(self)
        note = QLabel("管理含量／直径试验可复用的默认定义。已有试验保存自己的参数副本，不随类别库修改。", self)
        note.setWordWrap(True)
        layout.addWidget(note)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("搜索名称或别名")
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)
        self.list = QListWidget(self)
        self.list.itemDoubleClicked.connect(lambda _: self.edit())
        layout.addWidget(self.list, 1)
        row = QHBoxLayout()
        for title, slot in (("添加", self.add), ("复制", self.duplicate), ("编辑", self.edit),
                            ("删除", self.delete), ("恢复内置定义", self.restore)):
            button = QPushButton(title, self)
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh()

    def _selected(self):
        row = self.list.currentItem()
        return self.catalog.get(row.data(Qt.ItemDataRole.UserRole)) if row is not None else None

    def _refresh(self, selected=None):
        self.list.clear()
        for material in self.catalog.materials():
            item = QListWidgetItem(material.name)
            item.setData(Qt.ItemDataRole.UserRole, material.id)
            item.setToolTip(f"密度：{material.density if material.density is not None else '未设置'}\n别名：{'、'.join(material.aliases)}")
            self.list.addItem(item)
            if material.id == selected:
                self.list.setCurrentItem(item)
        self._filter()

    def _filter(self):
        query = self.search.text().strip().casefold()
        for index in range(self.list.count()):
            item = self.list.item(index)
            material = self.catalog.get(item.data(Qt.ItemDataRole.UserRole))
            item.setHidden(query not in " ".join((material.name, *material.aliases)).casefold())

    def _edit(self, material):
        dialog = MaterialEditDialog(material, self.catalog, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.catalog.upsert(dialog.material())
            except ValueError as error:
                self.status.setText(str(error))
                return
            self.status.setText("更改将在点击“保存”后写入类别库。")
            self._refresh(material.id)

    def add(self):
        self._edit(MaterialSnapshot(uuid4().hex, "新类别", provenance={"source": "user"}))

    def edit(self):
        selected = self._selected()
        if selected is not None:
            self._edit(selected)

    def duplicate(self):
        selected = self._selected()
        if selected is not None:
            self._edit(replace(selected, id=uuid4().hex, name=selected.name + " 副本", aliases=(),
                provenance={**selected.provenance, "copied_from_id": selected.id}))

    def delete(self):
        selected = self._selected()
        if selected is None:
            return
        references = self.catalog.references(selected.id)
        note = "已有试验的参数副本不受影响。"
        if references:
            note += "\n以下类别的越限建议将同时清除：" + "、".join(self.catalog.get(key).name for key in references)
        if QMessageBox.question(self, "删除类别库定义", f"删除 {selected.name}？\n{note}") != QMessageBox.StandardButton.Yes:
            return
        self.catalog.delete(selected.id, clear_references=True)
        self._refresh()

    def restore(self):
        if QMessageBox.question(self, "恢复内置定义", "恢复所有内置类别的默认参数，保留自定义类别。已有试验不改变。") != QMessageBox.StandardButton.Yes:
            return
        try:
            self.catalog.restore_builtin()
            self._refresh()
        except ValueError as error:
            self.status.setText(str(error))

    def accept(self):
        try:
            self.catalog.save(self.path)
        except (OSError, ValueError) as error:
            self.status.setText(f"类别库未保存：{error}")
            return
        super().accept()


class TrialMaterialsDialog(QDialog):
    def __init__(self, session, parent=None, *, catalog=None):
        super().__init__(parent)
        self.session = session
        self.operator_id, self.direction = session._owner(None, None)
        self.catalog = catalog or MaterialCatalog.load()
        self._materials = list(deepcopy(session.operators[self.operator_id].materials[self.direction]))
        self.setWindowTitle("选择本次纤维 · 本试验参数")
        self.resize(580, 520)
        layout = QVBoxLayout(self)
        direction = "经向" if self.direction == "warp" else "纬向"
        note = QLabel(f"{session.operators[self.operator_id].name} · {direction} · 最多九类。类别参数保存为本试验副本；列表顺序对应数字选类。", self)
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        self.available = QComboBox(self)
        for item in self.catalog.materials():
            self.available.addItem(item.name, item.id)
        row.addWidget(self.available, 1)
        add = QPushButton("加入本次", self)
        add.clicked.connect(self.add)
        row.addWidget(add)
        layout.addLayout(row)
        self.list = QListWidget(self)
        self.list.itemDoubleClicked.connect(lambda _: self.edit())
        layout.addWidget(self.list, 1)
        row = QHBoxLayout()
        for title, slot in (("本试验参数", self.edit), ("移除", self.remove),
                            ("上移", lambda: self.move(-1)), ("下移", lambda: self.move(1))):
            button = QPushButton(title, self)
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh()

    def materials(self):
        return tuple(deepcopy(self._materials))

    def _refresh(self, row=-1):
        self.list.clear()
        for index, item in enumerate(self._materials):
            limits = f"{item.lower_limit if item.lower_limit is not None else '不限'} ～ {item.upper_limit if item.upper_limit is not None else '不限'} μm"
            self.list.addItem(f"{index + 1}  {item.name}    密度 {item.density}    {limits}")
        self.list.setCurrentRow(row)

    def add(self):
        item = self.catalog.get(self.available.currentData())
        if item is None:
            return
        if len(self._materials) >= 9:
            self.status.setText("每个方向最多九类。")
        elif any(existing.id == item.id or existing.name == item.name for existing in self._materials):
            self.status.setText("本次已加入此类别，修改参数请选中列表中的类别。")
        else:
            self._materials.append(item)
            self._refresh(len(self._materials) - 1)

    def edit(self):
        row = self.list.currentRow()
        if row < 0:
            return
        dialog = MaterialEditDialog(self._materials[row], self.catalog, self, scope="本试验参数")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            item = dialog.material()
            if any(index != row and other.name == item.name for index, other in enumerate(self._materials)):
                self.status.setText("同一方向类别名称不能重复。")
                return
            self._materials[row] = item
            self._refresh(row)

    def remove(self):
        row = self.list.currentRow()
        if row < 0:
            return
        key = self._materials[row].id
        records = (*self.session.measurements.values(), *self.session.count_adjustments.values())
        if any(record.operator_id == self.operator_id and record.direction == self.direction and record.material_id == key for record in records):
            self.status.setText("此类别已有记录或补计，请先完成归类或移除记录，再移除此类别。")
            return
        del self._materials[row]
        self._refresh(min(row, len(self._materials) - 1))

    def move(self, offset):
        row = self.list.currentRow()
        target = row + offset
        if row >= 0 and 0 <= target < len(self._materials):
            self._materials[row], self._materials[target] = self._materials[target], self._materials[row]
            self._refresh(target)


class DiameterRangeDialog(QDialog):
    def __init__(self, session, parent=None, *, defaults_file=None):
        super().__init__(parent)
        self.defaults_file = defaults_file
        self.setWindowTitle("本场直径试验 · 允许范围")
        layout = QVBoxLayout(self)
        note = QLabel("空白表示不限。等于边界允许通过；超出范围时由操作者保留或放弃，仍可继续测量。", self)
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.lower, self.upper = QLineEdit(self), QLineEdit(self)
        for editor, value in ((self.lower, session.diameter_range_policy.lower_limit_um),
                              (self.upper, session.diameter_range_policy.upper_limit_um)):
            editor.setPlaceholderText("不限")
            editor.setText("" if value is None else str(value))
        form.addRow("下限（μm）", self.lower)
        form.addRow("上限（μm）", self.upper)
        layout.addLayout(form)
        self.save_default = QPushButton("设为新直径试验默认", self)
        self.save_default.clicked.connect(self._save_default)
        layout.addWidget(self.save_default)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def policy(self):
        return DiameterRangePolicy(_optional_number(self.lower), _optional_number(self.upper))

    def _save_default(self):
        try:
            save_diameter_range_default(self.policy(), path=self.defaults_file)
            self.status.setText("已保存为新直径试验默认，其他已有试验不改变。")
        except (ValueError, OSError) as error:
            self.status.setText(f"默认值未保存：{error}")

    def accept(self):
        try:
            self.policy()
        except ValueError as error:
            self.status.setText(str(error))
            return
        super().accept()
