"""Searchable material selection and an independent library draft editor."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from uuid import uuid4
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMenu, QMessageBox, QPushButton, QStackedWidget, QToolButton, QVBoxLayout, QWidget)
from fdm.live_experiment.material_catalog import MaterialCatalog
from fdm.live_experiment.material_combinations import MaterialCombinations
from fdm.live_experiment.models import DiameterRangePolicy, MaterialSnapshot, built_in_materials
from fdm.live_experiment.trial_defaults import save_diameter_range_default
from fdm.ui.material_components import (MaterialParameterEditor, ResponsiveMaterialPanes,
    fit_dialog, material_summary, material_table, number_text, range_text, source_text, table_item)


def _optional_number(editor):
    return float(editor.text().strip()) if editor.text().strip() else None


def _label(text, parent=None):
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


def _button(text, callback, layout):
    button = QPushButton(text)
    button.clicked.connect(callback)
    layout.addWidget(button)
    return button


class MaterialEditDialog(QDialog):
    """Compatibility wrapper; the two primary windows embed this editor."""
    def __init__(self, material, catalog, parent=None, *, scope="类别库默认"):
        super().__init__(parent)
        self.setWindowTitle(f"{scope} · {material.name}")
        layout = QVBoxLayout(self)
        self.editor = MaterialParameterEditor(self)
        self.editor.load(material, catalog, scope=scope)
        layout.addWidget(self.editor)
        for key in ("name", "aliases", "density", "lower", "upper", "below", "above", "source", "error"):
            setattr(self, key, getattr(self.editor, key))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        fit_dialog(self, 470, 500)

    def material(self):
        return self.editor.material()

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
        self.path, self.catalog = path, MaterialCatalog.load(path)
        self._initial = self.catalog.materials()
        self._builtin = {item.id for item in built_in_materials()}
        self._editing_id, self._refreshing = None, False
        self.setWindowTitle("纤维类别库")
        layout = QVBoxLayout(self)
        layout.addWidget(_label("类别库默认 · 修改只供以后选用，已有试验的参数副本保持不变。", self))
        toolbar = QHBoxLayout()
        _button("新增类别", self.add, toolbar)
        self.copy_button = _button("复制", self.duplicate, toolbar)
        self.delete_button = _button("删除", self.delete, toolbar)
        more = QToolButton(self)
        more.setText("更多")
        more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(more)
        self.restore_action = menu.addAction("恢复当前内置定义", self.restore_selected)
        menu.addAction("恢复全部内置定义…", self.restore)
        more.setMenu(menu)
        toolbar.addWidget(more)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        left, right = QWidget(self), QWidget(self)
        left_layout, right_layout = QVBoxLayout(left), QVBoxLayout(right)
        left_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setContentsMargins(0, 0, 0, 0)
        search_row = QHBoxLayout()
        self.search = QLineEdit(left)
        self.search.setPlaceholderText("搜索名称或别名")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        search_row.addWidget(self.search, 1)
        self.filter = QComboBox(left)
        for title, key in (("全部", "all"), ("内置", "builtin"), ("自定义", "custom"), ("待完善", "incomplete")):
            self.filter.addItem(title, key)
        self.filter.currentIndexChanged.connect(self._filter)
        search_row.addWidget(self.filter)
        left_layout.addLayout(search_row)
        self.table = material_table(left)
        self.table.currentCellChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(lambda *_: self.edit())
        self.table.setSortingEnabled(True)
        left_layout.addWidget(self.table, 1)
        self.count = _label("", left)
        left_layout.addWidget(self.count)
        _button("编辑选中类别", self.edit, left_layout)
        self.editor = MaterialParameterEditor(right)
        self.editor.changed.connect(lambda: self.status.setText("草稿尚未保存；保存类别库后供以后选用。"))
        for key in ("name", "aliases", "density", "lower", "upper", "source"):
            getattr(self.editor, key).editingFinished.connect(self._editor_finished)
        right_layout.addWidget(self.editor, 1)
        self.panes = ResponsiveMaterialPanes(left, right, self,
            labels=("类别列表", "详情编辑"), breakpoint=880)
        layout.addWidget(self.panes, 1)
        self.status = _label("", self)
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存类别库")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._refresh()
        fit_dialog(self, 1000, 640)

    def _row_id(self, row):
        item = self.table.item(row, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _select_id(self, key):
        for row in range(self.table.rowCount()):
            if self._row_id(row) == key:
                blocked = self.table.blockSignals(True)
                self.table.setCurrentCell(row, 0)
                self.table.selectRow(row)
                self.table.blockSignals(blocked)
                return

    def _flush_editor(self):
        if self.editor.original is None:
            return True
        try:
            material = self.editor.material()
            self.catalog.upsert(material)
        except (ValueError, TypeError) as error:
            self.editor.error.setText(str(error))
            self.status.setText("请先修正当前类别的参数；取消仍可放弃全部草稿。")
            return False
        self._editing_id = material.id
        return True

    def _selection_changed(self, row, _column, *_):
        if self._refreshing:
            return
        key = self._row_id(row)
        if key is None or key == self._editing_id:
            return
        if not self._flush_editor():
            self._select_id(self._editing_id)
            return
        self._load_editor(key)
        self._refresh(key)

    def _editor_finished(self):
        if self._refreshing or not self._flush_editor():
            return
        material = self.catalog.get(self._editing_id)
        if material is None:
            return
        # Update the draft's row without rebuilding the model beneath the
        # active editor, which would disturb focus or the current text cursor.
        status = "内置" if material.id in self._builtin else "自定义"
        if material.density is None or material.density <= 0:
            status += " · 待确认密度"
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        for row in range(self.table.rowCount()):
            if self._row_id(row) == material.id:
                for column, text in enumerate((material.name, number_text(material.density), range_text(material), status)):
                    self.table.item(row, column).setText(text)
                break
        self.table.setSortingEnabled(sorting)
        self._filter()

    def _load_editor(self, key):
        item = self.catalog.get(key)
        if item is None:
            self.editor.original = None
            self.editor.setEnabled(False)
            self._editing_id = None
        else:
            self.editor.setEnabled(True)
            self.editor.load(item, self.catalog, scope="类别库默认")
            self._editing_id = key
        self.restore_action.setEnabled(key in self._builtin)
        self.copy_button.setEnabled(item is not None)
        self.delete_button.setEnabled(item is not None)

    def _refresh(self, selected=None):
        items = self.catalog.materials()
        selected = selected or self._editing_id or (items[0].id if items else None)
        self._refreshing = True
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(items))
        for row, item in enumerate(items):
            status = "内置" if item.id in self._builtin else "自定义"
            if item.density is None or item.density <= 0:
                status += " · 待确认密度"
            for column, text in enumerate((item.name, number_text(item.density), range_text(item), status)):
                cell = table_item(text, item.id, numeric=column in (1, 2))
                cell.setToolTip(f"{item.name}\n别名：{'、'.join(item.aliases) or '无'}\n{material_summary(item)}\n来源：{source_text(item)}")
                self.table.setItem(row, column, cell)
        self.table.setSortingEnabled(sorting)
        self._select_id(selected)
        self._refreshing = False
        if self.editor.original is None or self._editing_id != selected:
            self._load_editor(selected)
        self._filter()

    def _filter(self, *_):
        query, kind = self.search.text().strip().casefold(), self.filter.currentData()
        visible = 0
        for row in range(self.table.rowCount()):
            material = self.catalog.get(self._row_id(row))
            if material is None:
                continue
            match = query in " ".join((material.name, *material.aliases)).casefold()
            match &= (kind == "all" or kind == "builtin" and material.id in self._builtin
                      or kind == "custom" and material.id not in self._builtin
                      or kind == "incomplete" and (material.density is None or material.density <= 0))
            self.table.setRowHidden(row, not match)
            visible += bool(match)
        self.count.setText(f"显示 {visible} / {self.table.rowCount()} 类")

    def _unique_name(self, base):
        existing = {m.name.casefold() for m in self.catalog.materials()}
        name, suffix = base, 2
        while name.casefold() in existing:
            name, suffix = f"{base} {suffix}", suffix + 1
        return name

    def _start_edit(self, material):
        self.catalog.upsert(material)
        self._load_editor(material.id)
        self._refresh(material.id)
        self.search.clear()
        self.filter.setCurrentIndex(0)
        self.panes.show_right()
        self.editor.name.setFocus()
        self.editor.name.selectAll()

    def add(self):
        if self._flush_editor():
            self._start_edit(MaterialSnapshot(uuid4().hex, self._unique_name("新类别"), provenance={"source": "user"}))

    def edit(self):
        if self._editing_id is not None:
            self.panes.show_right()
            self.editor.name.setFocus()

    def duplicate(self):
        if self._editing_id is None or not self._flush_editor():
            return
        item = self.catalog.get(self._editing_id)
        self._start_edit(replace(item, id=uuid4().hex, name=self._unique_name(item.name + " 副本"), aliases=(),
            provenance={**item.provenance, "source": "user", "copied_from_id": item.id}))

    def delete(self):
        if self._editing_id is None or not self._flush_editor():
            return
        item = self.catalog.get(self._editing_id)
        references = self.catalog.references(item.id)
        note = "已有试验的参数副本不受影响。"
        if references:
            note += "\n下列类别的越限建议将清除：" + "、".join(self.catalog.get(key).name for key in references)
        if QMessageBox.question(self, "删除类别库定义", f"删除 {item.name}？\n{note}") != QMessageBox.StandardButton.Yes:
            return
        self.catalog.delete(item.id, clear_references=True)
        self.editor.original, self._editing_id = None, None
        self._refresh()

    def _restore(self, key=None):
        if not self._flush_editor():
            return
        prompt = "恢复当前类别的内置参数？已有试验不改变。" if key else "恢复全部内置参数并保留自定义类别？已有试验不改变。"
        if QMessageBox.question(self, "恢复内置定义", prompt) != QMessageBox.StandardButton.Yes:
            return
        try:
            self.catalog.restore_builtin(key)
            selected = self._editing_id
            self.editor.original = None
            self._refresh(selected)
        except ValueError as error:
            self.status.setText(str(error))

    def restore_selected(self):
        if self._editing_id in self._builtin:
            self._restore(self._editing_id)

    def restore(self):
        self._restore()

    def accept(self):
        if not self._flush_editor():
            return
        try:
            if self.catalog.materials() != self._initial:
                self.catalog.save(self.path)
        except (OSError, ValueError) as error:
            self.status.setText(f"类别库未保存：{error}")
            return
        super().accept()


class TrialMaterialsDialog(QDialog):
    def __init__(self, session, parent=None, *, catalog=None, in_use=None, suggested_id=None, combinations_path=None):
        super().__init__(parent)
        self.session = session
        self.operator_id, self.direction = session._owner(None, None)
        self.catalog = catalog or MaterialCatalog.load()
        self._initial = tuple(deepcopy(session.operators[self.operator_id].materials[self.direction]))
        self._materials = list(self._initial)
        self._in_use = in_use or (lambda: set())
        self.combinations_path = combinations_path
        self._refreshing, self._editing_id = False, None
        self.setWindowTitle("选择本次纤维")
        layout = QVBoxLayout(self)
        heading = QHBoxLayout()
        direction = "经向" if self.direction == "warp" else "纬向"
        heading.addWidget(_label(f"{session.operators[self.operator_id].name} · {direction} · 本试验参数副本", self), 1)
        self.combination_button = QToolButton(self)
        self.combination_button.setText("常用组合")
        self.combination_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.combination_menu = QMenu(self.combination_button)
        self.combination_menu.aboutToShow.connect(self._build_combination_menu)
        self.combination_button.setMenu(self.combination_menu)
        heading.addWidget(self.combination_button)
        layout.addLayout(heading)
        left, right = QWidget(self), QWidget(self)
        left_layout, right_layout = QVBoxLayout(left), QVBoxLayout(right)
        left_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit(left)
        self.search.setPlaceholderText("搜索名称或别名，勾选即可加入")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        left_layout.addWidget(self.search)
        self.table = material_table(left, selectable=True)
        self.table.itemChanged.connect(self._checked)
        self.table.cellDoubleClicked.connect(self._candidate_activated)
        self.table.setSortingEnabled(True)
        left_layout.addWidget(self.table, 1)
        self.catalog_count = _label("", left)
        left_layout.addWidget(self.catalog_count)
        self.right_stack = QStackedWidget(right)
        selection = QWidget(self.right_stack)
        selected_layout = QVBoxLayout(selection)
        selected_layout.setContentsMargins(0, 0, 0, 0)
        self.selected_title = _label("", selection)
        selected_layout.addWidget(self.selected_title)
        self.list = QListWidget(selection)
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list.currentRowChanged.connect(self._selected_changed)
        self.list.itemDoubleClicked.connect(lambda *_: self.edit())
        self.list.model().rowsMoved.connect(self._rows_moved)
        selected_layout.addWidget(self.list)
        row = QHBoxLayout()
        self.edit_button = _button("本次参数", self.edit, row)
        self.remove_button = _button("移除", self.remove, row)
        self.up_button = _button("↑", lambda: self.move(-1), row)
        self.up_button.setAccessibleName("上移类别")
        self.down_button = _button("↓", lambda: self.move(1), row)
        self.down_button.setAccessibleName("下移类别")
        selected_layout.addLayout(row)
        self.summary = _label("勾选左侧类别后，可在此调整数字选类顺序。", selection)
        selected_layout.addWidget(self.summary)
        selected_layout.addStretch()
        self.right_stack.addWidget(selection)
        self.parameter_page = QWidget(self.right_stack)
        parameter_layout = QVBoxLayout(self.parameter_page)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        _button("← 返回已选类别", self._back, parameter_layout)
        self.editor = MaterialParameterEditor(self.parameter_page)
        parameter_layout.addWidget(self.editor, 1)
        self.right_stack.addWidget(self.parameter_page)
        right_layout.addWidget(self.right_stack)
        self.panes = ResponsiveMaterialPanes(left, right, self)
        layout.addWidget(self.panes, 1)
        self.count = _label("", self)
        layout.addWidget(self.count)
        layout.addWidget(_label("顺序对应数字选类；已关联 Excel 的原始数据列将同步调整。", self))
        self.status = _label("", self)
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("应用到本试验")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        for key, delta in (("Alt+Up", -1), ("Alt+Down", 1)):
            shortcut = QShortcut(QKeySequence(key), self.list)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda value=delta: self.move(value))
        try:
            self.combinations = MaterialCombinations.load(combinations_path)
        except (OSError, ValueError) as error:
            self.combinations = None
            self.status.setText(f"常用组合暂不可用，仍可手动挑选：{error}")
        self._refresh()
        if suggested_id:
            self.focus_material(suggested_id)
        self._lock_timer = QTimer(self)
        self._lock_timer.setInterval(300)
        self._lock_timer.timeout.connect(self._refresh_locks)
        self._last_locks = self._locks()
        self._lock_timer.start()
        fit_dialog(self, 940, 620)

    def _locks(self):
        result = {key: "待确认或正在计算的测量仍引用此类别" for key in self._in_use()}
        for record in (*self.session.measurements.values(), *self.session.count_adjustments.values()):
            if record.operator_id == self.operator_id and record.direction == self.direction:
                result[record.material_id] = "已有记录或补计；请先完成归类或移除记录"
        return result

    def _refresh_locks(self):
        locks = self._locks()
        if locks != self._last_locks:
            self._last_locks = locks
            self._refresh(self._current_id())

    def _current_id(self):
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def materials(self):
        return tuple(deepcopy(self._materials))

    def _flush_editor(self):
        if self._editing_id is None:
            return True
        try:
            material = self.editor.material()
            if any(m.id != material.id and m.name == material.name for m in self._materials):
                raise ValueError("同一方向类别名称不能重复")
            index = next(i for i, m in enumerate(self._materials) if m.id == material.id)
            self._materials[index] = material
        except (ValueError, TypeError, StopIteration) as error:
            self.editor.error.setText(str(error))
            self.panes.show_right()
            return False
        return True

    def _back(self):
        if self._editing_id is None:
            return True
        if not self._flush_editor():
            return False
        selected, self._editing_id = self._editing_id, None
        self.right_stack.setCurrentIndex(0)
        self._refresh(selected)
        return True

    def _refresh(self, selected=None):
        selected = selected or self._current_id() or (self._materials[0].id if self._materials else None)
        self._refreshing = True
        try:
            self.list.clear()
            locks = self._locks()
            for index, material in enumerate(self._materials):
                extra = " · 使用中" if material.id in locks else ""
                item = QListWidgetItem(f"{index + 1} · {material.name}{extra}\n{number_text(material.density)} g/cm³ · {range_text(material)} μm")
                item.setData(Qt.ItemDataRole.UserRole, material.id)
                item.setToolTip(locks.get(material.id, "拖动调整顺序，Alt+↑／↓ 也可移动"))
                self.list.addItem(item)
                if material.id == selected:
                    self.list.setCurrentItem(item)
            self.list.setFixedHeight(min(252, max(72, len(self._materials) * 48 + 4)))
            self.selected_title.setText(f"已选顺序 · {len(self._materials)}/9")
            self.count.setText(f"已选 {len(self._materials)}/9 类" + (" · " + "、".join(m.name for m in self._materials) if self._materials else ""))
            candidates = list(self.catalog.materials())
            known = {m.id for m in candidates}
            candidates += [m for m in self._materials if m.id not in known]
            selected_by_id = {m.id: m for m in self._materials}
            sorting = self.table.isSortingEnabled()
            self.table.setSortingEnabled(False)
            self.table.setRowCount(len(candidates))
            for row, candidate in enumerate(candidates):
                material = selected_by_id.get(candidate.id, candidate)
                checked = material.id in selected_by_id
                cell = table_item("", material.id)
                flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                if not (checked and material.id in locks) and (checked or len(self._materials) < 9):
                    flags |= Qt.ItemFlag.ItemIsUserCheckable
                cell.setFlags(flags)
                cell.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                cell.setToolTip(locks.get(material.id, "最多九类" if not checked and len(self._materials) >= 9 else "勾选加入本次"))
                self.table.setItem(row, 0, cell)
                for column, text in enumerate((material.name, number_text(material.density), range_text(material),
                                                "仅本试验" if material.id not in known else ""), start=1):
                    item = table_item(text, material.id, numeric=column in (2, 3))
                    item.setToolTip(f"{material.name}\n别名：{'、'.join(material.aliases) or '无'}\n{material_summary(material)}\n{locks.get(material.id, '')}")
                    self.table.setItem(row, column, item)
            self.table.setSortingEnabled(sorting)
            self._filter()
        finally:
            self._refreshing = False
        self._selected_changed()

    def _filter(self, *_):
        query = self.search.text().strip().casefold()
        local = {m.id: m for m in self._materials}
        visible = 0
        for row in range(self.table.rowCount()):
            key = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            material = local.get(key) or self.catalog.get(key)
            match = query in " ".join((material.name, *material.aliases)).casefold()
            self.table.setRowHidden(row, not match)
            visible += match
        self.catalog_count.setText(f"显示 {visible} / {self.table.rowCount()} 类")

    def _checked(self, item):
        if self._refreshing or item.column() != 0:
            return
        self.set_selected(item.data(Qt.ItemDataRole.UserRole), item.checkState() == Qt.CheckState.Checked)

    def _candidate_activated(self, row, _column):
        if _column == 0:
            return
        cell = self.table.item(row, 0)
        self.set_selected(cell.data(Qt.ItemDataRole.UserRole), cell.checkState() != Qt.CheckState.Checked)

    def set_selected(self, key, selected):
        if not self._back():
            self._refresh()
            return False
        current = next((m for m in self._materials if m.id == key), None)
        reason = None
        if selected and current is None:
            candidate = self.catalog.get(key)
            if candidate is None:
                reason = "此类别已不在类别库中，请重新选择。"
            elif len(self._materials) >= 9:
                reason = "每个方向最多九类，请先移除未使用的类别。"
            elif any(m.name == candidate.name for m in self._materials):
                reason = "本次已有同名类别，请先核对类别来源。"
            else:
                self._materials.append(candidate)
        elif not selected and current is not None:
            reason = self._locks().get(key)
            if reason is None:
                self._materials = [m for m in self._materials if m.id != key]
        self.status.setText(reason or "")
        self._refresh(key)
        return reason is None

    def focus_material(self, key):
        self.search.clear()
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == key:
                self.table.setCurrentCell(row, 1)
                self.table.selectRow(row)
                self.table.scrollToItem(self.table.item(row, 1))
                self.table.setFocus()
                return

    def _selected_changed(self, *_):
        row = self.list.currentRow()
        if self._refreshing:
            return
        selected = 0 <= row < len(self._materials)
        self.edit_button.setEnabled(selected)
        self.remove_button.setEnabled(selected and self._materials[row].id not in self._locks())
        self.up_button.setEnabled(selected and row > 0)
        self.down_button.setEnabled(selected and row < len(self._materials) - 1)
        if selected:
            material = self._materials[row]
            suffix = self._locks().get(material.id, "可修改本次参数；不会改动类别库默认。")
            self.summary.setText(f"{material.name}\n{material_summary(material)}\n{suffix}")
            self.remove_button.setToolTip(self._locks().get(material.id, "从本次选择移除"))
        else:
            self.summary.setText("勾选候选类别后，可在此调整数字选类顺序。")

    def edit(self):
        row = self.list.currentRow()
        if row < 0:
            return
        self._editing_id = self._materials[row].id
        self.editor.load(self._materials[row], self.catalog, scope="本试验参数")
        self.right_stack.setCurrentWidget(self.parameter_page)
        self.panes.show_right()

    def remove(self):
        if self._current_id():
            self.set_selected(self._current_id(), False)

    def move(self, offset):
        row = self.list.currentRow()
        target = row + offset
        if 0 <= row < len(self._materials) and 0 <= target < len(self._materials):
            self._materials[row], self._materials[target] = self._materials[target], self._materials[row]
            self._refresh(self._materials[target].id)

    def _rows_moved(self, *_):
        if self._refreshing:
            return
        mapping = {m.id: m for m in self._materials}
        self._materials = [mapping[self.list.item(row).data(Qt.ItemDataRole.UserRole)] for row in range(self.list.count())]
        QTimer.singleShot(0, self, lambda: self._refresh(self._current_id()))

    def _build_combination_menu(self):
        menu = self.combination_menu
        menu.clear()
        if self.combinations is None:
            menu.addAction("常用组合文件不可用").setEnabled(False)
            return
        entries = self.combinations.items()
        if not entries:
            menu.addAction("尚无常用组合").setEnabled(False)
        for entry in entries:
            menu.addAction("加入“" + entry["name"] + "”", lambda checked=False, name=entry["name"]: self.add_combination(name))
        menu.addSeparator()
        menu.addAction("保存当前组合…", self._save_combination).setEnabled(bool(self._materials))
        menu.addAction("重命名组合…", self._rename_combination).setEnabled(bool(entries))
        menu.addAction("删除组合…", self._delete_combination).setEnabled(bool(entries))

    def add_combination(self, name):
        if self.combinations is None or not self._back():
            return False
        entry = self.combinations.get(name)
        if entry is None:
            self.status.setText("组合已不存在，请重新选择。")
            return False
        candidates = [self.catalog.get(key) for key in entry["material_ids"]]
        if any(m is None for m in candidates):
            self.status.setText(f"组合“{name}”含 {sum(m is None for m in candidates)} 个已不在库中的类别，整组未加入；请重新保存组合。")
            return False
        current = {m.id for m in self._materials}
        additions = [m for m in candidates if m.id not in current]
        if len(self._materials) + len(additions) > 9:
            self.status.setText(f"组合还需加入 {len(additions)} 类，本次已有 {len(current)} 类，超过九类；整组未加入。")
            return False
        if len({m.name for m in [*self._materials, *additions]}) != len(self._materials) + len(additions):
            self.status.setText("组合与本次已有类别重名，整组未加入，请先核对。")
            return False
        self._materials.extend(additions)
        self.status.setText("已加入组合，原有类别的顺序和参数保持不变。" if additions else "本次已包含该组合。")
        self._refresh(additions[-1].id if additions else None)
        return True

    def _write_combinations(self, draft):
        try:
            draft.save(self.combinations_path)
        except (OSError, ValueError) as error:
            self.status.setText(f"常用组合未保存：{error}")
            return False
        self.combinations = draft
        self.status.setText("常用组合已保存；仅记忆类别和顺序，不含参数。")
        return True

    def save_combination(self, name):
        if self.combinations is None or not self._back():
            return False
        if any(self.catalog.get(m.id) is None for m in self._materials):
            self.status.setText("本次含仅试验中存在的类别，请先在类别库建立定义，再保存组合。")
            return False
        draft = MaterialCombinations(self.combinations.items())
        try:
            draft.set(name, [m.id for m in self._materials])
        except ValueError as error:
            self.status.setText(str(error))
            return False
        return self._write_combinations(draft)

    def _save_combination(self):
        name, ok = QInputDialog.getText(self, "保存常用组合", "组合名称（只记忆类别和顺序，不包含参数）")
        if not ok:
            return
        if any(item["name"].casefold() == name.strip().casefold() for item in self.combinations.items()):
            if QMessageBox.question(self, "替换常用组合", f"替换已有组合“{name.strip()}”？") != QMessageBox.StandardButton.Yes:
                return
        self.save_combination(name)

    def _choose_combination(self, title):
        return QInputDialog.getItem(self, title, "选择组合", [item["name"] for item in self.combinations.items()], 0, False)

    def _rename_combination(self):
        old, ok = self._choose_combination("重命名常用组合")
        if not ok:
            return
        name, ok = QInputDialog.getText(self, "重命名常用组合", "新名称", text=old)
        if not ok:
            return
        draft = MaterialCombinations(self.combinations.items())
        try:
            draft.set(name, draft.get(old)["material_ids"], previous_name=old)
        except ValueError as error:
            self.status.setText(str(error))
            return
        self._write_combinations(draft)

    def _delete_combination(self):
        name, ok = self._choose_combination("删除常用组合")
        if ok and QMessageBox.question(self, "删除常用组合", f"删除“{name}”？本试验选择不变。") == QMessageBox.StandardButton.Yes:
            draft = MaterialCombinations(self.combinations.items())
            draft.delete(name)
            self._write_combinations(draft)

    def accept(self):
        if not self._flush_editor():
            return
        if set(self._locks()) - {m.id for m in self._materials}:
            self.status.setText("当前已有记录或待确认测量引用被移除的类别，请恢复选择后再应用。")
            return
        current = self.session.operators.get(self.operator_id)
        if current is None or tuple(current.materials[self.direction]) != self._initial:
            self.status.setText("本试验类别已在后台变化，请取消并重新打开，避免覆盖新设置。")
            return
        super().accept()


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
