from __future__ import annotations

from collections.abc import Iterable, Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


DEFAULT_OBJECT_SNAP_KINDS = (
    "point",
    "endpoint",
    "midpoint",
    "center",
    "quadrant",
    "intersection",
)


_SNAP_KIND_LABELS = {
    "point": "普通点",
    "endpoint": "端点",
    "midpoint": "中点",
    "center": "圆心",
    "quadrant": "象限点",
    "intersection": "交点",
    "nearest": "最近点",
    "perpendicular": "垂足",
    "tangent": "切点",
}


_CONSTRUCTION_KIND_LABELS = {
    "point": "自由点",
    "free_point": "自由点",
    "midpoint": "中点",
    "intersection": "交点",
    "line": "辅助线",
    "segment": "辅助线段",
    "ray": "射线",
    "infinite_line": "无限直线",
    "circle": "圆",
    "circle_center_radius": "圆心—半径圆",
    "circle_center_diameter": "圆心—直径圆",
    "circle_two_point": "两点直径圆",
    "circle_three_point": "三点圆",
    "parallel": "平行线",
    "parallel_through_point": "过点平行线",
    "offset_parallel": "定距平行线",
    "parallel_array": "平行阵列",
    "perpendicular": "垂线",
    "perpendicular_bisector": "垂直平分线",
    "tangent": "切线",
    "concentric_circle": "同心圆",
    "offset_circle": "偏移圆",
    "point_circle_tangent": "点到圆切线",
    "common_tangent": "两圆公切线",
    "tangent_tangent_radius_circle": "相切—相切—半径圆",
    "three_tangent_circle": "三相切圆",
    "construction": "辅助对象",
}


def construction_kind_token(entity: object) -> str:
    definition = getattr(entity, "definition", None)
    token = getattr(definition, "kind", None)
    if token is None:
        token = getattr(definition, "definition_kind", None)
    if token is None:
        token = type(definition).__name__
    normalized = str(token or "construction").strip().lower()
    normalized = normalized.removesuffix("definition").replace("definition", "")
    normalized = normalized.replace("def", "").strip("_")
    return normalized or "construction"


def construction_kind_label(entity: object) -> str:
    token = construction_kind_token(entity)
    definition = getattr(entity, "definition", None)
    if token == "line":
        axis = str(
            getattr(getattr(definition, "axis_constraint", None), "value", "")
        )
        if axis == "horizontal":
            return "水平辅助线"
        if axis == "vertical":
            return "垂直辅助线"
        extent = str(getattr(getattr(definition, "extent", None), "value", ""))
        return {
            "segment": "辅助线段",
            "ray": "射线",
            "infinite": "无限直线",
        }.get(extent, "辅助线")
    if token == "common_tangent":
        mode = str(getattr(getattr(definition, "mode", None), "value", ""))
        return "两圆内公切线" if mode == "internal" else "两圆外公切线"
    return _CONSTRUCTION_KIND_LABELS.get(token, token or "辅助对象")


class ObjectSnapStatusButton(QToolButton):
    """Persistent, mouse-first object-snap control for the status bar."""

    enabledChanged = Signal(bool)
    kindsChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("objectSnapStatusButton")
        self.setCheckable(True)
        self.setChecked(True)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setToolTip("开启或关闭几何对象捕捉；箭头可选择捕捉类型")
        self.setAccessibleName("对象捕捉，可展开选择捕捉类型")
        self._actions: dict[str, QAction] = {}
        menu = self._build_menu()
        self.setMenu(menu)
        self.toggled.connect(self._on_toggled)
        self._refresh_text()

    def _build_menu(self):
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        menu.setObjectName("objectSnapMenu")
        for kind in (
            "point",
            "endpoint",
            "midpoint",
            "center",
            "quadrant",
            "intersection",
            "nearest",
        ):
            action = QAction(_SNAP_KIND_LABELS[kind], menu)
            action.setCheckable(True)
            action.setChecked(kind in DEFAULT_OBJECT_SNAP_KINDS)
            action.toggled.connect(self._emit_kinds_changed)
            self._actions[kind] = action
            menu.addAction(action)
        menu.addSeparator()
        hint = QAction("垂足和切点由相关构造工具临时启用", menu)
        hint.setEnabled(False)
        menu.addAction(hint)
        return menu

    def _on_toggled(self, enabled: bool) -> None:
        self._refresh_text()
        self.enabledChanged.emit(bool(enabled))

    def _refresh_text(self) -> None:
        self.setText("对象捕捉：开" if self.isChecked() else "对象捕捉：关")

    def _emit_kinds_changed(self, _checked: bool = False) -> None:
        self.kindsChanged.emit(self.enabledKinds())

    def enabledKinds(self) -> frozenset[str]:
        return frozenset(
            kind for kind, action in self._actions.items() if action.isChecked()
        )

    def setSnapState(self, enabled: bool, kinds: Iterable[str]) -> None:
        normalized = {str(kind) for kind in kinds}
        self.blockSignals(True)
        self.setChecked(bool(enabled))
        self.blockSignals(False)
        for kind, action in self._actions.items():
            action.blockSignals(True)
            action.setChecked(kind in normalized)
            action.blockSignals(False)
        self._refresh_text()

    def showActiveKind(self, kind: str | None) -> None:
        token = str(kind or "")
        if token and token in _SNAP_KIND_LABELS and self.isChecked():
            self.setText(f"对象捕捉：{_SNAP_KIND_LABELS[token]}")
        else:
            self._refresh_text()


class ConstructionContextWidget(QWidget):
    """Responsive command context; every operation remains mouse-completable."""

    backRequested = Signal()
    finishRequested = Signal()
    cancelRequested = Signal()
    parameterChanged = Signal(str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._distance_px = 20.0
        self._pixels_per_display_unit = 1.0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.promptLabel = QLabel("辅助几何", self)
        self.promptLabel.setProperty("contextHeader", True)
        self.promptLabel.setMinimumWidth(150)
        layout.addWidget(self.promptLabel)

        self.distanceLabel = QLabel("距离", self)
        self.distanceSpin = QDoubleSpinBox(self)
        self.distanceSpin.setRange(0.0, 1_000_000_000.0)
        self.distanceSpin.setDecimals(3)
        self.distanceSpin.setValue(self._distance_px)
        self.distanceSpin.setSuffix(" px")
        self.distanceSpin.setKeyboardTracking(False)
        self.distanceSpin.valueChanged.connect(self._on_distance_changed)
        layout.addWidget(self.distanceLabel)
        layout.addWidget(self.distanceSpin)

        self.countLabel = QLabel("每侧条数", self)
        self.countSpin = QSpinBox(self)
        self.countSpin.setRange(1, 10_000)
        self.countSpin.setValue(2)
        self.countSpin.valueChanged.connect(
            lambda value: self.parameterChanged.emit("count", int(value))
        )
        layout.addWidget(self.countLabel)
        layout.addWidget(self.countSpin)

        self.bothSidesCheck = QCheckBox("双侧", self)
        self.bothSidesCheck.toggled.connect(
            lambda value: self.parameterChanged.emit("both_sides", bool(value))
        )
        layout.addWidget(self.bothSidesCheck)

        self.extendCheck = QCheckBox("按延长线求交", self)
        self.extendCheck.setChecked(False)
        self.extendCheck.toggled.connect(
            lambda value: self.parameterChanged.emit("extend", bool(value))
        )
        layout.addWidget(self.extendCheck)

        self.backButton = QToolButton(self)
        self.backButton.setProperty("contextTool", True)
        self.backButton.setText("撤销上一步")
        self.backButton.clicked.connect(self.backRequested)
        layout.addWidget(self.backButton)

        self.finishButton = QToolButton(self)
        self.finishButton.setProperty("contextTool", True)
        self.finishButton.setText("完成")
        self.finishButton.clicked.connect(self.finishRequested)
        layout.addWidget(self.finishButton)

        self.cancelButton = QToolButton(self)
        self.cancelButton.setProperty("contextTool", True)
        self.cancelButton.setText("取消")
        self.cancelButton.clicked.connect(self.cancelRequested)
        layout.addWidget(self.cancelButton)
        self.configure("")

    def setPrompt(self, text: str) -> None:
        self.promptLabel.setText(str(text or "辅助几何"))
        self.promptLabel.setToolTip(self.promptLabel.text())

    def configure(self, tool_kind: str) -> None:
        kind = str(tool_kind or "")
        distance_visible = kind in {
            "circle_center_radius",
            "circle_center_diameter",
            "parallel_offset",
            "parallel_array",
            "offset_circle",
            "concentric_circle",
            "tangent_circle_ttr",
        }
        count_visible = kind in {"parallel_array", "polar_array"}
        both_visible = kind == "parallel_array"
        extend_visible = kind in {
            "intersection",
            "tangent_circle_ttr",
            "tangent_circle_3",
        }
        self.distanceLabel.setVisible(distance_visible)
        self.distanceSpin.setVisible(distance_visible)
        self.countLabel.setVisible(count_visible)
        self.countSpin.setVisible(count_visible)
        self.bothSidesCheck.setVisible(both_visible)
        self.extendCheck.setVisible(extend_visible)

    def setDistanceUnit(self, unit: str, pixels_per_unit: float | None = None) -> None:
        normalized = str(unit or "px")
        try:
            factor = float(pixels_per_unit) if pixels_per_unit is not None else 1.0
        except (TypeError, ValueError):
            factor = 1.0
        if factor <= 0.0:
            factor = 1.0
        self._pixels_per_display_unit = factor
        self.distanceSpin.blockSignals(True)
        self.distanceSpin.setValue(self._distance_px / factor)
        self.distanceSpin.setSuffix(f" {normalized}")
        self.distanceSpin.blockSignals(False)

    def setDistanceValuePixels(self, value: float) -> None:
        try:
            normalized = max(0.0, float(value))
        except (TypeError, ValueError):
            return
        self._distance_px = normalized
        self.distanceSpin.blockSignals(True)
        self.distanceSpin.setValue(
            normalized / max(self._pixels_per_display_unit, 1e-12)
        )
        self.distanceSpin.blockSignals(False)

    def setCommandState(
        self,
        *,
        distance_px: float | None = None,
        count: int | None = None,
        both_sides: bool | None = None,
        extend: bool | None = None,
    ) -> None:
        """Apply one canvas command snapshot without feeding it back to Canvas."""

        if distance_px is not None:
            self.setDistanceValuePixels(distance_px)
        if count is not None:
            try:
                normalized_count = max(1, min(10_000, int(count)))
            except (TypeError, ValueError):
                normalized_count = None
            if normalized_count is not None:
                self.countSpin.blockSignals(True)
                self.countSpin.setValue(normalized_count)
                self.countSpin.blockSignals(False)
        if both_sides is not None:
            self.bothSidesCheck.blockSignals(True)
            self.bothSidesCheck.setChecked(bool(both_sides))
            self.bothSidesCheck.blockSignals(False)
        if extend is not None:
            self.extendCheck.blockSignals(True)
            self.extendCheck.setChecked(bool(extend))
            self.extendCheck.blockSignals(False)

    def _on_distance_changed(self, value: float) -> None:
        self._distance_px = float(value) * self._pixels_per_display_unit
        self.parameterChanged.emit("distance", self._distance_px)


class ConstructionManagerPanel(QWidget):
    """Document-local construction layer manager used beside the ROI page."""

    selectionChanged = Signal(object)
    metadataChangeRequested = Signal(str, str, object)
    batchColorChangeRequested = Signal(object, str)
    locateRequested = Signal(str)
    copyRequested = Signal(object)
    deleteRequested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._refreshing = False
        self._entity_by_id: dict[str, object] = {}
        self._item_by_id: dict[str, QTreeWidgetItem] = {}
        self._entity_order: tuple[str, ...] = ()
        self._resolution_signature_by_id: dict[str, tuple[bool, str]] = {}
        self._resolution_mapping_token: object = None
        self._content_revision: object = None
        self._has_content_snapshot = False
        self._entities_container_token: object = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.searchEdit = QLineEdit(self)
        self.searchEdit.setPlaceholderText("搜索辅助对象")
        self.searchEdit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.searchEdit)

        self.tree = QTreeWidget(self)
        self.tree.setObjectName("constructionObjectTree")
        self.tree.setHeaderLabels(["名称", "状态", "显示", "锁定", "捕捉"])
        self.tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemSelectionChanged.connect(self._emit_selection)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._locate_item)
        layout.addWidget(self.tree, 1)

        row = QHBoxLayout()
        self.locateButton = QPushButton("定位", self)
        self.locateButton.clicked.connect(self._locate_selected)
        row.addWidget(self.locateButton)
        self.copyButton = QPushButton("复制到…", self)
        self.copyButton.clicked.connect(
            lambda: self.copyRequested.emit(self.selectedIds())
        )
        row.addWidget(self.copyButton)
        self.colorButton = QPushButton("颜色…", self)
        self.colorButton.clicked.connect(self._choose_color)
        row.addWidget(self.colorButton)
        self.deleteButton = QPushButton("删除", self)
        self.deleteButton.clicked.connect(
            lambda: self.deleteRequested.emit(self.selectedIds())
        )
        row.addWidget(self.deleteButton)
        layout.addLayout(row)
        self._update_buttons()

    def setEntities(
        self,
        entities: Iterable[object],
        *,
        selected_id: str | None = None,
        resolution_by_id: Mapping[str, object] | None = None,
        content_revision: object = None,
    ) -> None:
        if (
            self._has_content_snapshot
            and entities is self._entities_container_token
            and content_revision == self._content_revision
            and resolution_by_id is self._resolution_mapping_token
        ):
            self.selectEntity(selected_id)
            return
        sequence = tuple(entities)
        entity_order = tuple(
            str(getattr(entity, "id", "")) for entity in sequence
        )
        resolution_signatures = {
            entity_id: self._resolution_signature(
                resolution_by_id.get(entity_id)
                if resolution_by_id is not None
                else None
            )
            for entity_id in entity_order
        }
        if (
            content_revision == self._content_revision
            and entity_order == self._entity_order
            and resolution_signatures == self._resolution_signature_by_id
        ):
            self.selectEntity(selected_id)
            return
        if entity_order == self._entity_order and self._item_by_id:
            self._refreshing = True
            filter_changed = False
            try:
                for entity in sequence:
                    entity_id = str(getattr(entity, "id", ""))
                    previous = self._entity_by_id.get(entity_id)
                    previous_name = str(getattr(previous, "name", "") or "")
                    next_name = str(getattr(entity, "name", "") or "")
                    if (
                        previous is not entity
                        or self._resolution_signature_by_id.get(entity_id)
                        != resolution_signatures[entity_id]
                    ):
                        item = self._item_by_id[entity_id]
                        resolved = (
                            resolution_by_id.get(entity_id)
                            if resolution_by_id is not None
                            else None
                        )
                        self._update_entity_item(item, entity, resolved)
                    filter_changed = filter_changed or previous_name != next_name
                self._entity_by_id = {
                    entity_id: entity
                    for entity_id, entity in zip(
                        entity_order,
                        sequence,
                        strict=True,
                    )
                }
                self._resolution_signature_by_id = resolution_signatures
                self._resolution_mapping_token = resolution_by_id
                self._content_revision = content_revision
                self._has_content_snapshot = True
                self._entities_container_token = entities
                if filter_changed:
                    self._apply_filter()
            finally:
                self._refreshing = False
            self.selectEntity(selected_id)
            return

        self._refreshing = True
        try:
            self.tree.clear()
            self._entity_by_id = {
                str(getattr(entity, "id", "")): entity
                for entity in sequence
            }
            self._item_by_id = {}
            for entity in sequence:
                entity_id = str(getattr(entity, "id", ""))
                resolved = (
                    resolution_by_id.get(entity_id)
                    if resolution_by_id is not None
                    else None
                )
                item = QTreeWidgetItem(["", "", "", "", ""])
                self._update_entity_item(item, entity, resolved)
                self.tree.addTopLevelItem(item)
                self._item_by_id[entity_id] = item
                if entity_id and entity_id == selected_id:
                    item.setSelected(True)
            self._entity_order = entity_order
            self._resolution_signature_by_id = resolution_signatures
            self._resolution_mapping_token = resolution_by_id
            self._content_revision = content_revision
            self._has_content_snapshot = True
            self._entities_container_token = entities
            self.tree.resizeColumnToContents(0)
            self.tree.resizeColumnToContents(1)
            self._apply_filter()
        finally:
            self._refreshing = False
        self._update_buttons()

    @staticmethod
    def _resolution_signature(resolved: object | None) -> tuple[bool, str]:
        valid = bool(getattr(resolved, "valid", True))
        error = getattr(resolved, "error", None)
        reason = str(getattr(error, "message", "") or "")
        return valid, reason

    def _update_entity_item(
        self,
        item: QTreeWidgetItem,
        entity: object,
        resolved: object | None,
    ) -> None:
        entity_id = str(getattr(entity, "id", ""))
        name = str(
            getattr(entity, "name", "") or construction_kind_label(entity)
        )
        valid, reason = self._resolution_signature(resolved)
        status = "有效" if valid else "不可解"
        item.setText(0, name)
        item.setText(1, status)
        item.setData(0, Qt.ItemDataRole.UserRole, entity_id)
        item.setToolTip(0, f"{construction_kind_label(entity)} · {entity_id}")
        item.setToolTip(1, reason if reason else status)
        item.setForeground(0, QBrush())
        item.setForeground(1, QBrush())
        if not valid:
            warning_brush = QBrush(QColor("#D84315"))
            item.setForeground(0, warning_brush)
            item.setForeground(1, warning_brush)
            if reason:
                item.setToolTip(
                    0,
                    f"{construction_kind_label(entity)} · {entity_id}\n不可解：{reason}",
                )
        for column, checked in (
            (2, bool(getattr(entity, "visible", True))),
            (3, bool(getattr(entity, "locked", False))),
            (
                4,
                bool(
                    getattr(
                        entity,
                        "snap_enabled",
                        getattr(entity, "snappable", True),
                    )
                ),
            ),
        ):
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                column,
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked,
            )

    def selectedIds(self) -> tuple[str, ...]:
        return tuple(
            str(item.data(0, Qt.ItemDataRole.UserRole) or "")
            for item in self.tree.selectedItems()
            if item.data(0, Qt.ItemDataRole.UserRole)
        )

    def selectEntity(self, entity_id: str | None) -> None:
        self._refreshing = True
        try:
            self.tree.clearSelection()
            if entity_id is None:
                return
            item = self._item_by_id.get(entity_id)
            if item is not None:
                item.setSelected(True)
                self.tree.scrollToItem(item)
        finally:
            self._refreshing = False
        self._update_buttons()

    def _apply_filter(self) -> None:
        query = self.searchEdit.text().strip().casefold()
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            haystack = f"{item.text(0)} {item.toolTip(0)}".casefold()
            item.setHidden(bool(query and query not in haystack))

    def _emit_selection(self) -> None:
        self._update_buttons()
        if not self._refreshing:
            self.selectionChanged.emit(self.selectedIds())

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._refreshing or column not in {2, 3, 4}:
            return
        entity_id = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
        field = {2: "visible", 3: "locked", 4: "snap_enabled"}[column]
        value = item.checkState(column) == Qt.CheckState.Checked
        self.metadataChangeRequested.emit(entity_id, field, value)

    def _locate_item(self, item: QTreeWidgetItem, _column: int) -> None:
        entity_id = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
        if entity_id:
            self.locateRequested.emit(entity_id)

    def _locate_selected(self) -> None:
        selected = self.selectedIds()
        if selected:
            self.locateRequested.emit(selected[0])

    def _update_buttons(self) -> None:
        selected = self.selectedIds()
        contains_locked = any(
            bool(getattr(self._entity_by_id.get(entity_id), "locked", False))
            for entity_id in selected
        )
        self.locateButton.setEnabled(len(selected) == 1)
        self.copyButton.setEnabled(bool(selected))
        self.colorButton.setEnabled(bool(selected) and not contains_locked)
        self.deleteButton.setEnabled(bool(selected) and not contains_locked)

    def _choose_color(self) -> None:
        selected = self.selectedIds()
        if not selected or any(
            bool(getattr(self._entity_by_id.get(entity_id), "locked", False))
            for entity_id in selected
        ):
            return
        entity = self._entity_by_id.get(selected[0])
        style = getattr(entity, "style", None)
        initial = str(getattr(style, "stroke_color", "#29B6C8"))
        from PySide6.QtGui import QColor

        color = QColorDialog.getColor(QColor(initial), self, "辅助对象颜色")
        if not color.isValid():
            return
        self.batchColorChangeRequested.emit(
            selected,
            color.name(QColor.NameFormat.HexRgb),
        )
