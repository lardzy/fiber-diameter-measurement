"""Embeddable project-ROI manager for the measurement workspace.

The panel deliberately follows a controlled-component contract: it renders the
immutable :class:`~fdm.project_roi.ProjectRoi` values supplied by its host and
only emits immutable requests.  It never mutates ``ProjectState`` or a ROI
instance by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QColorDialog,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fdm.project_roi import ProjectRoi, ProjectRoiKind, RoiBooleanOperator
from fdm.ui.icons import themed_icon


_KIND_LABELS: dict[ProjectRoiKind, str] = {
    ProjectRoiKind.RECTANGLE: "矩形 ROI",
    ProjectRoiKind.ELLIPSE: "椭圆 ROI",
    ProjectRoiKind.POLYGON: "多边形 ROI",
    ProjectRoiKind.FREEHAND: "自由形状 ROI",
    ProjectRoiKind.COMPOSITE: "组合 ROI",
}

_BOOLEAN_LABELS: dict[RoiBooleanOperator, str] = {
    RoiBooleanOperator.UNION: "并集",
    RoiBooleanOperator.INTERSECTION: "交集",
    RoiBooleanOperator.DIFFERENCE: "差集",
    RoiBooleanOperator.XOR: "异或",
}


def _new_request_id() -> str:
    return uuid4().hex


def _required_text(
    value: object,
    *,
    field_name: str,
    maximum_length: int | None = None,
) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name}不能为空")
    if maximum_length is not None and len(text) > maximum_length:
        raise ValueError(f"{field_name}不能超过 {maximum_length} 个字符")
    return text


@dataclass(frozen=True, slots=True)
class RoiObjectRef:
    """Reference used by mutating requests to reject stale UI actions."""

    roi_id: str
    expected_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "roi_id",
            _required_text(self.roi_id, field_name="ROI ID"),
        )
        if (
            isinstance(self.expected_revision, bool)
            or not isinstance(self.expected_revision, int)
            or self.expected_revision < 0
        ):
            raise ValueError("expected_revision 必须是非负整数")


@dataclass(frozen=True, slots=True)
class RoiCreateRequest:
    request_id: str
    document_id: str
    kind: ProjectRoiKind

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _required_text(self.request_id, field_name="request_id"),
        )
        object.__setattr__(
            self,
            "document_id",
            _required_text(self.document_id, field_name="document_id"),
        )
        object.__setattr__(self, "kind", ProjectRoiKind(self.kind))
        if self.kind is ProjectRoiKind.COMPOSITE:
            raise ValueError("组合 ROI 必须通过布尔运算创建")


@dataclass(frozen=True, slots=True)
class RoiCreateFromAreaRequest:
    request_id: str
    document_id: str
    measurement_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _required_text(self.request_id, field_name="request_id"),
        )
        object.__setattr__(
            self,
            "document_id",
            _required_text(self.document_id, field_name="document_id"),
        )
        object.__setattr__(
            self,
            "measurement_id",
            _required_text(self.measurement_id, field_name="measurement_id"),
        )


@dataclass(frozen=True, slots=True)
class RoiMetadataChangeRequest:
    """Complete desired metadata for one ROI.

    Supplying the complete metadata avoids ambiguous ``None`` patch semantics
    for clearing a group while still leaving geometry entirely untouched.
    """

    request_id: str
    document_id: str
    target: RoiObjectRef
    name: str
    group: str | None
    visible: bool
    locked: bool
    color: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _required_text(self.request_id, field_name="request_id"),
        )
        object.__setattr__(
            self,
            "document_id",
            _required_text(self.document_id, field_name="document_id"),
        )
        object.__setattr__(
            self,
            "name",
            _required_text(
                self.name,
                field_name="ROI 名称",
                maximum_length=256,
            ),
        )
        if self.group is not None:
            group = str(self.group).strip()
            if len(group) > 256:
                raise ValueError("ROI 分组不能超过 256 个字符")
            object.__setattr__(self, "group", group or None)
        if not isinstance(self.visible, bool):
            raise TypeError("visible 必须是布尔值")
        if not isinstance(self.locked, bool):
            raise TypeError("locked 必须是布尔值")
        color = str(self.color or "").strip().upper()
        if len(color) != 7 or color[0] != "#":
            raise ValueError("color 必须是 #RRGGBB")
        try:
            int(color[1:], 16)
        except ValueError as error:
            raise ValueError("color 必须是 #RRGGBB") from error
        object.__setattr__(self, "color", color)


@dataclass(frozen=True, slots=True)
class RoiDeleteRequest:
    request_id: str
    document_id: str
    targets: tuple[RoiObjectRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _required_text(self.request_id, field_name="request_id"),
        )
        object.__setattr__(
            self,
            "document_id",
            _required_text(self.document_id, field_name="document_id"),
        )
        targets = tuple(self.targets)
        if not targets:
            raise ValueError("至少需要选择一个 ROI")
        if len({target.roi_id for target in targets}) != len(targets):
            raise ValueError("targets 不能包含重复 ROI")
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True, slots=True)
class RoiBooleanRequest:
    request_id: str
    document_id: str
    operator: RoiBooleanOperator
    operands: tuple[RoiObjectRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _required_text(self.request_id, field_name="request_id"),
        )
        object.__setattr__(
            self,
            "document_id",
            _required_text(self.document_id, field_name="document_id"),
        )
        object.__setattr__(self, "operator", RoiBooleanOperator(self.operator))
        operands = tuple(self.operands)
        if len(operands) < 2:
            raise ValueError("ROI 布尔运算至少需要两个对象")
        if len({operand.roi_id for operand in operands}) != len(operands):
            raise ValueError("operands 不能包含重复 ROI")
        object.__setattr__(self, "operands", operands)


@dataclass(frozen=True, slots=True)
class RoiSelectionRequest:
    document_id: str | None
    roi_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.document_id is not None:
            object.__setattr__(
                self,
                "document_id",
                _required_text(self.document_id, field_name="document_id"),
            )
        roi_ids = tuple(
            _required_text(roi_id, field_name="ROI ID")
            for roi_id in self.roi_ids
        )
        if len(set(roi_ids)) != len(roi_ids):
            raise ValueError("roi_ids 不能包含重复 ROI")
        object.__setattr__(self, "roi_ids", roi_ids)


class RoiManagerPanel(QWidget):
    """Compact ROI list intended for the project-navigation dock."""

    createRequested = Signal(object)
    createFromAreaRequested = Signal(object)
    metadataChangeRequested = Signal(object)
    deleteRequested = Signal(object)
    booleanOperationRequested = Signal(object)
    selectionChanged = Signal(object)
    locateRequested = Signal(object)

    _ROI_ID_ROLE = Qt.ItemDataRole.UserRole

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("roiManagerPanel")
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._document_id: str | None = None
        self._current_area_measurement_id: str | None = None
        self._all_rois: tuple[ProjectRoi, ...] = ()
        self._roi_by_id: dict[str, ProjectRoi] = {}
        self._updating = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("ROI", self)
        title.setObjectName("roiManagerTitle")
        self._count_label = QLabel("0 个", self)
        self._count_label.setObjectName("roiManagerCount")
        self._count_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self._count_label)
        root.addLayout(header)

        self._search_edit = QLineEdit(self)
        self._search_edit.setObjectName("roiSearchEdit")
        self._search_edit.setPlaceholderText("搜索 ROI 名称或分组")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._apply_filter)
        root.addWidget(self._search_edit)

        create_host = QWidget(self)
        create_row = QHBoxLayout(create_host)
        create_row.setContentsMargins(0, 0, 0, 0)
        create_row.setSpacing(6)
        self._create_button = QToolButton(create_host)
        self._create_button.setObjectName("roiCreateButton")
        self._create_button.setText("新建 ROI")
        self._create_button.setIcon(themed_icon("add", color="#7BD389"))
        self._create_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._create_button.setMinimumWidth(0)
        self._create_button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._create_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self._create_menu = QMenu(self._create_button)
        for kind in (
            ProjectRoiKind.RECTANGLE,
            ProjectRoiKind.ELLIPSE,
            ProjectRoiKind.POLYGON,
            ProjectRoiKind.FREEHAND,
        ):
            action = self._create_menu.addAction(_KIND_LABELS[kind])
            action.setData(kind.value)
            action.triggered.connect(
                lambda checked=False, selected_kind=kind: self.request_create(
                    selected_kind
                )
            )
        self._create_button.setMenu(self._create_menu)
        create_row.addWidget(self._create_button)

        self._from_area_button = QToolButton(create_host)
        self._from_area_button.setObjectName("roiFromAreaButton")
        self._from_area_button.setText("从面积创建")
        self._from_area_button.setIcon(
            themed_icon("polygon_area", color="#7BD389")
        )
        self._from_area_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._from_area_button.setMinimumWidth(0)
        self._from_area_button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._from_area_button.setToolTip(
            "将画布当前选中的面积测量按原始 rings 创建为 ROI"
        )
        self._from_area_button.clicked.connect(self.request_create_from_area)
        create_row.addWidget(self._from_area_button)
        root.addWidget(create_host)

        self._tree = QTreeWidget(self)
        self._tree.setObjectName("roiTree")
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(("名称 / 分组", "显示", "锁定", "颜色"))
        self._tree.setRootIsDecorated(False)
        self._tree.setItemsExpandable(False)
        self._tree.setUniformRowHeights(True)
        self._tree.setAllColumnsShowFocus(True)
        self._tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._tree.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tree.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._tree.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._tree.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        self._tree.setMinimumWidth(0)
        header_view = self._tree.header()
        header_view.setStretchLastSection(False)
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2):
            header_view.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.Fixed,
            )
            self._tree.setColumnWidth(column, 42)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._tree.setColumnWidth(3, 38)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.itemActivated.connect(self._emit_locate_request)
        root.addWidget(self._tree, 1)

        self._empty_label = QLabel("当前图片暂无 ROI", self)
        self._empty_label.setObjectName("roiEmptyLabel")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)
        root.addWidget(self._empty_label, 1)

        action_host = QWidget(self)
        action_row = QHBoxLayout(action_host)
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(6)

        self._edit_button = QToolButton(action_host)
        self._edit_button.setObjectName("roiEditButton")
        self._edit_button.setText("编辑")
        self._edit_button.setIcon(themed_icon("rename", color="#D7E3FC"))
        self._edit_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._edit_button.setMinimumWidth(0)
        self._edit_button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._edit_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )
        edit_menu = QMenu(self._edit_button)
        rename_action = edit_menu.addAction("重命名…")
        rename_action.triggered.connect(self._prompt_rename)
        group_action = edit_menu.addAction("设置分组…")
        group_action.triggered.connect(self._prompt_group)
        color_action = edit_menu.addAction("设置颜色…")
        color_action.triggered.connect(self._prompt_color)
        self._edit_button.setMenu(edit_menu)
        action_row.addWidget(self._edit_button)

        self._boolean_button = QToolButton(action_host)
        self._boolean_button.setObjectName("roiBooleanButton")
        self._boolean_button.setText("组合")
        self._boolean_button.setToolTip(
            "对选中的 ROI 执行并集、交集、差集或异或"
        )
        self._boolean_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self._boolean_button.setMinimumWidth(0)
        self._boolean_button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._boolean_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )
        boolean_menu = QMenu(self._boolean_button)
        for operator in RoiBooleanOperator:
            action = boolean_menu.addAction(_BOOLEAN_LABELS[operator])
            action.setData(operator.value)
            if operator is RoiBooleanOperator.DIFFERENCE:
                action.setToolTip("当前行减去其他选中的 ROI")
            action.triggered.connect(
                lambda checked=False, selected_operator=operator: (
                    self.request_boolean(selected_operator)
                )
            )
        self._boolean_button.setMenu(boolean_menu)
        action_row.addWidget(self._boolean_button)

        self._delete_button = QToolButton(action_host)
        self._delete_button.setObjectName("roiDeleteButton")
        self._delete_button.setText("删除")
        self._delete_button.setIcon(themed_icon("delete", color="#F28482"))
        self._delete_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._delete_button.setMinimumWidth(0)
        self._delete_button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._delete_button.clicked.connect(self._confirm_delete)
        action_row.addWidget(self._delete_button)
        root.addWidget(action_host)

        self.setStyleSheet(
            "QLabel#roiManagerTitle { font-weight: 600; }"
            "QLabel#roiManagerCount, QLabel#roiEmptyLabel {"
            " color: palette(placeholder-text);"
            "}"
            "QTreeWidget#roiTree {"
            " border: 1px solid palette(mid);"
            " border-radius: 6px;"
            " background: palette(base);"
            "}"
            "QTreeWidget#roiTree::item { min-height: 30px; }"
        )
        self._refresh_controls()

    def set_current_document(self, document_id: str | None) -> None:
        normalized = str(document_id or "").strip() or None
        if normalized == self._document_id:
            return
        self._document_id = normalized
        self._current_area_measurement_id = None
        self._rebuild()

    def current_document_id(self) -> str | None:
        return self._document_id

    def set_rois(self, rois: tuple[ProjectRoi, ...] | list[ProjectRoi]) -> None:
        normalized = tuple(rois)
        if any(not isinstance(roi, ProjectRoi) for roi in normalized):
            raise TypeError("rois 必须全部是 ProjectRoi")
        if len({roi.id for roi in normalized}) != len(normalized):
            raise ValueError("rois 不能包含重复 ID")
        self._all_rois = normalized
        self._rebuild()

    def set_current_area_measurement(self, measurement_id: str | None) -> None:
        self._current_area_measurement_id = (
            str(measurement_id or "").strip() or None
        )
        self._refresh_controls()

    def selected_roi_ids(self) -> tuple[str, ...]:
        return tuple(
            roi_id
            for item in self._selected_items_in_operation_order()
            if (roi_id := self._item_roi_id(item)) is not None
        )

    def select_rois(
        self,
        roi_ids: tuple[str, ...] | list[str],
        *,
        emit_signal: bool = False,
    ) -> None:
        wanted_order = tuple(dict.fromkeys(str(roi_id) for roi_id in roi_ids))
        wanted = set(wanted_order)
        self._tree.blockSignals(True)
        try:
            first_wanted: QTreeWidgetItem | None = None
            if wanted_order:
                first_id = wanted_order[0]
                for index in range(self._tree.topLevelItemCount()):
                    candidate = self._tree.topLevelItem(index)
                    if self._item_roi_id(candidate) == first_id:
                        first_wanted = candidate
                        break
            if first_wanted is not None:
                # Establishing the current row may clear an extended
                # selection, so do it before restoring every selected row.
                self._tree.setCurrentItem(first_wanted)
            for index in range(self._tree.topLevelItemCount()):
                item = self._tree.topLevelItem(index)
                item.setSelected(self._item_roi_id(item) in wanted)
            if first_wanted is not None:
                self._tree.scrollToItem(
                    first_wanted,
                    QAbstractItemView.ScrollHint.EnsureVisible,
                )
        finally:
            self._tree.blockSignals(False)
        self._refresh_controls()
        if emit_signal:
            self._emit_selection_request()

    def request_create(self, kind: ProjectRoiKind) -> bool:
        if self._document_id is None:
            return False
        request = RoiCreateRequest(
            request_id=_new_request_id(),
            document_id=self._document_id,
            kind=kind,
        )
        self.createRequested.emit(request)
        return True

    def request_create_from_area(self) -> bool:
        if (
            self._document_id is None
            or self._current_area_measurement_id is None
        ):
            return False
        request = RoiCreateFromAreaRequest(
            request_id=_new_request_id(),
            document_id=self._document_id,
            measurement_id=self._current_area_measurement_id,
        )
        self.createFromAreaRequested.emit(request)
        return True

    def request_rename(self, name: str) -> bool:
        roi = self._single_selected_roi()
        normalized = str(name or "").strip()
        if roi is None or not normalized or normalized == roi.name:
            return False
        self._emit_metadata(roi, name=normalized)
        return True

    def request_group(self, group: str | None) -> bool:
        roi = self._single_selected_roi()
        if roi is None:
            return False
        normalized = str(group or "").strip() or None
        if normalized == roi.group:
            return False
        self._emit_metadata(roi, group=normalized)
        return True

    def request_color(self, color: str | QColor) -> bool:
        roi = self._single_selected_roi()
        if roi is None:
            return False
        qcolor = QColor(color)
        if not qcolor.isValid():
            return False
        normalized = qcolor.name(QColor.NameFormat.HexRgb).upper()
        if normalized == roi.color:
            return False
        self._emit_metadata(roi, color=normalized)
        return True

    def request_delete(self) -> bool:
        if self._document_id is None:
            return False
        targets = self._selected_refs()
        if not targets:
            return False
        self.deleteRequested.emit(
            RoiDeleteRequest(
                request_id=_new_request_id(),
                document_id=self._document_id,
                targets=targets,
            )
        )
        return True

    def request_boolean(self, operator: RoiBooleanOperator) -> bool:
        if self._document_id is None:
            return False
        operands = self._selected_refs()
        if len(operands) < 2:
            return False
        self.booleanOperationRequested.emit(
            RoiBooleanRequest(
                request_id=_new_request_id(),
                document_id=self._document_id,
                operator=operator,
                operands=operands,
            )
        )
        return True

    def _rebuild(self) -> None:
        selected_ids = set(self.selected_roi_ids())
        current_id = self._item_roi_id(self._tree.currentItem())
        document_rois = tuple(
            roi
            for roi in self._all_rois
            if roi.document_id == self._document_id
        )
        self._roi_by_id = {roi.id: roi for roi in document_rois}
        self._updating = True
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
            current_item: QTreeWidgetItem | None = None
            selected_items: list[QTreeWidgetItem] = []
            for roi in document_rois:
                item = QTreeWidgetItem(self._tree)
                item.setData(0, self._ROI_ID_ROLE, roi.id)
                item.setText(0, self._display_name(roi))
                item.setToolTip(0, self._metadata_tooltip(roi))
                item.setTextAlignment(
                    1,
                    Qt.AlignmentFlag.AlignCenter,
                )
                item.setTextAlignment(
                    2,
                    Qt.AlignmentFlag.AlignCenter,
                )
                item.setCheckState(
                    1,
                    (
                        Qt.CheckState.Checked
                        if roi.visible
                        else Qt.CheckState.Unchecked
                    ),
                )
                item.setCheckState(
                    2,
                    (
                        Qt.CheckState.Checked
                        if roi.locked
                        else Qt.CheckState.Unchecked
                    ),
                )
                item.setToolTip(3, f"颜色：{roi.color}\n双击可修改")
                item.setIcon(3, self._color_swatch_icon(roi.color))
                flags = (
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setFlags(flags)
                if roi.id in selected_ids:
                    selected_items.append(item)
                if roi.id == current_id:
                    current_item = item
            if current_item is not None:
                self._tree.setCurrentItem(current_item)
            elif selected_items:
                self._tree.setCurrentItem(selected_items[0])
            for item in selected_items:
                item.setSelected(True)
        finally:
            self._tree.blockSignals(False)
            self._updating = False
        self._count_label.setText(f"{len(document_rois)} 个")
        self._apply_filter()
        self._refresh_controls()

    def _apply_filter(self) -> None:
        token = self._search_edit.text().strip().casefold()
        visible_count = 0
        for index in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(index)
            roi = self._roi_for_item(item)
            matches = (
                roi is not None
                and (
                    not token
                    or token in roi.name.casefold()
                    or token in str(roi.group or "").casefold()
                )
            )
            item.setHidden(not matches)
            visible_count += int(matches)
        has_document_rois = bool(self._roi_by_id)
        self._tree.setVisible(has_document_rois)
        self._empty_label.setVisible(not has_document_rois)
        if not has_document_rois:
            self._empty_label.setText(
                "请先打开图片"
                if self._document_id is None
                else "当前图片暂无 ROI"
            )
        elif visible_count == 0:
            self._tree.setVisible(False)
            self._empty_label.setVisible(True)
            self._empty_label.setText("没有匹配的 ROI")

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating or column not in (1, 2):
            return
        roi = self._roi_for_item(item)
        if roi is None:
            return
        proposed = item.checkState(column) == Qt.CheckState.Checked
        self._updating = True
        try:
            if column == 1:
                item.setCheckState(
                    column,
                    (
                        Qt.CheckState.Checked
                        if roi.visible
                        else Qt.CheckState.Unchecked
                    ),
                )
                if proposed != roi.visible:
                    self._emit_metadata(roi, visible=proposed)
            else:
                item.setCheckState(
                    column,
                    (
                        Qt.CheckState.Checked
                        if roi.locked
                        else Qt.CheckState.Unchecked
                    ),
                )
                if proposed != roi.locked:
                    self._emit_metadata(roi, locked=proposed)
        finally:
            self._updating = False

    def _on_selection_changed(self) -> None:
        self._refresh_controls()
        self._emit_selection_request()

    def _on_item_double_clicked(
        self,
        item: QTreeWidgetItem,
        column: int,
    ) -> None:
        self._tree.setCurrentItem(item)
        item.setSelected(True)
        if column == 0:
            self._prompt_rename()
        elif column == 3:
            self._prompt_color()
        else:
            self._emit_locate_request(item)

    def _emit_selection_request(self) -> None:
        self.selectionChanged.emit(
            RoiSelectionRequest(
                document_id=self._document_id,
                roi_ids=self.selected_roi_ids(),
            )
        )

    def _emit_locate_request(
        self,
        item: QTreeWidgetItem,
        _column: int = 0,
    ) -> None:
        roi_id = self._item_roi_id(item)
        if roi_id is not None:
            self.locateRequested.emit(
                RoiSelectionRequest(
                    document_id=self._document_id,
                    roi_ids=(roi_id,),
                )
            )

    def _emit_metadata(
        self,
        roi: ProjectRoi,
        *,
        name: str | None = None,
        group: str | None | object = ...,
        visible: bool | None = None,
        locked: bool | None = None,
        color: str | None = None,
    ) -> None:
        if self._document_id is None:
            return
        self.metadataChangeRequested.emit(
            RoiMetadataChangeRequest(
                request_id=_new_request_id(),
                document_id=self._document_id,
                target=RoiObjectRef(roi.id, roi.revision),
                name=roi.name if name is None else name,
                group=roi.group if group is ... else group,  # type: ignore[arg-type]
                visible=roi.visible if visible is None else visible,
                locked=roi.locked if locked is None else locked,
                color=roi.color if color is None else color,
            )
        )

    def _prompt_rename(self) -> None:
        roi = self._single_selected_roi()
        if roi is None:
            return
        name, accepted = QInputDialog.getText(
            self,
            "重命名 ROI",
            "名称：",
            text=roi.name,
        )
        if not accepted:
            return
        if not str(name).strip():
            QMessageBox.warning(self, "名称无效", "ROI 名称不能为空。")
            return
        self.request_rename(name)

    def _prompt_group(self) -> None:
        roi = self._single_selected_roi()
        if roi is None:
            return
        group, accepted = QInputDialog.getText(
            self,
            "设置 ROI 分组",
            "分组（留空表示不分组）：",
            text=roi.group or "",
        )
        if accepted:
            self.request_group(group)

    def _prompt_color(self) -> None:
        roi = self._single_selected_roi()
        if roi is None:
            return
        color = QColorDialog.getColor(
            QColor(roi.color),
            self,
            "选择 ROI 颜色",
        )
        if color.isValid():
            self.request_color(color)

    def _confirm_delete(self) -> None:
        selected_count = len(self.selected_roi_ids())
        if not selected_count:
            return
        answer = QMessageBox.question(
            self,
            "删除 ROI",
            f"确定删除选中的 {selected_count} 个 ROI 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.request_delete()

    def _selected_refs(self) -> tuple[RoiObjectRef, ...]:
        refs: list[RoiObjectRef] = []
        for item in self._selected_items_in_operation_order():
            roi = self._roi_for_item(item)
            if roi is not None:
                refs.append(RoiObjectRef(roi.id, roi.revision))
        return tuple(refs)

    def _selected_items_in_operation_order(self) -> tuple[QTreeWidgetItem, ...]:
        selected = list(self._tree.selectedItems())
        current = self._tree.currentItem()
        if current in selected:
            selected.remove(current)
            selected.insert(0, current)
        return tuple(selected)

    def _single_selected_roi(self) -> ProjectRoi | None:
        selected = self._selected_items_in_operation_order()
        if len(selected) != 1:
            return None
        return self._roi_for_item(selected[0])

    def _roi_for_item(
        self,
        item: QTreeWidgetItem | None,
    ) -> ProjectRoi | None:
        roi_id = self._item_roi_id(item)
        return None if roi_id is None else self._roi_by_id.get(roi_id)

    def _item_roi_id(self, item: QTreeWidgetItem | None) -> str | None:
        if item is None:
            return None
        value = item.data(0, self._ROI_ID_ROLE)
        return str(value) if value is not None else None

    def _refresh_controls(self) -> None:
        selected_count = len(self.selected_roi_ids())
        has_document = self._document_id is not None
        self._create_button.setEnabled(has_document)
        self._from_area_button.setEnabled(
            has_document and self._current_area_measurement_id is not None
        )
        self._edit_button.setEnabled(selected_count == 1)
        self._delete_button.setEnabled(selected_count > 0)
        self._boolean_button.setEnabled(selected_count >= 2)

    @staticmethod
    def _display_name(roi: ProjectRoi) -> str:
        if roi.group:
            return f"{roi.name}  ·  {roi.group}"
        return roi.name

    @staticmethod
    def _metadata_tooltip(roi: ProjectRoi) -> str:
        lines = [
            f"名称：{roi.name}",
            f"类型：{_KIND_LABELS[roi.kind]}",
        ]
        if roi.group:
            lines.append(f"分组：{roi.group}")
        lines.extend(
            (
                f"状态：{'显示' if roi.visible else '隐藏'}"
                f" / {'已锁定' if roi.locked else '未锁定'}",
                f"颜色：{roi.color}",
            )
        )
        return "\n".join(lines)

    @staticmethod
    def _color_swatch_icon(color: str) -> QIcon:
        pixmap = QPixmap(14, 14)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(color).darker(135), 1.0))
        painter.setBrush(QColor(color))
        painter.drawEllipse(2, 2, 10, 10)
        painter.end()
        return QIcon(pixmap)


__all__ = [
    "RoiBooleanRequest",
    "RoiCreateFromAreaRequest",
    "RoiCreateRequest",
    "RoiDeleteRequest",
    "RoiManagerPanel",
    "RoiMetadataChangeRequest",
    "RoiObjectRef",
    "RoiSelectionRequest",
]
