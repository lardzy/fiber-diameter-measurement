"""Reusable named-profile controls for Analyze parameter dialogs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from uuid import uuid4

from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QWidget,
)

from fdm.services.analysis_profiles import (
    AnalysisMeasurementProfile,
    AnalysisMeasurementProfileStore,
    AnalysisOutputFieldSchema,
    analysis_output_field_schema,
)
from fdm.ui.widgets import NoWheelComboBox


class AnalysisOutputFieldSelector(QGroupBox):
    """Chinese checkbox list driven by one immutable output-field schema."""

    def __init__(
        self,
        tool_id: str,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("输出字段", parent)
        self.schema: AnalysisOutputFieldSchema | None = (
            analysis_output_field_schema(tool_id)
        )
        self._checks: dict[str, QCheckBox] = {}
        layout = QGridLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(5)
        if self.schema is None:
            self.setVisible(False)
            return
        for index, field in enumerate(self.schema.fields):
            checkbox = QCheckBox(field.chinese_name, self)
            checkbox.setObjectName(f"analysisOutputField_{field.key}")
            checkbox.setChecked(True)
            if field.description:
                checkbox.setToolTip(field.description)
            self._checks[field.key] = checkbox
            layout.addWidget(checkbox, index // 2, index % 2)
        audit_note = QCheckBox("审计来源与必要上下文字段（始终保留）", self)
        audit_note.setChecked(True)
        audit_note.setEnabled(False)
        audit_note.setToolTip(
            "来源图片、像素修订、ROI/测量引用、标定签名及解释所需字段"
            "不会被输出字段选择删除。"
        )
        audit_row = (len(self.schema.fields) + 1) // 2
        layout.addWidget(audit_note, audit_row, 0, 1, 2)

    def output_fields(self) -> tuple[str, ...] | None:
        if self.schema is None:
            return None
        return tuple(
            field.key
            for field in self.schema.fields
            if self._checks[field.key].isChecked()
        )

    def set_output_fields(self, fields: Iterable[str] | None) -> None:
        if self.schema is None:
            if fields is not None and tuple(fields):
                raise ValueError("当前分析工具不支持输出字段选择")
            return
        normalized = self.schema.normalize(fields, legacy_defaults=False)
        selected = set(normalized or ())
        for field in self.schema.fields:
            self._checks[field.key].setChecked(field.key in selected)


class AnalysisProfileControls(QWidget):
    """Load/save/delete profiles without weakening schema validation."""

    def __init__(
        self,
        *,
        tool_id: str,
        tool_version: str,
        read_parameters: Callable[[], Mapping[str, object]],
        apply_parameters: Callable[[Mapping[str, object]], None],
        read_output_fields: Callable[[], Iterable[str] | None] | None = None,
        apply_output_fields: Callable[[Iterable[str] | None], None] | None = None,
        store: AnalysisMeasurementProfileStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._tool_id = str(tool_id)
        self._tool_version = str(tool_version)
        self._read_parameters = read_parameters
        self._apply_parameters = apply_parameters
        self._read_output_fields = read_output_fields
        self._apply_output_fields = apply_output_fields
        self._store = store or AnalysisMeasurementProfileStore()
        self._profiles: dict[str, AnalysisMeasurementProfile] = {}
        self._refreshing = False

        self.combo = NoWheelComboBox(self)
        self.combo.setObjectName("analysisProfileCombo")
        self.save_button = QPushButton("另存预设…", self)
        self.save_button.setObjectName("analysisProfileSaveButton")
        self.delete_button = QPushButton("删除预设", self)
        self.delete_button.setObjectName("analysisProfileDeleteButton")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.combo, 1)
        layout.addWidget(self.save_button)
        layout.addWidget(self.delete_button)

        self.combo.currentIndexChanged.connect(self._load_selected)
        self.save_button.clicked.connect(self._save)
        self.delete_button.clicked.connect(self._delete)
        self.refresh()

    def refresh(self, *, select_profile_id: str | None = None) -> None:
        try:
            loaded = self._store.load()
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "分析预设", str(exc))
            loaded = ()
        compatible = tuple(
            profile
            for profile in loaded
            if profile.tool_id == self._tool_id
            and profile.tool_version == self._tool_version
        )
        self._profiles = {profile.profile_id: profile for profile in compatible}
        self._refreshing = True
        try:
            self.combo.clear()
            self.combo.addItem("当前参数（未载入预设）", None)
            for profile in compatible:
                self.combo.addItem(profile.name, profile.profile_id)
            if select_profile_id is not None:
                index = self.combo.findData(select_profile_id)
                if index >= 0:
                    self.combo.setCurrentIndex(index)
        finally:
            self._refreshing = False
        self.delete_button.setEnabled(self.combo.currentData() is not None)

    def _load_selected(self, _index: int) -> None:
        profile_id = self.combo.currentData()
        self.delete_button.setEnabled(profile_id is not None)
        if self._refreshing or profile_id is None:
            return
        profile = self._profiles.get(str(profile_id))
        if profile is None:
            return
        try:
            self._apply_parameters(profile.parameters)
            if self._apply_output_fields is not None:
                self._apply_output_fields(profile.output_fields)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(
                self,
                "载入分析预设",
                f"预设与当前参数 schema 不兼容：\n{exc}",
            )

    def _save(self) -> None:
        try:
            parameters = dict(self._read_parameters())
            selected_fields = (
                None
                if self._read_output_fields is None
                else self._read_output_fields()
            )
            output_fields = (
                None if selected_fields is None else tuple(selected_fields)
            )
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "保存分析预设", str(exc))
            return
        current_id = self.combo.currentData()
        current = self._profiles.get(str(current_id)) if current_id else None
        name, accepted = QInputDialog.getText(
            self,
            "保存分析预设",
            "预设名称：",
            text="" if current is None else current.name,
        )
        name = str(name).strip()
        if not accepted or not name:
            return
        same_name = next(
            (
                profile
                for profile in self._profiles.values()
                if profile.name.casefold() == name.casefold()
            ),
            None,
        )
        existing = current or same_name
        try:
            profile = (
                AnalysisMeasurementProfile(
                    profile_id=f"profile_{uuid4().hex}",
                    name=name,
                    tool_id=self._tool_id,
                    tool_version=self._tool_version,
                    parameters=parameters,
                    output_fields=output_fields,
                )
                if existing is None
                else existing.with_updates(
                    name=name,
                    parameters=parameters,
                    output_fields=output_fields,
                )
            )
            self._store.upsert(profile)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "保存分析预设", str(exc))
            return
        self.refresh(select_profile_id=profile.profile_id)

    def _delete(self) -> None:
        profile_id = self.combo.currentData()
        profile = self._profiles.get(str(profile_id)) if profile_id else None
        if profile is None:
            return
        answer = QMessageBox.question(
            self,
            "删除分析预设",
            f"确定删除预设“{profile.name}”吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._store.delete(profile.profile_id)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "删除分析预设", str(exc))
            return
        self.refresh()


__all__ = ["AnalysisOutputFieldSelector", "AnalysisProfileControls"]
