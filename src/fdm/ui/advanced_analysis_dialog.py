"""Chinese parameter editor for the seven advanced analysis tools."""

from __future__ import annotations

from collections.abc import Iterable
import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fdm.raster import RasterPixelType
from fdm.services.analysis_profiles import AnalysisMeasurementProfileStore
from fdm.ui.analysis_parameters_dialog import analysis_parameter_schema
from fdm.ui.analysis_profile_controls import (
    AnalysisOutputFieldSelector,
    AnalysisProfileControls,
)
from fdm.ui.image_analysis_controller import AnalysisTool
from fdm.ui.widgets import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)


SPATIAL_POINT_SCOPE_KEY = "__fdm_point_scope"
SPATIAL_STUDY_AREA_MODE_KEY = "__fdm_study_area_mode"


class AdvancedAnalysisParametersDialog(QDialog):
    """Edit one advanced analysis request without wheel-sensitive editors."""

    def __init__(
        self,
        tool: AnalysisTool,
        *,
        pixel_type: RasterPixelType,
        has_analysis_mask: bool = False,
        active_group_label: str | None = None,
        initial_parameters: dict[str, object] | None = None,
        profile_store: AnalysisMeasurementProfileStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if tool not in {
            AnalysisTool.DIRECTIONALITY,
            AnalysisTool.SKELETON,
            AnalysisTool.LOCAL_THICKNESS,
            AnalysisTool.TUBENESS,
            AnalysisTool.GLCM,
            AnalysisTool.SPATIAL_DISTRIBUTION,
            AnalysisTool.SURFACE,
        }:
            raise ValueError(f"{tool.chinese_name} 不属于高级分析参数页")
        self.tool = tool
        self.pixel_type = pixel_type
        self._initial = dict(initial_parameters or {})
        if tool is AnalysisTool.GLCM:
            value_range = self._initial.get("value_range")
            if (
                isinstance(value_range, (tuple, list))
                and len(value_range) == 2
            ):
                self._initial["use_value_range"] = True
                self._initial["value_minimum"] = value_range[0]
                self._initial["value_maximum"] = value_range[1]
        self._editors: dict[str, QWidget] = {}
        self._optional_checks: dict[str, QCheckBox] = {}

        self.setWindowTitle(f"{tool.chinese_name}参数")
        self.setObjectName("advancedAnalysisParametersDialog")
        self.setModal(True)
        self.setMinimumSize(520, 420)
        self.resize(640, 600)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)

        title = QLabel(tool.chinese_name, self)
        title_font = title.font()
        title_font.setBold(True)
        title_font.setPointSizeF(title_font.pointSizeF() + 2)
        title.setFont(title_font)
        root.addWidget(title)

        help_label = QLabel(self._tool_help(tool), self)
        help_label.setObjectName("advancedAnalysisHelp")
        help_label.setWordWrap(True)
        help_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(help_label)

        scroll = QScrollArea(self)
        scroll.setObjectName("advancedAnalysisParameterScroll")
        scroll.setWidgetResizable(True)
        scroll.setProperty("redirectEditorWheel", True)
        content = QWidget(scroll)
        form_group = QGroupBox("分析参数", content)
        form = QFormLayout(form_group)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(9)
        self._build_form(
            form,
            has_analysis_mask=has_analysis_mask,
            active_group_label=active_group_label,
        )
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(6, 6, 6, 6)
        content_layout.addWidget(form_group)
        self.output_field_selector = AnalysisOutputFieldSelector(
            f"fdm.{tool.value}",
            parent=content,
        )
        content_layout.addWidget(self.output_field_selector)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        schema = analysis_parameter_schema(tool)
        self.profile_controls = AnalysisProfileControls(
            tool_id=f"fdm.{tool.value}",
            tool_version=schema.version,
            read_parameters=self.parameters,
            apply_parameters=self.set_parameters,
            read_output_fields=self.output_fields,
            apply_output_fields=self.set_output_fields,
            store=profile_store,
            parent=self,
        )
        root.addWidget(self.profile_controls)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("开始分析")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def output_fields(self) -> tuple[str, ...] | None:
        return self.output_field_selector.output_fields()

    def set_output_fields(self, fields: Iterable[str] | None) -> None:
        self.output_field_selector.set_output_fields(fields)

    def parameters(self) -> dict[str, object]:
        """Return validated kernel parameters plus private spatial UI tokens."""

        tool = self.tool
        if tool is AnalysisTool.DIRECTIONALITY:
            return {
                "channel": self._combo_value("channel"),
                "algorithm_version": self._int_combo_value(
                    "algorithm_version"
                ),
                "bins": self._int_value("bins"),
                "gradient_sigma": self._float_value("gradient_sigma"),
                "minimum_gradient": self._float_value("minimum_gradient"),
                "histogram_smoothing_bins": self._float_value(
                    "histogram_smoothing_bins"
                ),
                "peak_min_fraction": self._float_value("peak_min_fraction"),
                "max_peaks": self._int_value("max_peaks"),
            }
        if tool is AnalysisTool.SKELETON:
            result = {
                "channel": self._combo_value("channel"),
                "foreground": self._combo_value("foreground"),
                "already_skeletonized": self._checked("already_skeletonized"),
            }
            if tool is AnalysisTool.SKELETON:
                result["algorithm_version"] = self._int_combo_value(
                    "algorithm_version"
                )
                result["prune_terminal_branches_below"] = self._float_value(
                    "prune_terminal_branches_below"
                )
            threshold = self._optional_float("threshold")
            if threshold is not None:
                result["threshold"] = threshold
            return result
        if tool is AnalysisTool.LOCAL_THICKNESS:
            result = {
                "channel": self._combo_value("channel"),
                "foreground": self._combo_value("foreground"),
            }
            threshold = self._optional_float("threshold")
            if threshold is not None:
                result["threshold"] = threshold
            return result
        if tool is AnalysisTool.TUBENESS:
            result = {
                "channel": self._combo_value("channel"),
                "scales": self._float_sequence("scales", positive=True),
                "beta": self._float_value("beta"),
                "bright_ridges": self._checked("bright_ridges"),
            }
            structure_scale = self._optional_float("structure_scale")
            if structure_scale is not None:
                result["structure_scale"] = structure_scale
            return result
        if tool is AnalysisTool.GLCM:
            result = {
                "channel": self._combo_value("channel"),
                "levels": self._int_value("levels"),
                "distances": self._int_sequence("distances", positive=True),
                "directions_degrees": self._float_sequence(
                    "directions_degrees"
                ),
                "symmetric": self._checked("symmetric"),
            }
            if self._checked("use_value_range"):
                low = self._float_value("value_minimum")
                high = self._float_value("value_maximum")
                if high <= low:
                    raise ValueError("GLCM 数值上限必须大于下限。")
                result["value_range"] = (low, high)
            return result
        if tool is AnalysisTool.SPATIAL_DISTRIBUTION:
            result = {
                "algorithm_version": self._int_combo_value(
                    "algorithm_version"
                ),
                SPATIAL_POINT_SCOPE_KEY: self._combo_value("point_scope"),
                SPATIAL_STUDY_AREA_MODE_KEY: self._combo_value(
                    "study_area_mode"
                ),
            }
            ripley_radii = self._optional_float_sequence(
                "ripley_radii",
                positive=True,
            )
            if ripley_radii:
                if tuple(sorted(ripley_radii)) != ripley_radii:
                    raise ValueError("Ripley 半径必须严格递增。")
                result["ripley_radii"] = ripley_radii
            if result[SPATIAL_STUDY_AREA_MODE_KEY] == "custom":
                if result["algorithm_version"] == 2:
                    raise ValueError(
                        "Ripley K/L v2 需要完整矩形研究区域，"
                        "不能只提供面积；请选择当前矩形范围、点集包围框，"
                        "或切换到 v1。"
                    )
                result["study_area"] = self._float_value("study_area")
            return result
        if tool is AnalysisTool.SURFACE:
            return {
                "channel": self._combo_value("channel"),
                "sample_step_x": self._int_value("sample_step_x"),
                "sample_step_y": self._int_value("sample_step_y"),
            }
        raise ValueError(f"不支持的高级分析工具：{tool.value}")

    def set_parameters(self, values: dict[str, object] | object) -> None:
        if not isinstance(values, dict):
            try:
                values = dict(values)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise TypeError("分析预设参数必须是对象") from exc
        normalized = dict(values)
        if self.tool is AnalysisTool.SPATIAL_DISTRIBUTION:
            if "point_scope" in normalized:
                normalized[SPATIAL_POINT_SCOPE_KEY] = normalized.pop(
                    "point_scope"
                )
            if "study_area_mode" in normalized:
                normalized[SPATIAL_STUDY_AREA_MODE_KEY] = normalized.pop(
                    "study_area_mode"
                )
        key_aliases = {
            SPATIAL_POINT_SCOPE_KEY: "point_scope",
            SPATIAL_STUDY_AREA_MODE_KEY: "study_area_mode",
        }
        if self.tool is AnalysisTool.GLCM:
            value_range = normalized.pop("value_range", None)
            check = self._editors.get("use_value_range")
            if isinstance(check, QCheckBox):
                check.setChecked(value_range is not None)
            if value_range is not None:
                if (
                    not isinstance(value_range, (tuple, list))
                    or len(value_range) != 2
                ):
                    raise ValueError("GLCM 数值范围必须包含下限和上限")
                normalized["value_minimum"] = value_range[0]
                normalized["value_maximum"] = value_range[1]
        for raw_key, value in normalized.items():
            key = key_aliases.get(raw_key, raw_key)
            editor = self._editors.get(key)
            if editor is None:
                raise ValueError(f"预设包含当前工具不支持的参数：{raw_key}")
            if isinstance(editor, NoWheelComboBox):
                index = editor.findData(value)
                if index < 0:
                    raise ValueError(f"参数 {raw_key} 的预设选项不受支持")
                editor.setCurrentIndex(index)
            elif isinstance(editor, NoWheelSpinBox):
                editor.setValue(int(value))
            elif isinstance(editor, NoWheelDoubleSpinBox):
                editor.setValue(float(value))
            elif isinstance(editor, QCheckBox):
                editor.setChecked(bool(value))
            elif isinstance(editor, QLineEdit):
                if isinstance(value, (tuple, list)):
                    editor.setText(", ".join(str(item) for item in value))
                else:
                    editor.setText(str(value))
            else:
                raise TypeError(f"参数 {raw_key} 的编辑器类型不受支持")
            optional = self._optional_checks.get(key)
            if optional is not None:
                optional.setChecked(value is not None)

    def _build_form(
        self,
        form: QFormLayout,
        *,
        has_analysis_mask: bool,
        active_group_label: str | None,
    ) -> None:
        tool = self.tool
        if tool in {
            AnalysisTool.DIRECTIONALITY,
            AnalysisTool.SKELETON,
            AnalysisTool.LOCAL_THICKNESS,
            AnalysisTool.TUBENESS,
            AnalysisTool.GLCM,
            AnalysisTool.SURFACE,
        }:
            self._add_channel(form)
        if tool is AnalysisTool.DIRECTIONALITY:
            self._add_combo(
                form,
                "algorithm_version",
                "算法版本",
                (
                    ("v2：5×5 梯度、Fourier 融合与峰宽", 2),
                    ("v1：历史 3×3 梯度结果", 1),
                ),
                2,
            )
            self._add_int(form, "bins", "方向区间数", 180, 4, 4096)
            self._add_float(
                form, "gradient_sigma", "梯度平滑 σ（px）", 1.0, 0.0, 1000.0
            )
            self._add_float(
                form, "minimum_gradient", "最小梯度", 0.0, 0.0, 1.0e12
            )
            self._add_float(
                form,
                "histogram_smoothing_bins",
                "直方图平滑宽度（bin）",
                1.0,
                0.0,
                1000.0,
            )
            self._add_float(
                form,
                "peak_min_fraction",
                "方向峰最小相对权重",
                0.1,
                0.0,
                1.0,
                decimals=4,
            )
            self._add_int(form, "max_peaks", "最多方向峰", 8, 1, 128)
            return
        if tool in {AnalysisTool.SKELETON, AnalysisTool.LOCAL_THICKNESS}:
            self._add_combo(
                form,
                "foreground",
                "前景方向",
                (("亮于或等于阈值", "above"), ("暗于或等于阈值", "below")),
                "above",
            )
            self._add_optional_float(
                form,
                "threshold",
                "显式二值阈值",
                self._default_threshold(),
                -1.0e12,
                1.0e12,
                checked=not has_analysis_mask,
                hint=(
                    "未启用时使用当前 ROI / 面积对象作为二值掩膜。"
                    if has_analysis_mask
                    else "当前没有 ROI / 面积掩膜，必须启用显式阈值。"
                ),
            )
            if tool is AnalysisTool.SKELETON:
                self._add_combo(
                    form,
                    "algorithm_version",
                    "算法版本",
                    (
                        ("v2：节点分类、分支统计与可审计剪枝", 2),
                        ("v1：历史骨架网络结果", 1),
                    ),
                    2,
                )
                self._add_check(
                    form,
                    "already_skeletonized",
                    "输入已经是单像素骨架",
                    False,
                )
                self._add_float(
                    form,
                    "prune_terminal_branches_below",
                    "末端分支剪枝阈值（物理单位）",
                    0.0,
                    0.0,
                    1.0e12,
                )
            return
        if tool is AnalysisTool.TUBENESS:
            self._add_line(
                form,
                "scales",
                "尺度 σ（px，逗号分隔）",
                "1, 2, 4",
            )
            self._add_float(form, "beta", "线状结构 beta", 0.5, 0.000001, 1.0e6)
            self._add_optional_float(
                form,
                "structure_scale",
                "结构响应尺度",
                1.0,
                0.000001,
                1.0e6,
                checked=False,
                hint="未启用时由算法根据图像响应自动确定。",
            )
            self._add_check(form, "bright_ridges", "检测亮色纤维脊线", True)
            return
        if tool is AnalysisTool.GLCM:
            self._add_int(form, "levels", "量化级数", 32, 2, 256)
            self._add_line(form, "distances", "距离（px，逗号分隔）", "1")
            self._add_line(
                form,
                "directions_degrees",
                "方向（°，逗号分隔）",
                "0, 45, 90, 135",
            )
            self._add_check(form, "symmetric", "使用对称 GLCM", True)
            self._add_check(form, "use_value_range", "指定量化数值范围", False)
            self._add_float(
                form, "value_minimum", "量化下限", 0.0, -1.0e12, 1.0e12
            )
            self._add_float(
                form,
                "value_maximum",
                "量化上限",
                self._default_maximum(),
                -1.0e12,
                1.0e12,
            )
            use_range = self._editors["use_value_range"]
            assert isinstance(use_range, QCheckBox)
            use_range.toggled.connect(
                lambda checked: self._set_enabled(
                    ("value_minimum", "value_maximum"), checked
                )
            )
            self._set_enabled(("value_minimum", "value_maximum"), False)
            return
        if tool is AnalysisTool.SPATIAL_DISTRIBUTION:
            self._add_combo(
                form,
                "algorithm_version",
                "算法版本",
                (
                    ("v2：Ripley K/L（矩形窗口平移边界校正）", 2),
                    ("v1：历史最近邻与空间密度结果", 1),
                ),
                2,
            )
            active_label = (
                f"当前类别：{active_group_label}"
                if active_group_label
                else "当前类别（当前图片没有活动类别）"
            )
            self._add_combo(
                form,
                "point_scope",
                "计数点范围",
                ((active_label, "active_group"), ("当前图片全部计数点", "all")),
                "active_group" if active_group_label else "all",
            )
            point_scope = self._editors["point_scope"]
            assert isinstance(point_scope, NoWheelComboBox)
            if not active_group_label:
                point_scope.model().item(0).setEnabled(False)
            self._add_combo(
                form,
                "study_area_mode",
                "研究区域面积",
                (
                    ("当前 ROI / 当前视窗的有效面积", "scope"),
                    ("点集轴对齐包围框", "point_bounds"),
                    ("手工指定面积", "custom"),
                ),
                "scope",
            )
            self._add_line(
                form,
                "ripley_radii",
                "Ripley 半径（当前长度单位）",
                "",
            )
            self._editors["ripley_radii"].setToolTip(
                "使用逗号分隔的严格递增正数；留空时按研究区域尺寸自动选择。"
            )
            self._add_float(
                form, "study_area", "手工研究区域面积", 1.0, 1.0e-12, 1.0e18
            )
            mode = self._editors["study_area_mode"]
            assert isinstance(mode, NoWheelComboBox)
            mode.currentIndexChanged.connect(
                lambda _index: self._set_enabled(
                    ("study_area",), mode.currentData() == "custom"
                )
            )
            self._set_enabled(("study_area",), False)
            return
        if tool is AnalysisTool.SURFACE:
            self._add_int(form, "sample_step_x", "横向采样步长（px）", 1, 1, 10000)
            self._add_int(form, "sample_step_y", "纵向采样步长（px）", 1, 1, 10000)

    def _add_channel(self, form: QFormLayout) -> None:
        items = [("加权亮度", "luminance")]
        if self.pixel_type in {RasterPixelType.RGB8, RasterPixelType.RGBA8}:
            items.extend(
                (("红色通道", "red"), ("绿色通道", "green"), ("蓝色通道", "blue"))
            )
        self._add_combo(form, "channel", "分析通道", tuple(items), "luminance")

    def _add_combo(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        items: Iterable[tuple[str, object]],
        default: object,
    ) -> None:
        combo = NoWheelComboBox(self)
        for text, value in items:
            combo.addItem(text, value)
        selected = self._initial.get(key, default)
        index = combo.findData(selected)
        combo.setCurrentIndex(max(0, index))
        self._editors[key] = combo
        form.addRow(label, combo)

    def _add_int(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        default: int,
        minimum: int,
        maximum: int,
    ) -> None:
        editor = NoWheelSpinBox(self)
        editor.setRange(minimum, maximum)
        editor.setValue(int(self._initial.get(key, default)))
        self._editors[key] = editor
        form.addRow(label, editor)

    def _add_float(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        default: float,
        minimum: float,
        maximum: float,
        *,
        decimals: int = 6,
    ) -> None:
        editor = NoWheelDoubleSpinBox(self)
        editor.setDecimals(decimals)
        editor.setRange(minimum, maximum)
        editor.setValue(float(self._initial.get(key, default)))
        self._editors[key] = editor
        form.addRow(label, editor)

    def _add_optional_float(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        default: float,
        minimum: float,
        maximum: float,
        *,
        checked: bool,
        hint: str,
    ) -> None:
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        enabled = QCheckBox("启用", row)
        editor = NoWheelDoubleSpinBox(row)
        editor.setDecimals(6)
        editor.setRange(minimum, maximum)
        editor.setValue(float(self._initial.get(key, default)))
        initial_checked = key in self._initial or checked
        enabled.setChecked(initial_checked)
        editor.setEnabled(initial_checked)
        enabled.toggled.connect(editor.setEnabled)
        layout.addWidget(enabled)
        layout.addWidget(editor, 1)
        row.setToolTip(hint)
        self._optional_checks[key] = enabled
        self._editors[key] = editor
        form.addRow(label, row)

    def _add_check(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        default: bool,
    ) -> None:
        editor = QCheckBox(label, self)
        editor.setChecked(bool(self._initial.get(key, default)))
        self._editors[key] = editor
        form.addRow("", editor)

    def _add_line(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        default: str,
    ) -> None:
        initial = self._initial.get(key, default)
        if isinstance(initial, (tuple, list)):
            text = ", ".join(str(value) for value in initial)
        else:
            text = str(initial)
        editor = QLineEdit(text, self)
        editor.setClearButtonEnabled(True)
        self._editors[key] = editor
        form.addRow(label, editor)

    def _validate_and_accept(self) -> None:
        try:
            self.parameters()
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, self.tool.chinese_name, str(exc))
            return
        self.accept()

    def _combo_value(self, key: str) -> str:
        editor = self._editors[key]
        assert isinstance(editor, NoWheelComboBox)
        return str(editor.currentData())

    def _int_combo_value(self, key: str) -> int:
        editor = self._editors[key]
        if not isinstance(editor, NoWheelComboBox):
            raise TypeError(f"{key} 不是选择控件")
        return int(editor.currentData())

    def _int_value(self, key: str) -> int:
        editor = self._editors[key]
        assert isinstance(editor, NoWheelSpinBox)
        return int(editor.value())

    def _float_value(self, key: str) -> float:
        editor = self._editors[key]
        assert isinstance(editor, NoWheelDoubleSpinBox)
        value = float(editor.value())
        if not math.isfinite(value):
            raise ValueError(f"{key} 必须是有限数。")
        return value

    def _optional_float(self, key: str) -> float | None:
        enabled = self._optional_checks[key]
        return self._float_value(key) if enabled.isChecked() else None

    def _checked(self, key: str) -> bool:
        editor = self._editors[key]
        assert isinstance(editor, QCheckBox)
        return editor.isChecked()

    def _float_sequence(
        self,
        key: str,
        *,
        positive: bool = False,
    ) -> tuple[float, ...]:
        values = self._parse_sequence(key, float)
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"{key} 只能包含有限数。")
        if positive and any(value <= 0 for value in values):
            raise ValueError(f"{key} 只能包含正数。")
        return tuple(values)

    def _optional_float_sequence(
        self,
        key: str,
        *,
        positive: bool = False,
    ) -> tuple[float, ...]:
        editor = self._editors[key]
        assert isinstance(editor, QLineEdit)
        if not editor.text().strip():
            return ()
        return self._float_sequence(key, positive=positive)

    def _int_sequence(
        self,
        key: str,
        *,
        positive: bool = False,
    ) -> tuple[int, ...]:
        values = self._parse_sequence(key, int)
        if positive and any(value <= 0 for value in values):
            raise ValueError(f"{key} 只能包含正整数。")
        return tuple(values)

    def _parse_sequence(self, key: str, converter):
        editor = self._editors[key]
        assert isinstance(editor, QLineEdit)
        tokens = [
            token.strip()
            for token in editor.text().replace("，", ",").split(",")
            if token.strip()
        ]
        if not tokens:
            raise ValueError(f"{key} 不能为空。")
        try:
            values = [converter(token) for token in tokens]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 必须使用逗号分隔的有效数字。") from exc
        if len(set(values)) != len(values):
            raise ValueError(f"{key} 不能包含重复值。")
        return values

    def _set_enabled(self, keys: Iterable[str], enabled: bool) -> None:
        for key in keys:
            self._editors[key].setEnabled(enabled)

    def _default_threshold(self) -> float:
        return {
            RasterPixelType.GRAY16: 32767.0,
            RasterPixelType.GRAY32_FLOAT: 0.5,
        }.get(self.pixel_type, 127.0)

    def _default_maximum(self) -> float:
        return {
            RasterPixelType.GRAY16: 65535.0,
            RasterPixelType.GRAY32_FLOAT: 1.0,
        }.get(self.pixel_type, 255.0)

    @staticmethod
    def _tool_help(tool: AnalysisTool) -> str:
        return {
            AnalysisTool.DIRECTIONALITY: (
                "使用 Sobel 梯度统计纤维轴向方向。0° 指向右侧，"
                "逆时针为正，角度范围为 0°–180°。"
            ),
            AnalysisTool.SKELETON: (
                "从二值前景提取单像素骨架，统计端点、分支点、环路和分支长度。"
            ),
            AnalysisTool.LOCAL_THICKNESS: (
                "按最大内切圆定义计算前景各像素的局部厚度，不使用 2×EDT 近似。"
            ),
            AnalysisTool.TUBENESS: (
                "使用多尺度 Hessian 响应增强线状结构；尺度单位为原始像素。"
            ),
            AnalysisTool.GLCM: (
                "计算指定量化级数、距离和方向的 Haralick GLCM 纹理特征。"
            ),
            AnalysisTool.SPATIAL_DISTRIBUTION: (
                "从当前图片的 RAW 计数点计算最近邻距离、空间密度及 "
                "Ripley K/L。v2 使用矩形窗口平移边界校正；"
                "非矩形 ROI 需改用点集包围框或历史 v1。"
            ),
            AnalysisTool.SURFACE: (
                "对二维强度进行规则采样并生成表面数据；仅用于强度可视化，"
                "不生成伪三维几何测量。"
            ),
        }[tool]
