"""Read-only audit of image-processing parameter contracts and UI metadata.

This developer command deliberately reads both sources of truth involved in
the image-processing workbench:

* the service descriptor registry owns executable parameter contracts;
* the workbench catalogue owns Chinese labels, help text and units.

It never creates a ``QApplication``, executes an operation, or mutates either
registry.  The JSON shape and ordering are stable so audit results can be
reviewed or compared in CI.

Examples::

    python -m fdm.image_parameter_audit
    python -m fdm.image_parameter_audit --json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from typing import Iterable, Mapping, Sequence

from fdm.services.image_processing import image_operation_registry
from fdm.ui.image_processing_workbench import (
    ImageProcessingWorkbench,
    _CONDITIONAL_PARAMETER_FIELDS,
    _CROP_BOUNDS_PARAMETERS,
    _FREQUENCY_RESPONSE_PARAMETERS,
    _HISTOGRAM_EDITOR_PARAMETERS,
    _LINKED_DIMENSION_OPERATIONS,
    _OPERATION_CATALOG,
    _PERCENTILE_RANGE_PARAMETERS,
    _STRIPE_FREQUENCY_PARAMETERS,
    _STRUCTURING_ELEMENT_OPERATIONS,
    _parameter_help_text,
)


AUDIT_SCHEMA_VERSION = 1

_REGISTRY_ONLY_FIELD_NOTES: Mapping[str, tuple[str, str]] = {
    "adjust_levels.output_min": (
        "dynamic_default",
        "缺省时按当前步骤输入类型的工作范围解析，不能预填固定 0–255。",
    ),
    "adjust_levels.output_max": (
        "dynamic_default",
        "缺省时按当前步骤输入类型的工作范围解析，不能预填固定 0–255。",
    ),
    "copy.fill_value": (
        "compatibility_alias",
        "服务层保留的 ROI 填充值兼容别名，工作台使用 outside_value。",
    ),
    "crop.outside_value": (
        "compatibility_alias",
        "服务层保留的 ROI 填充值兼容别名，工作台使用 fill_value。",
    ),
    "flat_field_correction.reference_levels": (
        "internal_provenance",
        "最终任务从冻结参考图计算并写入配方，不允许用户手填。",
    ),
    "flat_field_correction.secondary_sha256": (
        "internal_provenance",
        "最终任务从冻结参考图生成来源摘要，不允许用户手填。",
    ),
    "gamma.minimum": (
        "dynamic_default",
        "省略时使用当前输入工作范围；固定显示会改变旧配方语义。",
    ),
    "gamma.maximum": (
        "dynamic_default",
        "省略时使用当前输入工作范围；固定显示会改变旧配方语义。",
    ),
    "gaussian_blur.sigma": (
        "compatibility_alias",
        "服务层保留的等向 Sigma 简写；工作台显式提供 sigma_x/sigma_y。",
    ),
    "invert.minimum": (
        "dynamic_default",
        "省略时使用当前输入工作范围；固定显示会改变旧配方语义。",
    ),
    "invert.maximum": (
        "dynamic_default",
        "省略时使用当前输入工作范围；固定显示会改变旧配方语义。",
    ),
    "threshold.background_value": (
        "dynamic_default",
        "省略时使用当前像素类型低端值，避免 16 位/float 被固定 8 位值污染。",
    ),
    "threshold.foreground_value": (
        "dynamic_default",
        "省略时使用当前像素类型高端值，避免 16 位/float 被固定 8 位值污染。",
    ),
}


# These rules identify places where a generic checkbox/spin-box/combo-box is a
# valid low-level editor but a poor professional workflow.  They are UI audit
# rules only: they do not add parameters or change numerical semantics.
_PROFESSIONAL_EDITOR_RULES: tuple[dict[str, object], ...] = (
    {
        "editor": "histogram_range",
        "priority": "P0",
        "operations": {
            "adjust_levels": ("black_point", "white_point"),
            "threshold": ("lower", "upper"),
            "binarize": ("threshold",),
            "canny_edges": ("threshold_low", "threshold_high"),
            "percentile_saturation": (
                "lower_percentile",
                "upper_percentile",
            ),
        },
        "reason": (
            "阈值范围应结合实际参数域直方图、双端控制柄和精确数值框；"
            "Canny 必须使用梯度幅值而非原始强度。"
        ),
    },
    {
        "editor": "slider_number",
        "priority": "P1",
        "operations": {
            "brightness_contrast": ("brightness", "contrast", "gamma"),
            "adjust_levels": ("gamma",),
            "rotate": ("angle_degrees",),
            "unsharp_mask": ("amount", "threshold"),
            "clahe": ("clip_limit",),
            "watershed": ("seed_threshold",),
            "watershed_v2": ("seed_threshold",),
            "stripe_suppression": (
                "notch_width",
                "protect_radius",
                "strength",
            ),
        },
        "reason": "高频连续参数应支持拖动预览，同时保留精确键盘输入。",
    },
    {
        "editor": "kernel_matrix",
        "priority": "P0",
        "operations": {
            "custom_convolution": (
                "kernel_width",
                "kernel_height",
                "kernel",
            ),
        },
        "reason": "卷积核需要二维矩阵、尺寸联动、预设和元素级校验。",
    },
    {
        "editor": "anchor_grid",
        "priority": "P1",
        "operations": {
            "resize_canvas": ("anchor",),
        },
        "reason": "九宫格锚点比文本下拉框更能表达画布扩展方向。",
    },
    {
        "editor": "linked_dimensions",
        "priority": "P1",
        "operations": {
            "crop": ("x", "y", "width", "height"),
            "resize": ("width", "height"),
            "resize_canvas": ("width", "height"),
        },
        "reason": "尺寸参数需要原始尺寸、宽高联动、比例锁定和边界反馈。",
    },
    {
        "editor": "structuring_element_preview",
        "priority": "P1",
        "operations": {
            "erode": ("radius", "iterations", "kernel"),
            "dilate": ("radius", "iterations", "kernel"),
            "morphology_open": ("radius", "iterations", "kernel"),
            "morphology_close": ("radius", "iterations", "kernel"),
            "top_hat": ("radius", "iterations", "kernel"),
            "black_hat": ("radius", "iterations", "kernel"),
        },
        "reason": "形态学参数需要结构元素预览和实际像素尺寸说明。",
    },
    {
        "editor": "frequency_response",
        "priority": "P1",
        "operations": {
            "fft_filter": (
                "mode",
                "low_cutoff",
                "high_cutoff",
                "order",
                "boundary",
            ),
            "fft_power_spectrum": ("window", "tukey_alpha"),
            "stripe_suppression": (
                "direction",
                "notch_width",
                "protect_radius",
            ),
        },
        "reason": "频域参数需要频谱/响应曲线与单位、边界策略的可视说明。",
    },
    {
        "editor": "compatible_image_picker",
        "priority": "P0",
        "operations": {
            "image_calculator": ("secondary_document_id",),
            "flat_field_correction": ("secondary_document_id",),
        },
        "reason": "第二图像选择器应显示尺寸、类型、通道、标定和不兼容原因。",
    },
    {
        "editor": "conditional_group",
        "priority": "P0",
        "operations": {
            "convert_type": (
                "target_type",
                "scale_mode",
                "nonfinite_policy",
            ),
            "convert_color": (
                "target_model",
                "grayscale_method",
                "drop_alpha",
            ),
            "rotate": ("border_mode", "border_value"),
            "translate": ("border_mode", "border_value"),
            "adaptive_threshold": ("method", "k", "r", "p", "q"),
            "log_v2": ("result_mode", "output_min", "output_max"),
            "exp_v2": ("result_mode", "output_min", "output_max"),
            "sqrt_v2": ("result_mode", "output_min", "output_max"),
            "fft_filter": (
                "mode",
                "low_cutoff",
                "high_cutoff",
                "boundary",
                "tukey_alpha",
                "frequency_unit",
                "pixel_size",
            ),
            "flat_field_correction": (
                "flat_field_source",
                "secondary_document_id",
                "radius",
                "method",
            ),
        },
        "reason": "从属参数必须按当前选择显隐，并在提交前做跨字段校验。",
    },
)

# A marker is deliberately a workbench method rather than merely an imported
# widget class: importing a component without routing an operation through it
# does not close the usability gap.  New integrations can add their marker
# here without changing the executable service registry.
_PROFESSIONAL_EDITOR_MARKERS: Mapping[str, tuple[str, ...]] = {
    "histogram_range": ("_add_histogram_range_editor",),
    "slider_number": ("_add_slider_number_editor",),
    "kernel_matrix": ("_add_kernel_matrix_editor",),
    "anchor_grid": ("_add_anchor_grid_editor",),
    "linked_dimensions": ("_add_linked_dimensions_editor",),
    "structuring_element_preview": (
        "_add_structuring_element_editor",
    ),
    "frequency_response": ("_add_frequency_response_editor",),
    "compatible_image_picker": (
        "_add_compatible_image_picker",
    ),
    # Presence of generic conditional-row support is useful evidence but is
    # not by itself proof that every operation-specific dependency is covered.
    # Keep this as an unresolved audit candidate until a declarative binding
    # method is introduced.
    "conditional_group": (
        "_add_declarative_conditional_parameter_group",
    ),
}


def _percentage(part: int, total: int) -> float:
    if total <= 0:
        return 100.0
    return round(part * 100.0 / total, 1)


def _sorted_counts(values: Iterable[str]) -> list[dict[str, object]]:
    counts = Counter(values)
    return [
        {"name": name, "count": counts[name]}
        for name in sorted(counts)
    ]


def _integrated_requirement_fields(
    editor: str,
    operation_id: str,
    fields: tuple[str, ...],
) -> tuple[str, ...]:
    """Return field-level evidence instead of trusting a class-wide marker."""

    definitions = {
        definition.operation.value: definition
        for definition in _OPERATION_CATALOG
    }
    definition = definitions.get(operation_id)
    if definition is None:
        return ()
    by_key = {
        parameter.key: parameter
        for parameter in definition.parameters
    }
    if editor == "histogram_range":
        configured = set(
            _HISTOGRAM_EDITOR_PARAMETERS.get(operation_id, ())
        )
        configured.update(
            _PERCENTILE_RANGE_PARAMETERS.get(operation_id, ())
        )
        return tuple(field for field in fields if field in configured)
    if editor == "slider_number":
        if not hasattr(
            ImageProcessingWorkbench,
            "_add_slider_number_editor",
        ):
            return ()
        return tuple(
            field
            for field in fields
            if field in by_key
            and ImageProcessingWorkbench._parameter_prefers_slider(  # noqa: SLF001
                by_key[field],
                operation_id,
            )
        )
    if editor == "conditional_group":
        configured = set(
            _CONDITIONAL_PARAMETER_FIELDS.get(operation_id, ())
        )
        return tuple(field for field in fields if field in configured)
    if editor == "linked_dimensions":
        operation = (
            definition.operation
            if definition is not None
            else None
        )
        configured = set(
            _CROP_BOUNDS_PARAMETERS.get(operation_id, ())
        )
        if operation in _LINKED_DIMENSION_OPERATIONS:
            configured.update({"width", "height"})
        return tuple(field for field in fields if field in configured)
    if editor == "structuring_element_preview":
        operation = (
            definition.operation
            if definition is not None
            else None
        )
        configured = (
            {"radius", "iterations", "kernel"}
            if operation in _STRUCTURING_ELEMENT_OPERATIONS
            else set()
        )
        return tuple(field for field in fields if field in configured)
    if editor == "frequency_response":
        configured = set(
            _FREQUENCY_RESPONSE_PARAMETERS.get(operation_id, ())
        )
        configured.update(
            _STRIPE_FREQUENCY_PARAMETERS.get(operation_id, ())
        )
        if operation_id == "fft_filter":
            # The boundary strategy remains the immediately adjacent,
            # conditionally rendered selector because it controls padding,
            # not the Butterworth transfer curve itself.
            configured.add("boundary")
        return tuple(field for field in fields if field in configured)
    exact_integrations: Mapping[str, Mapping[str, set[str]]] = {
        "kernel_matrix": {
            "custom_convolution": {
                "kernel_width",
                "kernel_height",
                "kernel",
            },
        },
        "anchor_grid": {
            "resize_canvas": {"anchor"},
        },
        "compatible_image_picker": {
            "image_calculator": {"secondary_document_id"},
            "flat_field_correction": {"secondary_document_id"},
        },
    }
    configured = exact_integrations.get(editor, {}).get(
        operation_id,
        set(),
    )
    markers = _PROFESSIONAL_EDITOR_MARKERS.get(editor, ())
    if not all(
        hasattr(ImageProcessingWorkbench, marker)
        for marker in markers
    ):
        return ()
    return tuple(field for field in fields if field in configured)


def _rule_requirements_by_operation() -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = defaultdict(list)
    definitions = {
        definition.operation.value: definition
        for definition in _OPERATION_CATALOG
    }
    for rule in _PROFESSIONAL_EDITOR_RULES:
        operations = rule["operations"]
        assert isinstance(operations, Mapping)
        editor = str(rule["editor"])
        for operation_id, raw_fields in operations.items():
            definition = definitions.get(str(operation_id))
            if (
                definition is not None
                and not definition.available_for_new_recipe
            ):
                # Replay-only v1 steps are intentionally read-only; editable
                # professional controls would imply that creating new results
                # with the retired contract is supported.
                continue
            fields = tuple(str(field) for field in raw_fields)
            integrated_fields = _integrated_requirement_fields(
                editor,
                str(operation_id),
                fields,
            )
            missing_fields = tuple(
                field
                for field in fields
                if field not in integrated_fields
            )
            result[str(operation_id)].append(
                {
                    "editor": editor,
                    "priority": str(rule["priority"]),
                    "fields": list(fields),
                    "reason": str(rule["reason"]),
                    "integrated_fields": list(integrated_fields),
                    "missing_fields": list(missing_fields),
                    "integration_detected": not missing_fields,
                }
            )
    for gaps in result.values():
        gaps.sort(
            key=lambda item: (
                str(item["priority"]),
                str(item["editor"]),
            )
        )
    return dict(result)


def collect_image_parameter_audit() -> dict[str, object]:
    """Return a deterministic, JSON-compatible audit report."""

    registry = image_operation_registry()
    catalog = tuple(_OPERATION_CATALOG)
    definitions = {
        definition.operation.value: definition for definition in catalog
    }
    registry_ids = set(registry)
    catalogue_ids = set(definitions)
    rule_requirements = _rule_requirements_by_operation()

    operation_rows: list[dict[str, object]] = []
    field_kinds: list[str] = []
    explicit_help_count = 0
    effective_help_count = 0
    explicit_suffix_count = 0
    required_count = 0
    conditional_count = 0
    registry_field_count = 0
    catalogue_field_count = 0
    registry_only_fields: list[str] = []
    catalogue_only_fields: list[str] = []

    all_operation_ids = sorted(registry_ids | catalogue_ids)
    for operation_id in all_operation_ids:
        descriptor = registry.get(operation_id)
        definition = definitions.get(operation_id)
        service_parameters = (
            tuple(descriptor.parameter_schema)
            if descriptor is not None
            else ()
        )
        ui_parameters = (
            tuple(definition.parameters)
            if definition is not None
            else ()
        )
        service_by_key = {item.key: item for item in service_parameters}
        ui_by_key = {item.key: item for item in ui_parameters}
        service_keys = set(service_by_key)
        ui_keys = set(ui_by_key)
        missing_ui = sorted(service_keys - ui_keys)
        missing_service = sorted(ui_keys - service_keys)
        registry_only_fields.extend(
            f"{operation_id}.{key}" for key in missing_ui
        )
        catalogue_only_fields.extend(
            f"{operation_id}.{key}" for key in missing_service
        )
        registry_field_count += len(service_parameters)
        catalogue_field_count += len(ui_parameters)

        field_rows: list[dict[str, object]] = []
        for key in sorted(service_keys | ui_keys):
            schema = service_by_key.get(key)
            presentation = ui_by_key.get(key)
            kind = (
                schema.kind
                if schema is not None
                else presentation.kind
                if presentation is not None
                else "unknown"
            )
            if presentation is not None:
                catalogue_field_count += 0
                field_kinds.append(kind)
                explicit_help_count += int(bool(presentation.help_text))
                if definition is not None:
                    effective_help_count += int(
                        bool(
                            _parameter_help_text(
                                definition,
                                presentation,
                            )
                        )
                    )
                explicit_suffix_count += int(bool(presentation.suffix))
            if schema is not None:
                required_count += int(bool(schema.required))
                conditional_count += int(bool(schema.required_when))
            field_rows.append(
                {
                    "key": key,
                    "kind": kind,
                    "label": (
                        presentation.label
                        if presentation is not None
                        else None
                    ),
                    "has_explicit_help": bool(
                        presentation is not None
                        and presentation.help_text
                    ),
                    "suffix": (
                        presentation.suffix
                        if presentation is not None
                        else ""
                    ),
                    "required": bool(
                        schema is not None and schema.required
                    ),
                    "required_when": (
                        [
                            {"field": other, "equals": expected}
                            for other, expected in schema.required_when
                        ]
                        if schema is not None
                        else []
                    ),
                    "minimum": (
                        schema.minimum if schema is not None else None
                    ),
                    "maximum": (
                        schema.maximum if schema is not None else None
                    ),
                    "choice_count": (
                        len(schema.choices) if schema is not None else 0
                    ),
                    "in_service_registry": schema is not None,
                    "in_workbench_catalogue": presentation is not None,
                }
            )

        requirements = rule_requirements.get(operation_id, [])
        operation_rows.append(
            {
                "operation_id": operation_id,
                "name": (
                    definition.label
                    if definition is not None
                    else descriptor.chinese_name
                    if descriptor is not None
                    else operation_id
                ),
                "category": (
                    definition.category
                    if definition is not None
                    else descriptor.category
                    if descriptor is not None
                    else ""
                ),
                "available_for_new_recipe": bool(
                    definition is not None
                    and definition.available_for_new_recipe
                ),
                "parameter_count": len(ui_parameters),
                "missing_from_workbench": missing_ui,
                "missing_from_service_registry": missing_service,
                "parameters": field_rows,
                "professional_editor_requirements": requirements,
                "professional_editor_gaps": [
                    item
                    for item in requirements
                    if not bool(item["integration_detected"])
                ],
            }
        )

    gap_rows: list[dict[str, object]] = []
    fully_integrated_editors: set[str] = set()
    partially_integrated_editors: set[str] = set()
    for rule in sorted(
        _PROFESSIONAL_EDITOR_RULES,
        key=lambda item: (str(item["priority"]), str(item["editor"])),
    ):
        editor = str(rule["editor"])
        operations = rule["operations"]
        assert isinstance(operations, Mapping)
        operation_ids = sorted(str(item) for item in operations)
        field_occurrences = sum(
            len(tuple(fields)) for fields in operations.values()
        )
        requirements = [
            requirement
            for operation_id in operation_ids
            for requirement in rule_requirements.get(operation_id, ())
            if requirement["editor"] == editor
        ]
        missing_occurrences = sum(
            len(requirement["missing_fields"])
            for requirement in requirements
        )
        integrated_occurrences = field_occurrences - missing_occurrences
        if missing_occurrences == 0 and field_occurrences:
            fully_integrated_editors.add(editor)
        elif integrated_occurrences:
            partially_integrated_editors.add(editor)
        gap_rows.append(
            {
                "editor": str(rule["editor"]),
                "priority": str(rule["priority"]),
                "integration_detected": missing_occurrences == 0,
                "integrated_field_occurrence_count": integrated_occurrences,
                "missing_field_occurrence_count": missing_occurrences,
                "operation_count": len(operation_ids),
                "field_occurrence_count": field_occurrences,
                "operations": operation_ids,
                "reason": str(rule["reason"]),
            }
        )

    category_counts = Counter(
        definition.category for definition in catalog
    )
    parameter_occurrences = catalogue_field_count
    unique_parameter_keys = {
        parameter.key
        for definition in catalog
        for parameter in definition.parameters
    }
    conditional_operations = sorted(
        {
            operation.operation_id
            for operation in registry.values()
            if any(
                parameter.required_when
                for parameter in operation.parameter_schema
            )
        }
    )
    unresolved_gap_rows = [
        row
        for row in gap_rows
        if not bool(row["integration_detected"])
    ]
    classified_registry_only_fields = [
        {
            "field": field,
            "classification": _REGISTRY_ONLY_FIELD_NOTES[field][0],
            "reason": _REGISTRY_ONLY_FIELD_NOTES[field][1],
        }
        for field in sorted(registry_only_fields)
        if field in _REGISTRY_ONLY_FIELD_NOTES
    ]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "sources": {
            "service_registry": "fdm.services.image_processing.image_operation_registry",
            "workbench_catalogue": (
                "fdm.ui.image_processing_workbench._OPERATION_CATALOG"
            ),
        },
        "summary": {
            "registered_operation_count": len(registry),
            "workbench_operation_count": len(catalog),
            "parameter_occurrence_count": parameter_occurrences,
            "unique_parameter_key_count": len(unique_parameter_keys),
            "service_parameter_occurrence_count": registry_field_count,
            "operations_without_parameters": sum(
                not definition.parameters for definition in catalog
            ),
            "field_kinds": _sorted_counts(field_kinds),
            "explicit_help_count": explicit_help_count,
            "explicit_help_coverage_percent": _percentage(
                explicit_help_count,
                parameter_occurrences,
            ),
            "effective_help_count": effective_help_count,
            "effective_help_coverage_percent": _percentage(
                effective_help_count,
                parameter_occurrences,
            ),
            "explicit_suffix_count": explicit_suffix_count,
            "explicit_suffix_coverage_percent": _percentage(
                explicit_suffix_count,
                parameter_occurrences,
            ),
            "required_field_count": required_count,
            "conditional_required_field_count": conditional_count,
            "conditional_required_operations": conditional_operations,
            "registry_only_fields": sorted(registry_only_fields),
            "classified_registry_only_fields": (
                classified_registry_only_fields
            ),
            "unclassified_registry_only_fields": sorted(
                set(registry_only_fields)
                - set(_REGISTRY_ONLY_FIELD_NOTES)
            ),
            "workbench_only_fields": sorted(catalogue_only_fields),
            "professional_editor_requirement_types": len(gap_rows),
            "professional_editor_integrated_types": sorted(
                fully_integrated_editors
            ),
            "professional_editor_partially_integrated_types": sorted(
                partially_integrated_editors
            ),
            "professional_editor_gap_types": len(unresolved_gap_rows),
            "professional_editor_gap_occurrences": sum(
                int(row["missing_field_occurrence_count"])
                for row in unresolved_gap_rows
            ),
        },
        "categories": [
            {"name": name, "operation_count": category_counts[name]}
            for name in sorted(category_counts)
        ],
        "professional_editor_requirements": gap_rows,
        "professional_editor_gaps": unresolved_gap_rows,
        "operations": operation_rows,
    }


def render_chinese_summary(report: Mapping[str, object]) -> str:
    """Render the report as a concise Chinese terminal summary."""

    summary = report["summary"]
    assert isinstance(summary, Mapping)
    field_kinds = summary["field_kinds"]
    assert isinstance(field_kinds, Sequence)
    kind_text = "、".join(
        f"{item['name']} {item['count']}"
        for item in field_kinds
        if isinstance(item, Mapping)
    )
    gaps = report["professional_editor_gaps"]
    assert isinstance(gaps, Sequence)
    gap_lines = []
    for item in gaps:
        if not isinstance(item, Mapping):
            continue
        gap_lines.append(
            f"  - {item['priority']} {item['editor']}："
            f"{item['operation_count']} 项操作 / "
            f"{item['missing_field_occurrence_count']} 个未覆盖字段位置"
        )
    mismatch_count = len(summary["registry_only_fields"]) + len(
        summary["workbench_only_fields"]
    )
    lines = [
        f"图像处理参数审计（schema {report['schema_version']}）",
        (
            f"- 操作：服务注册表 {summary['registered_operation_count']}，"
            f"工作台 {summary['workbench_operation_count']}"
        ),
        (
            f"- 参数：{summary['parameter_occurrence_count']} 个字段位置，"
            f"{summary['unique_parameter_key_count']} 个唯一键"
        ),
        f"- 基础控件类型：{kind_text}",
        (
            f"- 显式帮助：{summary['explicit_help_count']} / "
            f"{summary['parameter_occurrence_count']} "
            f"（{summary['explicit_help_coverage_percent']}%）"
        ),
        (
            f"- 运行时中文说明：{summary['effective_help_count']} / "
            f"{summary['parameter_occurrence_count']} "
            f"（{summary['effective_help_coverage_percent']}%）"
        ),
        (
            f"- 显式单位后缀：{summary['explicit_suffix_count']} / "
            f"{summary['parameter_occurrence_count']} "
            f"（{summary['explicit_suffix_coverage_percent']}%；"
            "该指标仅表示声明覆盖，不代表所有字段都应有单位）"
        ),
        (
            f"- required_when 条件字段："
            f"{summary['conditional_required_field_count']}；"
            f"注册表/工作台字段差异：{mismatch_count}"
            f"（未分类 "
            f"{len(summary['unclassified_registry_only_fields'])}）"
        ),
        (
            f"- 疑似专业编辑器缺口："
            f"{summary['professional_editor_gap_types']} 类 / "
            f"{summary['professional_editor_gap_occurrences']} 个字段位置"
        ),
        *gap_lines,
    ]
    return "\n".join(lines)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读审计图像处理工作台参数契约与专业控件覆盖。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出稳定、UTF-8、可机器读取的 JSON。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    report = collect_image_parameter_audit()
    if args.json:
        print(
            json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
    else:
        print(render_chinese_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
