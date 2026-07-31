from __future__ import annotations

import json

from fdm.image_parameter_audit import (
    AUDIT_SCHEMA_VERSION,
    collect_image_parameter_audit,
    main,
    render_chinese_summary,
)
from fdm.services.image_processing import image_operation_registry
from fdm.ui.image_processing_workbench import (
    _OPERATION_CATALOG,
    default_operation_spec,
)


def test_parameter_audit_is_deterministic_and_matches_both_registries() -> None:
    first = collect_image_parameter_audit()
    second = collect_image_parameter_audit()

    assert first == second
    assert first["schema_version"] == AUDIT_SCHEMA_VERSION
    assert json.dumps(
        first,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ) == json.dumps(
        second,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )

    summary = first["summary"]
    assert summary["registered_operation_count"] == len(
        image_operation_registry()
    )
    assert summary["workbench_operation_count"] == len(_OPERATION_CATALOG)
    assert summary["parameter_occurrence_count"] == sum(
        len(definition.parameters)
        for definition in _OPERATION_CATALOG
    )
    assert summary["service_parameter_occurrence_count"] == sum(
        len(descriptor.parameter_schema)
        for descriptor in image_operation_registry().values()
    )


def test_parameter_audit_reports_contract_conditions_and_editor_gaps() -> None:
    report = collect_image_parameter_audit()
    summary = report["summary"]

    assert {
        "threshold.foreground_value",
        "threshold.background_value",
        "adjust_levels.output_min",
        "adjust_levels.output_max",
    }.issubset(summary["registry_only_fields"])
    assert summary["unclassified_registry_only_fields"] == []
    assert len(summary["classified_registry_only_fields"]) == len(
        summary["registry_only_fields"]
    )
    assert summary["workbench_only_fields"] == []
    assert summary["effective_help_count"] == summary[
        "parameter_occurrence_count"
    ]
    assert summary["effective_help_coverage_percent"] == 100.0
    assert summary["professional_editor_gap_occurrences"] == 0
    assert summary["conditional_required_operations"] == [
        "fft_filter",
        "flat_field_correction",
    ]

    operations = {
        item["operation_id"]: item for item in report["operations"]
    }
    threshold = operations["threshold"]
    assert threshold["professional_editor_gaps"] == []
    assert "histogram_range" in summary[
        "professional_editor_integrated_types"
    ]
    convolution = operations["custom_convolution"]
    assert {
        requirement["editor"]
        for requirement in convolution[
            "professional_editor_requirements"
        ]
    } == {"kernel_matrix"}
    assert convolution["professional_editor_gaps"] == []
    assert "kernel_matrix" in summary[
        "professional_editor_integrated_types"
    ]
    assert operations["brightness_contrast"][
        "professional_editor_gaps"
    ] == []
    assert operations["stripe_suppression"][
        "professional_editor_gaps"
    ] == []
    assert operations["fft_filter"]["professional_editor_gaps"] == []
    fft_pixel_size = next(
        item
        for item in operations["fft_filter"]["parameters"]
        if item["key"] == "pixel_size"
    )
    assert fft_pixel_size["required_when"] == [
        {"field": "frequency_unit", "equals": "cycles_per_unit"}
    ]

def test_every_new_workbench_step_uses_the_registered_current_version() -> None:
    registry = image_operation_registry()
    for definition in _OPERATION_CATALOG:
        operation = default_operation_spec(
            definition.operation,
            640,
            480,
            secondary_document_id="secondary",
        )
        assert operation.implementation == "fdm"
        assert (
            operation.implementation_version
            == registry[operation.operation_id].version
        )


def test_parameter_audit_summary_and_cli_outputs(capsys) -> None:
    report = collect_image_parameter_audit()
    summary = render_chinese_summary(report)
    assert "图像处理参数审计" in summary
    assert "疑似专业编辑器缺口" in summary

    assert main([]) == 0
    assert "图像处理参数审计" in capsys.readouterr().out

    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == AUDIT_SCHEMA_VERSION
    assert payload["summary"] == report["summary"]
