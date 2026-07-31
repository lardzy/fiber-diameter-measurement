from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fdm.analysis_artifacts import AnalysisCurve
from fdm.cancellation import CancellationTokenSource
from fdm.raster import RasterPixelType, RasterPlane
from fdm.ui.analysis_parameters_dialog import (
    ANALYSIS_PARAMETER_SCHEMAS,
    AnalysisParametersDialog,
    ProfilePreviewContext,
    analysis_parameter_schema,
    execute_profile_preview_task,
)
from fdm.ui.image_analysis_controller import (
    AnalysisCalibrationSnapshot,
    AnalysisTool,
    ImageAnalysisTaskRequest,
    ImageAnalysisTaskResult,
)
from fdm.services.analysis_profiles import (
    AnalysisMeasurementProfile,
    AnalysisMeasurementProfileStore,
    analysis_output_field_schema,
)


def test_every_analysis_tool_has_one_chinese_validatable_schema() -> None:
    assert set(ANALYSIS_PARAMETER_SCHEMAS) == set(AnalysisTool)
    for tool, schema in ANALYSIS_PARAMETER_SCHEMAS.items():
        assert schema.tool is tool
        assert schema.chinese_name
        assert schema.version
        assert schema.validate({}) == schema.defaults()
        json_schema = schema.to_json_schema()
        assert json_schema["additionalProperties"] is False
        assert set(json_schema["properties"]) == {
            field.key for field in schema.fields
        }


def test_maxima_schema_keeps_v1_default_and_exposes_topographic_v2() -> None:
    schema = analysis_parameter_schema(AnalysisTool.MAXIMA)

    defaults = schema.defaults()
    version_field = next(
        field for field in schema.fields if field.key == "algorithm_version"
    )

    assert defaults["algorithm_version"] == "1"
    assert [value for _label, value in version_field.choices] == ["1", "2"]


def test_dialog_uses_schema_defaults_and_returns_typed_parameters() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = AnalysisParametersDialog(
        AnalysisTool.HISTOGRAM,
        {"bins": 32, "log_counts": True},
    )

    parameters = dialog.parameters()

    assert parameters["bins"] == 32
    assert parameters["log_counts"] is True
    assert parameters["channel"] == "luminance"
    dialog.close()
    app.processEvents()


def test_core_analysis_output_fields_are_independent_and_default_to_all() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = AnalysisParametersDialog(
        AnalysisTool.INTENSITY,
        {"channel": "luminance"},
    )
    schema = analysis_output_field_schema("fdm.intensity")
    assert schema is not None
    original_parameters = dialog.parameters()

    assert dialog.output_fields() == schema.default_fields
    dialog.set_output_fields(("central_tendency", "percentiles"))
    assert dialog.output_fields() == ("central_tendency", "percentiles")
    assert dialog.parameters() == original_parameters

    dialog.close()
    app.processEvents()


def test_legacy_profile_without_output_fields_loads_as_all_outputs(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    store = AnalysisMeasurementProfileStore(tmp_path / "profiles.json")
    schema = analysis_parameter_schema(AnalysisTool.SHAPE)
    profile = AnalysisMeasurementProfile(
        profile_id="legacy-shape",
        name="旧版形状预设",
        tool_id="fdm.shape",
        tool_version=schema.version,
        parameters=schema.defaults(),
        output_fields=None,
    )
    store.save((profile,))
    dialog = AnalysisParametersDialog(
        AnalysisTool.SHAPE,
        profile_store=store,
    )
    output_schema = analysis_output_field_schema("fdm.shape")
    assert output_schema is not None

    index = dialog.profile_controls.combo.findData(profile.profile_id)
    assert index >= 0
    dialog.profile_controls.combo.setCurrentIndex(index)

    assert dialog.output_fields() == output_schema.default_fields
    dialog.close()
    app.processEvents()


def _profile_preview_context(
    *,
    line_points: tuple[tuple[float, float], ...] = ((2.0, 3.0), (12.0, 3.0)),
    rectangle_points: tuple[tuple[float, float], ...] = (),
) -> ProfilePreviewContext:
    pixels = np.arange(20 * 16, dtype=np.uint16).reshape((16, 20))
    return ProfilePreviewContext(
        plane=RasterPlane(
            width=20,
            height=16,
            pixel_type=RasterPixelType.GRAY16,
            data=pixels.astype("<u2").tobytes(),
        ),
        document_id="doc-profile-preview",
        calibration=AnalysisCalibrationSnapshot(
            pixel_size_x=0.5,
            pixel_size_y=0.5,
            unit="µm",
        ),
        line_points=line_points,
        rectangle_points=rectangle_points,
    )


def test_profile_preview_debounces_changes_and_uses_frozen_raw_points() -> None:
    app = QApplication.instance() or QApplication([])
    context = _profile_preview_context()
    dialog = AnalysisParametersDialog(
        AnalysisTool.PROFILE,
        profile_preview_context=context,
    )
    assert dialog._profile_preview_timer is not None
    assert dialog._profile_preview_controller is not None
    dialog._profile_preview_timer.stop()
    start = MagicMock(
        return_value=SimpleNamespace(
            request_id="preview-current",
            generation=7,
        )
    )
    dialog._profile_preview_controller.start = start  # type: ignore[method-assign]

    width_editor = dialog._editors["line_width"]
    width_editor.setValue(2.0)
    QTest.qWait(60)
    width_editor.setValue(3.0)
    QTest.qWait(100)
    assert start.call_count == 0
    QTest.qWait(70)
    app.processEvents()

    assert start.call_count == 1
    parameters = start.call_args.kwargs["parameters"]
    assert parameters["points"] == context.line_points
    assert parameters["line_width"] == 3.0
    assert dialog._profile_preview_request_id == "preview-current"
    assert dialog._profile_preview_generation == 7
    dialog.reject()
    app.processEvents()


def test_profile_preview_reports_missing_selection_without_starting_worker() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = AnalysisParametersDialog(
        AnalysisTool.PROFILE,
        profile_preview_context=_profile_preview_context(line_points=()),
    )
    assert dialog._profile_preview_timer is not None
    assert dialog._profile_preview_controller is not None
    dialog._profile_preview_timer.stop()
    start = MagicMock()
    dialog._profile_preview_controller.start = start  # type: ignore[method-assign]

    dialog._start_profile_preview()

    assert start.call_count == 0
    assert dialog._profile_preview_status is not None
    assert "没有可预览的线段或折线" in dialog._profile_preview_status.text()
    dialog.reject()
    app.processEvents()


def test_profile_preview_ignores_late_result_and_closes_private_controller() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = AnalysisParametersDialog(
        AnalysisTool.PROFILE,
        profile_preview_context=_profile_preview_context(),
    )
    assert dialog._profile_preview_timer is not None
    assert dialog._profile_preview_controller is not None
    dialog._profile_preview_timer.stop()
    dialog._profile_preview_request_id = "current"
    dialog._profile_preview_generation = 9
    assert dialog._profile_preview_status is not None
    dialog._profile_preview_status.setText("保持不变")
    curve = AnalysisCurve(
        name="强度剖面",
        x=(0.0, 1.0),
        y=(10.0, 20.0),
        x_unit="µm",
        y_unit="强度",
    )
    stale = ImageAnalysisTaskResult(
        tool=AnalysisTool.PROFILE,
        request_id="stale",
        generation=8,
        document_id="doc-profile-preview",
        source_pixel_revision=0,
        source_reference=None,
        calibration_signature=None,
        parameters={"sample_spacing": 1.0},
        scalars={"valid_sample_count": 2, "sample_count": 2},
        curves=(curve,),
    )
    dialog._on_profile_preview_ready(stale)
    assert dialog._profile_preview_status.text() == "保持不变"

    current = ImageAnalysisTaskResult(
        tool=AnalysisTool.PROFILE,
        request_id="current",
        generation=9,
        document_id="doc-profile-preview",
        source_pixel_revision=0,
        source_reference=None,
        calibration_signature=None,
        parameters={"sample_spacing": 1.0},
        scalars={"valid_sample_count": 2, "sample_count": 2},
        curves=(curve,),
    )
    dialog._on_profile_preview_ready(current)
    assert "有效 2/2 点" in dialog._profile_preview_status.text()

    close = MagicMock(wraps=dialog._profile_preview_controller.close)
    dialog._profile_preview_controller.close = close  # type: ignore[method-assign]
    dialog.reject()
    app.processEvents()
    close.assert_called_once_with()
    assert dialog._profile_preview_closed is True
    assert dialog._profile_preview_request_id is None


def test_profile_preview_executor_crops_pixels_before_analysis() -> None:
    context = _profile_preview_context(
        line_points=((8.0, 7.0), (12.0, 7.0)),
    )
    request = ImageAnalysisTaskRequest(
        tool=AnalysisTool.PROFILE,
        request_id="crop-check",
        generation=3,
        document_id=context.document_id,
        source_pixel_revision=0,
        plane=context.plane,
        calibration=context.calibration,
        parameters={
            "points": context.line_points,
            "aggregation": "line",
            "line_width": 1.0,
            "sample_spacing": 1.0,
            "channel": "luminance",
        },
    )
    sentinel = object()
    token = CancellationTokenSource().token
    with patch(
        "fdm.ui.analysis_parameters_dialog.execute_analysis_task",
        return_value=sentinel,
    ) as execute:
        result = execute_profile_preview_task(request, token, lambda phase: None)

    assert result is sentinel
    bounded = execute.call_args.args[0]
    assert bounded.request_id == request.request_id
    assert bounded.generation == request.generation
    assert bounded.plane.width < request.plane.width
    assert bounded.plane.height < request.plane.height
    assert bounded.parameters["points"][0] == [3.0, 3.0]


def test_profile_preview_controller_delivers_curve_without_formal_controller() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = AnalysisParametersDialog(
        AnalysisTool.PROFILE,
        profile_preview_context=_profile_preview_context(),
    )
    assert dialog._profile_preview_status is not None

    for _attempt in range(100):
        if "有效" in dialog._profile_preview_status.text():
            break
        QTest.qWait(20)
        app.processEvents()

    assert "有效" in dialog._profile_preview_status.text()
    assert dialog._profile_preview_curve is not None
    assert dialog._profile_preview_curve._curve is not None
    dialog.reject()
    app.processEvents()
