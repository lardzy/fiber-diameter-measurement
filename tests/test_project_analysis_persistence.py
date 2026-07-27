from __future__ import annotations

import json
from pathlib import Path

import pytest

from fdm.analysis_artifacts import (
    AnalysisArtifact,
    AnalysisArtifactStatus,
    AnalysisDependencySignature,
    AnalysisObjectKind,
    AnalysisObjectReference,
    calibration_signature_from_values,
)
from fdm.geometry import Line, Point
from fdm.models import Calibration, ImageDocument, Measurement, ProjectState
from fdm.project_io import ProjectIO
from fdm.project_roi import ProjectRoi, RectangleRoiGeometry


def _document() -> ImageDocument:
    document = ImageDocument(
        id="image_1",
        path="images/source.png",
        image_size=(64, 48),
        calibration=Calibration(
            pixels_per_unit=7.5,
            unit="um",
            mode="project",
            source_label="项目统一标定",
        ),
        measurements=[
            Measurement(
                id="measurement_1",
                image_id="image_1",
                fiber_group_id=None,
                mode="manual",
                measurement_kind="line",
                line_px=Line(Point(1, 1), Point(9, 1)),
            )
        ],
    )
    document.initialize_runtime_state()
    return document


def _roi(*, revision: int = 2) -> ProjectRoi:
    return ProjectRoi(
        id="roi_1",
        document_id="image_1",
        name="检验区域",
        geometry=RectangleRoiGeometry(1, 2, 10, 8),
        revision=revision,
    )


def _artifact(
    *,
    artifact_id: str = "analysis_1",
    pixel_revision: int = 4,
    reference: AnalysisObjectReference | None = None,
) -> AnalysisArtifact:
    return AnalysisArtifact(
        id=artifact_id,
        source_document_id="image_1",
        source_pixel_revision=pixel_revision,
        source_reference=reference
        or AnalysisObjectReference(AnalysisObjectKind.ROI, "roi_1", 2),
        tool_id="fdm.histogram",
        tool_version="1",
        parameters={"bins": 64},
        calibration_signature=calibration_signature_from_values(
            pixels_per_unit=7.5,
            unit="um",
        ),
        scalars={"n": 100},
        created_at="2026-07-27T08:00:00+00:00",
    )


def test_legacy_project_remains_sparse_without_new_fields() -> None:
    payload = {
        "version": "0.1.0",
        "documents": [],
        "metadata": {},
    }

    project = ProjectState.from_dict(payload)
    serialized = project.to_dict()

    assert project.project_rois == []
    assert project.analysis_artifacts == []
    assert "project_rois" not in serialized
    assert "analysis_artifacts" not in serialized


def test_project_io_roundtrip_persists_roi_and_analysis_without_inline_assets(
    tmp_path: Path,
) -> None:
    project = ProjectState(
        version="0.3.8",
        documents=[_document()],
        project_rois=[_roi()],
        analysis_artifacts=[_artifact()],
    )
    target = tmp_path / "analysis.fdmproj"

    ProjectIO.save(project, target)
    raw_payload = json.loads(target.read_text(encoding="utf-8"))
    restored = ProjectIO.load(target)

    assert raw_payload["project_rois"] == [project.project_rois[0].to_dict()]
    assert raw_payload["analysis_artifacts"] == [
        project.analysis_artifacts[0].to_dict()
    ]
    assert restored.project_rois == project.project_rois
    assert restored.analysis_artifacts == project.analysis_artifacts
    encoded = json.dumps(raw_payload, ensure_ascii=False, allow_nan=False)
    assert "pickle" not in encoded.lower()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("project_rois", {}),
        ("analysis_artifacts", None),
        ("project_rois", ["not-an-object"]),
        ("analysis_artifacts", ["not-an-object"]),
    ],
)
def test_project_rejects_invalid_new_collection_boundaries(
    field_name: str,
    value: object,
) -> None:
    payload = {
        "version": "0.3.8",
        "documents": [],
        field_name: value,
    }

    with pytest.raises(TypeError):
        ProjectState.from_dict(payload)


def test_project_rejects_duplicate_roi_and_artifact_ids() -> None:
    roi_payload = _roi().to_dict()
    artifact_payload = _artifact().to_dict()

    with pytest.raises(ValueError, match="project_rois.*重复 ID"):
        ProjectState.from_dict(
            {
                "version": "0.3.8",
                "documents": [],
                "project_rois": [roi_payload, roi_payload],
            }
        )
    with pytest.raises(ValueError, match="analysis_artifacts.*重复 ID"):
        ProjectState.from_dict(
            {
                "version": "0.3.8",
                "documents": [],
                "analysis_artifacts": [
                    artifact_payload,
                    artifact_payload,
                ],
            }
        )


def test_project_level_refresh_marks_stale_and_preserves_results() -> None:
    artifact = _artifact()
    other = AnalysisArtifact(
        id="other",
        source_document_id="image_2",
        source_pixel_revision=0,
        tool_id="fdm.histogram",
        tool_version="1",
        created_at="2026-07-27T08:00:00+00:00",
    )
    project = ProjectState(
        version="0.3.8",
        documents=[_document()],
        project_rois=[_roi()],
        analysis_artifacts=[artifact, other],
    )

    changed = project.refresh_analysis_validity(
        "image_1",
        current_pixel_revision=5,
    )

    assert changed == 1
    assert len(project.analysis_artifacts) == 2
    assert project.analysis_artifacts[0].status is AnalysisArtifactStatus.STALE
    assert project.analysis_artifacts[0].stale_reason == "来源图片像素已变化"
    assert project.analysis_artifacts[0].scalars == artifact.scalars
    assert project.analysis_artifacts[1] is other
    assert project.refresh_analysis_validity(
        "image_1",
        current_pixel_revision=5,
    ) == 0


def test_project_level_refresh_uses_roi_and_measurement_geometry_revisions() -> None:
    document = _document()
    measurement = document.measurements[0]
    measurement_artifact = _artifact(
        artifact_id="measurement_analysis",
        reference=AnalysisObjectReference(
            AnalysisObjectKind.MEASUREMENT,
            measurement.id,
            measurement.geometry_revision,
        ),
    )
    roi_artifact = _artifact(artifact_id="roi_analysis")
    project = ProjectState(
        version="0.3.8",
        documents=[document],
        project_rois=[_roi(revision=3)],
        analysis_artifacts=[roi_artifact, measurement_artifact],
    )

    measurement.replace_line_geometry(
        line_px=Line(Point(2, 2), Point(12, 2)),
        calibration=document.calibration,
    )
    changed = project.refresh_analysis_validity(
        "image_1",
        current_pixel_revision=4,
    )

    assert changed == 2
    assert project.analysis_artifacts[0].stale_reason == "引用的ROI几何已变化"
    assert (
        project.analysis_artifacts[1].stale_reason
        == "引用的测量对象几何已变化"
    )


def test_project_dependency_probe_detects_transitive_roi_change() -> None:
    document = _document()
    roi = _roi(revision=3)
    calibration = document.calibration
    assert calibration is not None
    dependency = AnalysisDependencySignature(
        calibration={
            "signature": calibration_signature_from_values(
                pixels_per_unit=calibration.pixels_per_unit,
                unit=calibration.unit,
            ),
            "pixel_size_x": 1.0 / calibration.pixels_per_unit,
            "pixel_size_y": 1.0 / calibration.pixels_per_unit,
            "unit": calibration.unit,
        },
        roi_transitive_refs={
            roi.id: {
                "revision": roi.revision,
                "kind": roi.kind.value,
            }
        },
    )
    project = ProjectState(
        version="0.3.8",
        documents=[document],
        project_rois=[roi],
    )

    assert project.analysis_dependency_is_current(document.id, dependency)
    project.project_rois = [_roi(revision=4)]
    assert not project.analysis_dependency_is_current(
        document.id,
        dependency,
    )


def test_project_level_refresh_marks_removed_document_stale() -> None:
    artifact = _artifact()
    project = ProjectState(
        version="0.3.8",
        documents=[],
        analysis_artifacts=[artifact],
    )

    changed = project.refresh_analysis_validity("image_1")

    assert changed == 1
    assert project.analysis_artifacts[0].stale_reason == "来源文档已不存在"


def test_nested_unknown_and_non_finite_fields_remain_rejected() -> None:
    roi_payload = _roi().to_dict()
    roi_payload["geometry"]["unexpected"] = True
    with pytest.raises(ValueError, match="未知"):
        ProjectState.from_dict(
            {
                "version": "0.3.8",
                "documents": [],
                "project_rois": [roi_payload],
            }
        )

    artifact_payload = _artifact().to_dict()
    artifact_payload["parameters"]["bad"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        ProjectState.from_dict(
            {
                "version": "0.3.8",
                "documents": [],
                "analysis_artifacts": [artifact_payload],
            }
        )
