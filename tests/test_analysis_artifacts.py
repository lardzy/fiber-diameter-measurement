from __future__ import annotations

import json
import math

import pytest

from fdm.analysis_artifacts import (
    AnalysisArtifact,
    AnalysisArtifactStatus,
    AnalysisAssetKind,
    AnalysisAssetReference,
    AnalysisCurve,
    AnalysisDependencySignature,
    AnalysisObjectKind,
    AnalysisObjectReference,
    AnalysisRegionSnapshot,
    AnalysisSourceDescriptor,
    AnalysisTable,
    AnalysisToolSpec,
    calibration_signature_from_values,
    refresh_artifact_validity,
    refresh_artifacts_validity,
)


def _artifact(**changes) -> AnalysisArtifact:
    values = {
        "id": "analysis_1",
        "source_document_id": "image_1",
        "source_pixel_revision": 3,
        "source_reference": AnalysisObjectReference(
            AnalysisObjectKind.ROI,
            "roi_1",
            7,
        ),
        "tool_id": "fdm.histogram",
        "tool_version": "1",
        "parameters": {"channel": "luminance", "bins": 256},
        "calibration_signature": "sha256:" + ("a" * 64),
        "scalars": {"n": 100, "mean": 12.5},
        "tables": (
            AnalysisTable(
                name="统计",
                columns=("名称", "值"),
                rows=(("均值", 12.5), ("有效 N", 100)),
            ),
        ),
        "curves": (
            AnalysisCurve(
                name="直方图",
                x=(0.0, 1.0, 2.0),
                y=(4.0, None, 2.0),
                x_unit="灰度",
                y_unit="频数",
            ),
        ),
        "assets": (
            AnalysisAssetReference(
                kind=AnalysisAssetKind.MASK,
                path="analysis/analysis_1/mask.npz",
                sha256="b" * 64,
                media_type="application/x-npz",
                metadata={"schema": "fdm.mask.v1", "shape": [10, 10]},
            ),
        ),
        "created_at": "2026-07-27T08:00:00+00:00",
    }
    values.update(changes)
    return AnalysisArtifact(**values)


def test_full_artifact_roundtrip_is_finite_and_lossless() -> None:
    original = _artifact()

    payload = original.to_dict()
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    )
    restored = AnalysisArtifact.from_dict(json.loads(encoded))

    assert restored == original
    assert restored.parameters == {"bins": 256, "channel": "luminance"}
    assert restored.scalars == {"mean": 12.5, "n": 100}
    assert restored.is_current
    assert restored.stale_reason is None


def test_artifact_warnings_roundtrip_and_survive_status_change() -> None:
    artifact = _artifact(
        warnings=("精确掩膜面积与矢量描述符来自不同来源，请复核。",),
    )

    restored = AnalysisArtifact.from_dict(artifact.to_dict())
    stale = restored.mark_stale("来源像素已变化")

    assert restored.warnings == artifact.warnings
    assert stale.warnings == artifact.warnings
    with pytest.raises(TypeError, match="字符串列表"):
        _artifact(warnings="不是列表")


def test_scientific_provenance_value_objects_roundtrip_and_hash_dependencies() -> None:
    region = AnalysisRegionSnapshot(
        mask_sha256="1" * 64,
        pixel_center_rule="integer-coordinate-is-pixel-center",
        components=2,
        holes=1,
        rings=(
            ((0.0, 0.0), (5.0, 0.0), (5.0, 5.0)),
            ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0)),
        ),
        source="measurement:m-1@revision-4",
    )
    source = AnalysisSourceDescriptor(
        kind="digital_slide_viewport",
        pixel_sha256="2" * 64,
        store_id="slide-1",
        focus=3,
        origin=(-20, 40),
        viewport_size=(1024, 768),
    )
    dependency = AnalysisDependencySignature(
        calibration={"signature": "sha256:" + ("3" * 64)},
        roi_transitive_refs={"roi-1": {"revision": 4, "parents": ["roi-root"]}},
        measurement_revisions={"m-1": 7},
        point_set={"sha256": "4" * 64, "count": 12},
        group={"id": "g-1", "revision": 2},
        study_region=region.to_dict(),
    )
    spec = AnalysisToolSpec(
        tool_id="fdm.shape",
        version="2",
        chinese_name="形状分析",
        parameter_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "required": ["component_count"]},
        convertible_kinds=("measurement", "roi"),
    )

    assert AnalysisRegionSnapshot.from_dict(region.to_dict()) == region
    assert AnalysisSourceDescriptor.from_dict(source.to_dict()) == source
    assert AnalysisDependencySignature.from_dict(dependency.to_dict()) == dependency
    assert AnalysisToolSpec.from_dict(spec.to_dict()) == spec
    assert dependency.sha256 == AnalysisDependencySignature(
        calibration={"signature": "sha256:" + ("3" * 64)},
        roi_transitive_refs={"roi-1": {"parents": ["roi-root"], "revision": 4}},
        measurement_revisions={"m-1": 7},
        point_set={"count": 12, "sha256": "4" * 64},
        group={"revision": 2, "id": "g-1"},
        study_region=region.to_dict(),
    ).sha256

    artifact = _artifact(
        region_snapshot=region,
        source_descriptor=source,
        dependency_signature=dependency,
        tool_version=spec.version,
    )
    assert AnalysisArtifact.from_dict(artifact.to_dict()) == artifact


def test_dependency_signature_rejects_tampered_payload() -> None:
    dependency = AnalysisDependencySignature(
        measurement_revisions={"m-1": 1},
    )
    payload = dependency.to_dict()
    payload["dependencies"]["measurement_revisions"]["m-1"] = 2

    with pytest.raises(ValueError, match="不一致"):
        AnalysisDependencySignature.from_dict(payload)


def test_dependency_signature_change_or_missing_dependency_marks_stale() -> None:
    dependency = AnalysisDependencySignature(
        measurement_revisions={"count-1": 2},
        point_set={"measurement_ids": ["count-1"], "sha256": "a" * 64},
    )
    artifact = _artifact(
        source_reference=None,
        dependency_signature=dependency,
    )

    current = refresh_artifact_validity(
        artifact,
        current_dependency_signatures={artifact.id: dependency.sha256},
    )
    changed = refresh_artifact_validity(
        artifact,
        current_dependency_signatures={artifact.id: "b" * 64},
    )
    missing = refresh_artifact_validity(
        artifact,
        current_dependency_signatures={artifact.id: None},
    )

    assert current is artifact
    assert changed.stale_reason == "分析依赖已变化"
    assert missing.stale_reason == "分析依赖已不存在或无法验证"


def test_source_descriptor_sha_or_viewport_change_marks_stale() -> None:
    frozen = AnalysisSourceDescriptor(
        kind="digital_slide_viewport",
        pixel_sha256="1" * 64,
        store_id="slide-1",
        focus=2,
        origin=(100, 200),
        viewport_size=(1024, 768),
    )
    artifact = _artifact(
        source_reference=None,
        source_descriptor=frozen,
    )

    assert (
        refresh_artifact_validity(
            artifact,
            current_source_descriptor=frozen,
        )
        is artifact
    )
    changed_pixels = refresh_artifact_validity(
        artifact,
        current_source_descriptor=AnalysisSourceDescriptor(
            kind="digital_slide_viewport",
            pixel_sha256="2" * 64,
            store_id="slide-1",
            focus=2,
            origin=(100, 200),
            viewport_size=(1024, 768),
        ),
    )
    changed_viewport = refresh_artifact_validity(
        artifact,
        current_source_descriptor=AnalysisSourceDescriptor(
            kind="digital_slide_viewport",
            pixel_sha256="1" * 64,
            store_id="slide-1",
            focus=2,
            origin=(101, 200),
            viewport_size=(1024, 768),
        ),
    )

    assert changed_pixels.stale_reason == "来源图片内容或冻结视窗已变化"
    assert changed_viewport.stale_reason == "来源图片内容或冻结视窗已变化"


def test_legacy_artifact_without_source_descriptor_skips_new_sha_check() -> None:
    artifact = _artifact(source_reference=None, source_descriptor=None)

    assert (
        refresh_artifact_validity(
            artifact,
            current_source_descriptor=AnalysisSourceDescriptor(
                kind="raster",
                pixel_sha256="3" * 64,
            ),
        )
        is artifact
    )


def test_schema_v1_artifact_remains_readable() -> None:
    payload = _artifact().to_dict()
    payload["schema_version"] = 1

    restored = AnalysisArtifact.from_dict(payload)

    assert restored.tool_version == "1"
    assert restored.region_snapshot is None
    assert restored.source_descriptor is None
    assert restored.dependency_signature is None


def test_parameters_scalars_and_asset_metadata_are_defensive_copies() -> None:
    parameters = {"nested": {"items": [1, 2]}}
    scalars = {"n": 2}
    metadata = {"schema": "fdm.label-image.v1", "shape": [2, 3]}
    asset = AnalysisAssetReference(
        kind="label_image",
        path="analysis/labels.npz",
        sha256="c" * 64,
        media_type="application/x-npz",
        metadata=metadata,
    )
    artifact = _artifact(
        parameters=parameters,
        scalars=scalars,
        assets=(asset,),
    )

    parameters["nested"]["items"].append(3)
    scalars["n"] = 99
    metadata["shape"].append(4)
    returned_parameters = artifact.parameters
    returned_parameters["nested"]["items"].append(5)
    returned_metadata = artifact.assets[0].metadata
    returned_metadata["shape"].append(6)

    assert artifact.parameters == {"nested": {"items": [1, 2]}}
    assert artifact.scalars == {"n": 2}
    assert artifact.assets[0].metadata == {
        "schema": "fdm.label-image.v1",
        "shape": [2, 3],
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"parameters": {"bad": math.nan}},
        {"parameters": {"bad": math.inf}},
        {"scalars": {"bad": math.nan}},
        {"source_pixel_revision": True},
        {"tool_id": "Histogram 中文"},
        {"created_at": "2026-07-27T08:00:00"},
    ],
)
def test_artifact_rejects_non_json_or_ambiguous_values(changes) -> None:
    with pytest.raises((TypeError, ValueError)):
        _artifact(**changes)


def test_nested_non_json_value_and_curve_nan_are_rejected() -> None:
    with pytest.raises(TypeError):
        _artifact(parameters={"bad": {1, 2}})
    with pytest.raises(ValueError):
        AnalysisCurve(name="曲线", x=(0.0,), y=(math.nan,))
    with pytest.raises(ValueError):
        AnalysisTable(name="表", columns=("值",), rows=((math.inf,),))


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/mask.npz",
        "../outside.npz",
        "analysis/../outside.npz",
        r"C:\absolute\mask.npz",
        "processed/mask.npz",
    ],
)
def test_asset_reference_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError):
        AnalysisAssetReference(
            kind="mask",
            path=path,
            sha256="d" * 64,
            media_type="application/x-npz",
            metadata={"schema": "fdm.mask.v1"},
        )


@pytest.mark.parametrize(
    ("path", "media_type", "metadata"),
    [
        (
            "analysis/result.pkl",
            "application/octet-stream",
            {"schema": "fdm.table.v1"},
        ),
        (
            "analysis/result.npz",
            "application/x-python-pickle",
            {"schema": "fdm.table.v1"},
        ),
        (
            "analysis/result.npz",
            "application/x-npz",
            {"schema": "fdm.table.v1", "allow_pickle": True},
        ),
        (
            "analysis/result.npz",
            "application/x-npz",
            {"schema": "fdm.table.v1", "dtype": "object"},
        ),
        (
            "analysis/result.npz",
            "application/x-npz",
            {},
        ),
    ],
)
def test_asset_reference_rejects_pickle_or_missing_safe_schema(
    path: str,
    media_type: str,
    metadata: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        AnalysisAssetReference(
            kind="table",
            path=path,
            sha256="d" * 64,
            media_type=media_type,
            metadata=metadata,
        )


def test_large_inline_arrays_must_be_stored_as_assets() -> None:
    with pytest.raises(ValueError, match="安全资产"):
        AnalysisTable(
            name="过大表格",
            columns=("值",),
            rows=tuple((index,) for index in range(100_001)),
        )
    with pytest.raises(ValueError, match="安全资产"):
        AnalysisCurve(
            name="过大曲线",
            x=tuple(float(index) for index in range(100_001)),
            y=tuple(float(index) for index in range(100_001)),
        )


def test_current_and_stale_status_contract() -> None:
    with pytest.raises(ValueError, match="current"):
        _artifact(stale_reason="不应出现")
    with pytest.raises(ValueError, match="stale"):
        _artifact(status="stale", stale_reason=None)

    stale = _artifact().mark_stale("来源已变化")

    assert stale.status is AnalysisArtifactStatus.STALE
    assert stale.stale_reason == "来源已变化"
    assert not stale.is_current
    assert AnalysisArtifact.from_dict(stale.to_dict()) == stale


def test_pixel_revision_change_marks_artifact_stale() -> None:
    artifact = _artifact()

    refreshed = refresh_artifact_validity(
        artifact,
        current_pixel_revision=4,
        current_calibration_signature=artifact.calibration_signature,
        roi_revisions={"roi_1": 7},
    )

    assert refreshed.stale_reason == "来源图片像素已变化"


def test_omitted_calibration_check_does_not_mean_uncalibrated() -> None:
    artifact = _artifact(source_reference=None)

    refreshed = refresh_artifact_validity(
        artifact,
        current_pixel_revision=3,
    )

    assert refreshed is artifact


def test_calibration_change_marks_artifact_stale() -> None:
    artifact = _artifact()

    refreshed = refresh_artifact_validity(
        artifact,
        current_pixel_revision=3,
        current_calibration_signature="sha256:" + ("f" * 64),
        roi_revisions={"roi_1": 7},
    )

    assert refreshed.stale_reason == "标定已变化"


@pytest.mark.parametrize(
    ("revisions", "expected"),
    [
        ({}, "引用的ROI已不存在"),
        ({"roi_1": 8}, "引用的ROI几何已变化"),
    ],
)
def test_roi_change_or_removal_marks_artifact_stale(revisions, expected) -> None:
    artifact = _artifact()

    refreshed = refresh_artifact_validity(
        artifact,
        current_pixel_revision=3,
        current_calibration_signature=artifact.calibration_signature,
        roi_revisions=revisions,
    )

    assert refreshed.stale_reason == expected


def test_measurement_reference_uses_measurement_revision_registry() -> None:
    artifact = _artifact(
        source_reference=AnalysisObjectReference(
            AnalysisObjectKind.MEASUREMENT,
            "measurement_1",
            2,
        ),
    )

    current = refresh_artifact_validity(
        artifact,
        current_pixel_revision=3,
        current_calibration_signature=artifact.calibration_signature,
        measurement_revisions={"measurement_1": 2},
    )
    stale = refresh_artifact_validity(
        artifact,
        current_pixel_revision=3,
        current_calibration_signature=artifact.calibration_signature,
        measurement_revisions={"measurement_1": 3},
    )

    assert current is artifact
    assert stale.stale_reason == "引用的测量对象几何已变化"


def test_stale_artifact_never_auto_revives() -> None:
    stale = _artifact(status="stale", stale_reason="历史结果")

    refreshed = refresh_artifact_validity(
        stale,
        current_pixel_revision=3,
        current_calibration_signature=stale.calibration_signature,
        roi_revisions={"roi_1": 7},
    )

    assert refreshed is stale
    assert refreshed.stale_reason == "历史结果"


def test_batch_invalidation_preserves_order_and_other_documents() -> None:
    first = _artifact(id="first")
    other = _artifact(id="other", source_document_id="image_2")

    refreshed = refresh_artifacts_validity(
        (first, other),
        document_id="image_1",
        current_pixel_revision=4,
        current_calibration_signature=first.calibration_signature,
        roi_revisions={"roi_1": 7},
    )

    assert [item.id for item in refreshed] == ["first", "other"]
    assert refreshed[0].status is AnalysisArtifactStatus.STALE
    assert refreshed[1] is other


def test_document_removal_marks_only_its_artifacts_stale() -> None:
    first = _artifact(id="first")
    other = _artifact(id="other", source_document_id="image_2")

    refreshed = refresh_artifacts_validity(
        (first, other),
        document_id="image_1",
        source_document_exists=False,
        current_calibration_signature=first.calibration_signature,
    )

    assert refreshed[0].stale_reason == "来源文档已不存在"
    assert refreshed[1] is other


def test_calibration_signature_is_stable_and_sensitive() -> None:
    first = calibration_signature_from_values(
        pixels_per_unit=7.5,
        unit="um",
    )
    same = calibration_signature_from_values(
        pixels_per_unit=7.5,
        unit="um",
    )
    different_scale = calibration_signature_from_values(
        pixels_per_unit=7.6,
        unit="um",
    )
    different_unit = calibration_signature_from_values(
        pixels_per_unit=7.5,
        unit="mm",
    )

    assert first == same
    assert first != different_scale
    assert first != different_unit
    assert calibration_signature_from_values(
        pixels_per_unit=None,
        unit="px",
    ) is None


def test_from_dict_rejects_unknown_fields_and_invalid_nested_schema() -> None:
    payload = _artifact().to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="未知"):
        AnalysisArtifact.from_dict(payload)

    payload = _artifact().to_dict()
    payload["tables"][0]["rows"][0].append("extra")
    with pytest.raises(ValueError, match="列数"):
        AnalysisArtifact.from_dict(payload)

    payload = _artifact().to_dict()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        AnalysisArtifact.from_dict(payload)
