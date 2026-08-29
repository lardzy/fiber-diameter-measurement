from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from fdm.services.dataset_export import (
    DatasetExportFormat,
    DatasetExportRequest,
    DatasetExportService,
    DatasetInstance,
    DatasetIssueSeverity,
    DatasetSample,
)


def _decode_uncompressed_rle(payload: dict[str, object]) -> np.ndarray:
    height, width = payload["size"]
    values: list[int] = []
    bit = 0
    for count in payload["counts"]:
        values.extend([bit] * int(count))
        bit = 1 - bit
    return np.asarray(values, dtype=bool).reshape((height, width), order="F")


def _sample(sample_id: str = "sample", source_group: str = "source-a") -> DatasetSample:
    return DatasetSample(
        sample_id=sample_id,
        source_group_id=source_group,
        source_name=sample_id,
        image=np.full((24, 32), 127, dtype=np.uint8),
        instances=[
            DatasetInstance(
                instance_id="with-hole",
                source_object_id="m1",
                category_name="Cotton",
                rings_px=[
                    [(2, 2), (20, 2), (20, 20), (2, 20)],
                    [(7, 7), (12, 7), (12, 12), (7, 12)],
                ],
            ),
            DatasetInstance(
                instance_id="overlap",
                source_object_id="m2",
                category_name="Cotton",
                rings_px=[[(16, 8), (28, 8), (28, 18), (16, 18)]],
            ),
        ],
        annotation_complete=True,
    )


def test_exports_all_five_formats_and_preserves_coco_holes_and_label_overlap(
    tmp_path: Path,
) -> None:
    output = tmp_path / "dataset"
    service = DatasetExportService()
    request = DatasetExportRequest(
        output_directory=output,
        samples=[_sample()],
        formats=tuple(DatasetExportFormat),
        category_mapping={"Cotton": "cotton"},
        split_train_validation=False,
    )

    result = service.export(request)

    assert result.sample_count == 1
    assert result.instance_count == 2
    coco = json.loads((output / "coco" / "annotations.json").read_text(encoding="utf-8"))
    first = _decode_uncompressed_rle(coco["annotations"][0]["segmentation"])
    assert first[3, 3]
    assert not first[9, 9]
    assert coco["annotations"][0]["iscrowd"] == 0
    assert (output / "coco" / "images" / "all" / "sample.png").is_file()
    assert (output / "coco" / "annotations" / "instances_all.json").is_file()
    assert (output / "yolo_detection" / "labels" / "all" / "sample.txt").is_file()
    assert (output / "yolo_detection" / "images" / "all" / "sample.png").is_file()
    detection_yaml = (output / "yolo_detection" / "dataset.yaml").read_text(encoding="utf-8")
    assert "path: ." in detection_yaml
    assert "train: images/all" in detection_yaml
    assert "val:" not in detection_yaml
    yolo_segmentation = (output / "yolo_segmentation" / "labels" / "all" / "sample.txt").read_text()
    assert len([line for line in yolo_segmentation.splitlines() if line]) == 1

    semantic = np.asarray(Image.open(output / "semantic_labels" / "labels" / "all" / "sample.png"))
    assert semantic[10, 18] == 65535
    instance = tifffile.imread(output / "instance_labels" / "labels" / "all" / "sample.tif")
    assert instance.dtype == np.uint32
    assert instance[10, 18] == np.uint32(0xFFFFFFFF)
    mapping = json.loads(
        (output / "instance_labels" / "mapping" / "all" / "sample.json").read_text(encoding="utf-8")
    )
    assert len(mapping["instances"]) == 2
    report = json.loads((output / "export_report.json").read_text(encoding="utf-8"))
    assert set(report["formats"]) == {item.value for item in DatasetExportFormat}


def test_preflight_requires_explicit_uncategorized_mapping_and_warns_partial_annotation(
    tmp_path: Path,
) -> None:
    sample = _sample()
    sample.annotation_complete = False
    sample.instances[0].category_name = "未分类"
    issues = DatasetExportService().preflight(
        DatasetExportRequest(
            output_directory=tmp_path / "new",
            samples=[sample],
            category_mapping={"Cotton": "cotton", "未分类": None},
        )
    )

    codes = {issue.code for issue in issues}
    assert "category_excluded" in codes
    assert "annotation_completeness_unconfirmed" in codes
    assert all(issue.severity is not DatasetIssueSeverity.ERROR for issue in issues)


def test_group_split_keeps_samples_from_same_source_together(tmp_path: Path) -> None:
    samples = [
        _sample("a-1", "slide-a"),
        _sample("a-2", "slide-a"),
        _sample("b-1", "slide-b"),
    ]
    output = tmp_path / "grouped"
    DatasetExportService().export(
        DatasetExportRequest(
            output_directory=output,
            samples=samples,
            category_mapping={"Cotton": "cotton"},
        )
    )
    report = json.loads((output / "export_report.json").read_text(encoding="utf-8"))
    split_by_sample = {item["sample_id"]: item["split"] for item in report["samples"]}
    assert split_by_sample["a-1"] == split_by_sample["a-2"]
    assert split_by_sample["a-1"] != split_by_sample["b-1"]


def test_cancellation_never_publishes_partial_dataset(tmp_path: Path) -> None:
    output = tmp_path / "cancelled"
    with pytest.raises(RuntimeError, match="已取消"):
        DatasetExportService().export(
            DatasetExportRequest(
                output_directory=output,
                samples=[_sample()],
                category_mapping={"Cotton": "cotton"},
            ),
            cancellation_requested=lambda: True,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".cancelled.staging-*"))


def test_high_bit_source_is_preserved_as_tiff_by_default(tmp_path: Path) -> None:
    sample = _sample()
    sample.image = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
    output = tmp_path / "high-bit"
    DatasetExportService().export(
        DatasetExportRequest(
            output_directory=output,
            samples=[sample],
            category_mapping={"Cotton": "cotton"},
        )
    )
    restored = tifffile.imread(output / "images" / "all" / "sample.tif")
    assert restored.dtype == np.uint16
    assert np.array_equal(restored, sample.image)


def test_empty_confirmed_sample_can_export_as_negative_only_dataset(tmp_path: Path) -> None:
    sample = DatasetSample(
        sample_id="negative",
        source_group_id="source-negative",
        source_name="negative",
        image=np.zeros((12, 16), dtype=np.uint8),
        annotation_complete=True,
    )
    output = tmp_path / "negative-only"

    result = DatasetExportService().export(
        DatasetExportRequest(
            output_directory=output,
            samples=[sample],
            formats=(
                DatasetExportFormat.COCO_INSTANCE,
                DatasetExportFormat.YOLO_DETECTION,
            ),
            category_mapping={},
            split_train_validation=False,
        )
    )

    assert result.instance_count == 0
    coco = json.loads((output / "coco" / "annotations.json").read_text(encoding="utf-8"))
    assert coco["categories"] == []
    assert coco["annotations"] == []
    assert (output / "yolo_detection" / "labels" / "all" / "negative.txt").read_text() == ""
    assert "names: {}" in (output / "yolo_detection" / "dataset.yaml").read_text(
        encoding="utf-8"
    )


def test_missing_slide_coverage_is_clipped_and_marked_ignore_in_label_maps(
    tmp_path: Path,
) -> None:
    sample = _sample()
    sample.valid_coverage = np.ones(sample.image.shape[:2], dtype=bool)
    sample.valid_coverage[:, 18:] = False
    output = tmp_path / "coverage"

    result = DatasetExportService().export(
        DatasetExportRequest(
            output_directory=output,
            samples=[sample],
            formats=(
                DatasetExportFormat.COCO_INSTANCE,
                DatasetExportFormat.YOLO_SEGMENTATION,
                DatasetExportFormat.SEMANTIC_LABELS,
                DatasetExportFormat.INSTANCE_LABELS,
            ),
            category_mapping={"Cotton": "cotton"},
            split_train_validation=False,
        )
    )

    assert "instance_hits_missing_coverage" in {issue.code for issue in result.issues}
    coco = json.loads((output / "coco" / "annotations.json").read_text(encoding="utf-8"))
    for annotation in coco["annotations"]:
        assert not np.any(_decode_uncompressed_rle(annotation["segmentation"])[:, 18:])
    semantic = np.asarray(Image.open(output / "semantic_labels" / "labels" / "all" / "sample.png"))
    assert np.all(semantic[:, 18:] == 65535)
    instance = tifffile.imread(output / "instance_labels" / "labels" / "all" / "sample.tif")
    assert np.all(instance[:, 18:] == np.uint32(0xFFFFFFFF))
    yolo_lines = (output / "yolo_segmentation" / "labels" / "all" / "sample.txt").read_text()
    for token in yolo_lines.split():
        if token.isdigit():
            continue
        assert 0.0 <= float(token) <= 1.0
