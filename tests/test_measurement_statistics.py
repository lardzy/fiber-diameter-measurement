from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.models import Calibration, ImageDocument, Measurement, ProjectState
from fdm.services.measurement_statistics import (
    MeasurementMetric,
    MeasurementStatisticsService,
    StatisticsScope,
)


def _measurement(
    measurement_id: str,
    *,
    image_id: str,
    kind: str = "line",
    value: float | None = None,
    group_id: str | None = None,
    status: str = "ready",
) -> Measurement:
    kwargs: dict[str, object] = {}
    if kind in {"line", "polyline"}:
        kwargs["diameter_px"] = value
        kwargs["diameter_unit"] = value
    elif kind == "area":
        kwargs["area_px"] = value
        kwargs["area_unit"] = value
    return Measurement(
        id=measurement_id,
        image_id=image_id,
        fiber_group_id=group_id,
        mode="test",
        measurement_kind=kind,
        status=status,
        **kwargs,
    )


def _document(
    document_id: str,
    measurements: list[Measurement],
    *,
    unit: str | None = None,
) -> ImageDocument:
    calibration = None
    if unit is not None:
        calibration = Calibration(
            mode="preset",
            pixels_per_unit=2.0,
            unit=unit,
            source_label="test",
        )
    return ImageDocument(
        id=document_id,
        path=f"/{document_id}.png",
        image_size=(100, 80),
        calibration=calibration,
        measurements=measurements,
    )


class MeasurementStatisticsServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = MeasurementStatisticsService()

    def test_measurement_families_are_never_mixed(self) -> None:
        document = _document(
            "image_1",
            [
                _measurement("line", image_id="image_1", value=10.0),
                _measurement("polyline", image_id="image_1", kind="polyline", value=20.0),
                _measurement("area", image_id="image_1", kind="area", value=200.0),
                _measurement("count", image_id="image_1", kind="count"),
            ],
        )
        project = ProjectState(version="test", documents=[document])

        length = self.service.summarize(
            project,
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.CURRENT_DOCUMENT,
            document_id=document.id,
        )[0]
        area = self.service.summarize(
            project,
            metric=MeasurementMetric.AREA,
            scope=StatisticsScope.CURRENT_DOCUMENT,
            document_id=document.id,
        )[0]
        count = self.service.summarize(
            project,
            metric=MeasurementMetric.COUNT,
            scope=StatisticsScope.CURRENT_DOCUMENT,
            document_id=document.id,
        )[0]

        self.assertEqual((length.total_count, length.valid_count, length.mean), (2, 2, 15.0))
        self.assertEqual((area.total_count, area.valid_count, area.mean), (1, 1, 200.0))
        self.assertEqual((count.total_count, count.valid_count, count.total_value), (1, 1, 1.0))
        self.assertIsNone(count.mean)
        self.assertIsNone(count.stddev)
        self.assertIsNone(count.cv_percent)
        self.assertEqual(count.histogram_counts, ())
        self.assertEqual(count.outlier_measurement_ids, ())
        self.assertEqual((length.unit, area.unit, count.unit), ("px", "px²", "个"))

    def test_scope_filters_current_category_document_and_project(self) -> None:
        first = _document(
            "first",
            [
                _measurement("group_a", image_id="first", group_id="a", value=10.0),
                _measurement("group_b", image_id="first", group_id="b", value=20.0),
                _measurement("uncategorized", image_id="first", value=30.0),
            ],
        )
        second = _document(
            "second",
            [_measurement("second_a", image_id="second", group_id="a2", value=40.0)],
        )
        project = ProjectState(version="test", documents=[first, second])

        category = self.service.summarize(
            project,
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.CURRENT_CATEGORY,
            document_id=first.id,
            fiber_group_id="a",
        )[0]
        uncategorized = self.service.summarize(
            project,
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.CURRENT_CATEGORY,
            document_id=first.id,
            fiber_group_id=None,
        )[0]
        current_document = self.service.summarize(
            project,
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.CURRENT_DOCUMENT,
            document_id=first.id,
        )[0]
        whole_project = self.service.summarize(
            project,
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.PROJECT,
        )[0]

        self.assertEqual((category.total_count, category.total_value), (1, 10.0))
        self.assertEqual((uncategorized.total_count, uncategorized.total_value), (1, 30.0))
        self.assertEqual((current_document.total_count, current_document.total_value), (3, 60.0))
        self.assertEqual((whole_project.total_count, whole_project.total_value), (4, 100.0))

    def test_quality_failures_are_counted_but_excluded_from_values(self) -> None:
        document = _document(
            "quality",
            [
                _measurement("valid", image_id="quality", value=10.0),
                _measurement("review", image_id="quality", value=20.0, status="manual_review"),
                _measurement("hard_finite", image_id="quality", value=100.0, status="line_too_short"),
                _measurement(
                    "hard_nan",
                    image_id="quality",
                    value=math.nan,
                    status="edge_pair_not_found",
                ),
                _measurement("nan", image_id="quality", value=math.nan),
                _measurement("infinite", image_id="quality", value=math.inf),
                _measurement("missing", image_id="quality", value=None),
                _measurement(
                    "legacy_hard_failure",
                    image_id="quality",
                    value=200.0,
                    status="component_not_found",
                ),
            ],
        )

        snapshot = self.service.summarize(
            ProjectState(version="test", documents=[document]),
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.PROJECT,
        )[0]

        self.assertEqual(snapshot.total_count, 8)
        self.assertEqual(snapshot.valid_count, 2)
        self.assertEqual(snapshot.excluded_count, 6)
        self.assertEqual(snapshot.hard_failure_count, 3)
        self.assertEqual(snapshot.manual_review_count, 1)
        self.assertEqual(snapshot.non_finite_count, 3)
        self.assertEqual(snapshot.missing_value_count, 1)
        self.assertEqual(snapshot.mean, 15.0)
        self.assertEqual(snapshot.stddev, 5.0)
        self.assertAlmostEqual(snapshot.cv_percent or 0.0, 100.0 / 3.0)

    def test_percentiles_histogram_and_boxplot_outliers_are_deterministic(self) -> None:
        values = [1.0, 1.0, 1.0, 1.0, 100.0]
        document = _document(
            "distribution",
            [
                _measurement(f"m{index}", image_id="distribution", value=value)
                for index, value in enumerate(values)
            ],
        )

        snapshot = self.service.summarize(
            ProjectState(version="test", documents=[document]),
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.PROJECT,
        )[0]

        self.assertEqual(snapshot.median, 1.0)
        self.assertEqual((snapshot.q1, snapshot.q3), (1.0, 1.0))
        self.assertEqual(snapshot.p10, 1.0)
        self.assertAlmostEqual(snapshot.p90 or 0.0, 60.4)
        self.assertEqual((snapshot.lower_whisker, snapshot.upper_whisker), (1.0, 1.0))
        self.assertEqual(snapshot.outlier_measurement_ids, ("m4",))
        self.assertEqual(snapshot.outlier_values, (100.0,))
        self.assertEqual(sum(snapshot.histogram_counts), snapshot.valid_count)
        self.assertEqual(len(snapshot.histogram_edges), len(snapshot.histogram_counts) + 1)
        self.assertLessEqual(len(snapshot.histogram_counts), 32)

    def test_project_snapshots_are_grouped_by_unit_without_conversion(self) -> None:
        pixel_document = _document(
            "pixels",
            [
                _measurement("px", image_id="pixels", value=10.0),
                _measurement("px_area", image_id="pixels", kind="area", value=100.0),
            ],
        )
        micrometer_document = _document(
            "micrometers",
            [
                _measurement("um", image_id="micrometers", value=5.0),
                _measurement("um_area", image_id="micrometers", kind="area", value=25.0),
            ],
            unit="um",
        )
        millimeter_document = _document(
            "millimeters",
            [
                _measurement("mm", image_id="millimeters", value=2.0),
                _measurement("mm_area", image_id="millimeters", kind="area", value=4.0),
            ],
            unit="mm",
        )

        snapshots = self.service.summarize(
            ProjectState(
                version="test",
                documents=[pixel_document, micrometer_document, millimeter_document],
            ),
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.PROJECT,
        )

        self.assertEqual([snapshot.unit for snapshot in snapshots], ["px", "um", "mm"])
        self.assertEqual([snapshot.total_value for snapshot in snapshots], [10.0, 5.0, 2.0])
        self.assertTrue(all(snapshot.valid_count == 1 for snapshot in snapshots))

        area_snapshots = self.service.summarize(
            ProjectState(
                version="test",
                documents=[pixel_document, micrometer_document, millimeter_document],
            ),
            metric=MeasurementMetric.AREA,
            scope=StatisticsScope.PROJECT,
        )
        self.assertEqual([snapshot.unit for snapshot in area_snapshots], ["px²", "um²", "mm²"])

    def test_category_comparison_combines_same_label_across_document_group_ids(self) -> None:
        first = _document(
            "first",
            [_measurement("a", image_id="first", value=10.0)],
        )
        first_group = first.create_group(color="#2A9D8F", label="棉")
        first.measurements[0].fiber_group_id = first_group.id
        first.rebuild_group_memberships()
        second = _document(
            "second",
            [_measurement("b", image_id="second", value=30.0)],
        )
        second_group = second.create_group(color="#D79B45", label="棉")
        second.measurements[0].fiber_group_id = second_group.id
        second.rebuild_group_memberships()

        comparisons = self.service.summarize_by_category(
            [first, second],
            metric=MeasurementMetric.LENGTH,
        )

        self.assertEqual(len(comparisons), 1)
        label, snapshots = comparisons[0]
        self.assertEqual(label, "棉")
        self.assertEqual(snapshots[0].valid_count, 2)
        self.assertEqual(snapshots[0].mean, 20.0)

    def test_project_does_not_report_units_from_documents_without_the_metric(self) -> None:
        measured = _document(
            "measured",
            [_measurement("length", image_id="measured", value=10.0)],
        )
        empty_micrometer = _document("empty_um", [], unit="um")

        snapshots = self.service.summarize(
            ProjectState(version="test", documents=[measured, empty_micrometer]),
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.PROJECT,
        )

        self.assertEqual([snapshot.unit for snapshot in snapshots], ["px"])

        single_empty = self.service.summarize(
            ProjectState(version="test", documents=[empty_micrometer]),
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.PROJECT,
        )[0]
        self.assertEqual(single_empty.unit, "um")
        self.assertEqual(single_empty.total_count, 0)

        two_empty_same_unit = self.service.summarize(
            ProjectState(
                version="test",
                documents=[empty_micrometer, _document("another_empty_um", [], unit="um")],
            ),
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.PROJECT,
        )[0]
        self.assertEqual(two_empty_same_unit.unit, "um")

    def test_extreme_finite_values_never_emit_non_finite_statistics_or_raise(self) -> None:
        document = _document(
            "extreme",
            [
                _measurement("large_1", image_id="extreme", value=1e308),
                _measurement("large_2", image_id="extreme", value=1e308),
            ],
        )

        snapshot = self.service.summarize(
            ProjectState(version="test", documents=[document]),
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.PROJECT,
        )[0]

        self.assertEqual(snapshot.valid_count, 2)
        self.assertEqual(snapshot.mean, 1e308)
        self.assertEqual(snapshot.median, 1e308)
        self.assertEqual(snapshot.stddev, 0.0)
        self.assertIsNone(snapshot.total_value)
        self.assertTrue(
            all(
                value is None or math.isfinite(value)
                for value in (
                    snapshot.mean,
                    snapshot.median,
                    snapshot.stddev,
                    snapshot.cv_percent,
                    snapshot.minimum,
                    snapshot.maximum,
                    snapshot.q1,
                    snapshot.q3,
                    snapshot.p10,
                    snapshot.p90,
                    snapshot.total_value,
                    snapshot.lower_whisker,
                    snapshot.upper_whisker,
                )
            )
        )

        opposing = _document(
            "opposing",
            [
                _measurement("negative", image_id="opposing", value=-1e308),
                _measurement("positive", image_id="opposing", value=1e308),
            ],
        )
        opposing_snapshot = self.service.summarize(
            ProjectState(version="test", documents=[opposing]),
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.PROJECT,
        )[0]
        self.assertEqual(opposing_snapshot.histogram_edges, ())
        self.assertEqual(opposing_snapshot.histogram_counts, ())
        self.assertEqual(opposing_snapshot.mean, 0.0)

        maximum_finite = sys.float_info.max
        boundary = _document(
            "boundary",
            [
                _measurement("max_1", image_id="boundary", value=maximum_finite),
                _measurement("max_2", image_id="boundary", value=maximum_finite),
            ],
        )
        boundary_snapshot = self.service.summarize(
            ProjectState(version="test", documents=[boundary]),
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.PROJECT,
        )[0]
        self.assertEqual(boundary_snapshot.mean, maximum_finite)
        self.assertTrue(all(math.isfinite(edge) for edge in boundary_snapshot.histogram_edges))
        self.assertLess(boundary_snapshot.histogram_edges[0], boundary_snapshot.histogram_edges[1])

    def test_empty_results_keep_a_stable_unit_and_snapshot_is_immutable(self) -> None:
        snapshot = self.service.summarize(
            ProjectState.empty(),
            metric=MeasurementMetric.LENGTH,
            scope=StatisticsScope.PROJECT,
        )[0]

        self.assertEqual(snapshot.unit, "px")
        self.assertEqual((snapshot.total_count, snapshot.valid_count), (0, 0))
        self.assertIsNone(snapshot.mean)
        self.assertEqual(snapshot.histogram_counts, ())
        with self.assertRaises(FrozenInstanceError):
            snapshot.mean = 1.0  # type: ignore[misc]

    def test_current_scopes_reject_missing_or_unknown_document(self) -> None:
        project = ProjectState.empty()

        with self.assertRaises(ValueError):
            self.service.summarize(
                project,
                metric=MeasurementMetric.LENGTH,
                scope=StatisticsScope.CURRENT_DOCUMENT,
            )
        with self.assertRaises(KeyError):
            self.service.summarize(
                project,
                metric=MeasurementMetric.LENGTH,
                scope=StatisticsScope.CURRENT_CATEGORY,
                document_id="missing",
            )


if __name__ == "__main__":
    unittest.main()
