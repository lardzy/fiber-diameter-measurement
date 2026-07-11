from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
import statistics

from fdm.models import ImageDocument, Measurement, ProjectState, UNCATEGORIZED_LABEL, square_unit


class StatisticsScope(str, Enum):
    """The document selection used to build a statistics snapshot."""

    CURRENT_CATEGORY = "current_category"
    CURRENT_DOCUMENT = "current_document"
    PROJECT = "project"


class MeasurementMetric(str, Enum):
    """Measurement families that must never be mixed in one snapshot."""

    LENGTH = "length"
    AREA = "area"
    COUNT = "count"


@dataclass(frozen=True, slots=True)
class MeasurementStatisticsSnapshot:
    """An immutable, single-unit summary for one metric and scope.

    Quality counters are intentionally independent.  For example, a hard-failure
    measurement whose stored value is NaN contributes to both
    ``hard_failure_count`` and ``non_finite_count``, but never to the numeric
    statistics.  Percentiles use linear interpolation (NumPy's default/type-7
    convention); whiskers are the most extreme inlier values within 1.5×IQR.
    """

    scope: StatisticsScope
    metric: MeasurementMetric
    unit: str
    total_count: int
    valid_count: int
    hard_failure_count: int
    manual_review_count: int
    non_finite_count: int
    missing_value_count: int
    mean: float | None
    median: float | None
    stddev: float | None
    cv_percent: float | None
    minimum: float | None
    maximum: float | None
    q1: float | None
    q3: float | None
    p10: float | None
    p90: float | None
    total_value: float | None
    histogram_edges: tuple[float, ...]
    histogram_counts: tuple[int, ...]
    lower_whisker: float | None
    upper_whisker: float | None
    outlier_measurement_ids: tuple[str, ...]
    outlier_values: tuple[float, ...]

    @property
    def excluded_count(self) -> int:
        """Number of records excluded from numeric statistics.

        Each record is counted once here even if it has more than one quality
        issue; the individual quality counters above may overlap.
        """

        return self.total_count - self.valid_count


@dataclass(slots=True)
class _UnitBucket:
    total_count: int = 0
    hard_failure_count: int = 0
    manual_review_count: int = 0
    non_finite_count: int = 0
    missing_value_count: int = 0
    values: list[tuple[str, float]] = field(default_factory=list)


class MeasurementStatisticsService:
    """Build descriptive statistics without mutating project measurements.

    ``summarize`` returns one snapshot per unit in first-measurement order.  This
    prevents a project containing, for example, both ``um`` and ``mm`` from
    being silently combined.  Documents without the requested metric do not
    create phantom unit groups.  A completely empty selection still returns one
    zero-count snapshot so consumers never need to invent a unit themselves.
    """

    DEFAULT_HARD_FAILURE_STATUSES = frozenset(
        {
            "line_too_short",
            "profile_too_flat",
            "edge_pair_not_found",
            "component_not_found",
            "centerline_not_found",
            "boundary_not_found",
        }
    )
    MANUAL_REVIEW_STATUS = "manual_review"
    MAX_HISTOGRAM_BINS = 32

    def __init__(self, *, hard_failure_statuses: Iterable[str] | None = None) -> None:
        statuses = (
            self.DEFAULT_HARD_FAILURE_STATUSES
            if hard_failure_statuses is None
            else hard_failure_statuses
        )
        self._hard_failure_statuses = frozenset(
            str(status).strip().casefold()
            for status in statuses
            if str(status).strip()
        )

    def summarize(
        self,
        project: ProjectState,
        *,
        metric: MeasurementMetric,
        scope: StatisticsScope,
        document_id: str | None = None,
        fiber_group_id: str | None = None,
    ) -> tuple[MeasurementStatisticsSnapshot, ...]:
        """Summarize a project selection, grouped by physical unit.

        ``document_id`` is required for current-document and current-category
        scopes.  For the category scope, ``fiber_group_id=None`` explicitly
        selects uncategorized measurements.
        """

        metric = MeasurementMetric(metric)
        scope = StatisticsScope(scope)
        documents = self._select_documents(project, scope=scope, document_id=document_id)
        return self.summarize_documents(
            documents,
            metric=metric,
            scope=scope,
            fiber_group_id=fiber_group_id,
        )

    def summarize_documents(
        self,
        documents: Iterable[ImageDocument],
        *,
        metric: MeasurementMetric,
        scope: StatisticsScope = StatisticsScope.PROJECT,
        fiber_group_id: str | None = None,
    ) -> tuple[MeasurementStatisticsSnapshot, ...]:
        """Low-level variant useful to UI previews and export adapters."""

        metric = MeasurementMetric(metric)
        scope = StatisticsScope(scope)
        selected_documents = tuple(documents)
        buckets: dict[str, _UnitBucket] = {}

        for document in selected_documents:
            unit = self._unit_for(document, metric)
            for measurement in document.measurements:
                if not self._matches_metric(measurement, metric):
                    continue
                if (
                    scope is StatisticsScope.CURRENT_CATEGORY
                    and measurement.fiber_group_id != fiber_group_id
                ):
                    continue
                bucket = buckets.setdefault(unit, _UnitBucket())
                self._collect(bucket, document, measurement, metric)

        if not buckets:
            # A single selected document has an unambiguous display unit even
            # before it contains this metric.  Multiple empty documents can use
            # their shared unit, but mixed units have no result to prefer.
            represented_units = tuple(
                dict.fromkeys(self._unit_for(document, metric) for document in selected_documents)
            )
            empty_unit = (
                represented_units[0]
                if len(represented_units) == 1
                else self._default_unit(metric)
            )
            buckets[empty_unit] = _UnitBucket()

        return tuple(
            self._snapshot(scope=scope, metric=metric, unit=unit, bucket=bucket)
            for unit, bucket in buckets.items()
        )

    def summarize_by_category(
        self,
        documents: Iterable[ImageDocument],
        *,
        metric: MeasurementMetric,
        scope: StatisticsScope = StatisticsScope.PROJECT,
    ) -> tuple[tuple[str, tuple[MeasurementStatisticsSnapshot, ...]], ...]:
        """Summarize matching measurements by display category and unit.

        Same-named categories from different project documents are combined;
        their document-local group IDs are intentionally not treated as a
        project-wide identity.
        """

        metric = MeasurementMetric(metric)
        scope = StatisticsScope(scope)
        categories: dict[str, tuple[str, dict[str, _UnitBucket]]] = {}
        for document in tuple(documents):
            unit = self._unit_for(document, metric)
            for measurement in document.measurements:
                if not self._matches_metric(measurement, metric):
                    continue
                group = document.get_group(measurement.fiber_group_id)
                if group is None:
                    label = UNCATEGORIZED_LABEL
                else:
                    # Group IDs and sequence numbers are document-local.  A
                    # project comparison therefore uses the stable category
                    # label when one exists, while still giving unnamed groups
                    # an intelligible fallback.
                    label = group.label.strip() or group.display_name()
                key = label.strip().casefold()
                display_label, buckets = categories.setdefault(key, (label, {}))
                bucket = buckets.setdefault(unit, _UnitBucket())
                self._collect(bucket, document, measurement, metric)

        return tuple(
            (
                display_label,
                tuple(
                    self._snapshot(
                        scope=scope,
                        metric=metric,
                        unit=unit,
                        bucket=bucket,
                    )
                    for unit, bucket in buckets.items()
                ),
            )
            for display_label, buckets in categories.values()
        )

    @staticmethod
    def _select_documents(
        project: ProjectState,
        *,
        scope: StatisticsScope,
        document_id: str | None,
    ) -> tuple[ImageDocument, ...]:
        if scope is StatisticsScope.PROJECT:
            return tuple(project.documents)
        if document_id is None:
            raise ValueError(f"{scope.value} scope requires document_id")
        document = project.get_document(document_id)
        if document is None:
            raise KeyError(f"Unknown document_id: {document_id}")
        return (document,)

    @staticmethod
    def _matches_metric(measurement: Measurement, metric: MeasurementMetric) -> bool:
        if metric is MeasurementMetric.LENGTH:
            return measurement.measurement_kind in {"line", "polyline"}
        if metric is MeasurementMetric.AREA:
            return measurement.measurement_kind == "area"
        return measurement.measurement_kind == "count"

    @staticmethod
    def _default_unit(metric: MeasurementMetric) -> str:
        if metric is MeasurementMetric.AREA:
            return square_unit("px")
        if metric is MeasurementMetric.COUNT:
            return "个"
        return "px"

    @classmethod
    def _unit_for(cls, document: ImageDocument, metric: MeasurementMetric) -> str:
        if metric is MeasurementMetric.COUNT:
            return "个"
        base_unit = document.calibration.unit if document.calibration is not None else "px"
        return square_unit(base_unit) if metric is MeasurementMetric.AREA else base_unit

    @classmethod
    def display_unit_for(cls, document: ImageDocument, metric: MeasurementMetric) -> str:
        """Return the unit label used for a document/metric statistics bucket."""

        return cls._unit_for(document, MeasurementMetric(metric))

    @staticmethod
    def _raw_value(
        document: ImageDocument,
        measurement: Measurement,
        metric: MeasurementMetric,
    ) -> object:
        if metric is MeasurementMetric.COUNT:
            return 1.0
        if metric is MeasurementMetric.AREA:
            return measurement.area_unit if document.calibration is not None else measurement.area_px
        return measurement.diameter_unit if document.calibration is not None else measurement.diameter_px

    def _collect(
        self,
        bucket: _UnitBucket,
        document: ImageDocument,
        measurement: Measurement,
        metric: MeasurementMetric,
    ) -> None:
        bucket.total_count += 1
        status = str(measurement.status or "").strip().casefold()
        is_hard_failure = status in self._hard_failure_statuses
        if is_hard_failure:
            bucket.hard_failure_count += 1
        if status == self.MANUAL_REVIEW_STATUS:
            bucket.manual_review_count += 1

        raw_value = self._raw_value(document, measurement, metric)
        if raw_value is None:
            bucket.missing_value_count += 1
            return
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            bucket.non_finite_count += 1
            return
        if not math.isfinite(value):
            bucket.non_finite_count += 1
            return
        if is_hard_failure:
            return
        bucket.values.append((measurement.id, value))

    def _snapshot(
        self,
        *,
        scope: StatisticsScope,
        metric: MeasurementMetric,
        unit: str,
        bucket: _UnitBucket,
    ) -> MeasurementStatisticsSnapshot:
        entries = bucket.values
        values = [value for _, value in entries]
        if not values:
            return MeasurementStatisticsSnapshot(
                scope=scope,
                metric=metric,
                unit=unit,
                total_count=bucket.total_count,
                valid_count=0,
                hard_failure_count=bucket.hard_failure_count,
                manual_review_count=bucket.manual_review_count,
                non_finite_count=bucket.non_finite_count,
                missing_value_count=bucket.missing_value_count,
                mean=None,
                median=None,
                stddev=None,
                cv_percent=None,
                minimum=None,
                maximum=None,
                q1=None,
                q3=None,
                p10=None,
                p90=None,
                total_value=None,
                histogram_edges=(),
                histogram_counts=(),
                lower_whisker=None,
                upper_whisker=None,
                outlier_measurement_ids=(),
                outlier_values=(),
            )

        sorted_values = sorted(values)
        q1 = self._percentile(sorted_values, 0.25)
        q3 = self._percentile(sorted_values, 0.75)
        if metric is MeasurementMetric.COUNT:
            # A count marker represents one observed object, not a sample whose
            # measured value happens to be 1.  Descriptive statistics and a
            # distribution of constant ones would therefore be misleading.
            return MeasurementStatisticsSnapshot(
                scope=scope,
                metric=metric,
                unit=unit,
                total_count=bucket.total_count,
                valid_count=len(values),
                hard_failure_count=bucket.hard_failure_count,
                manual_review_count=bucket.manual_review_count,
                non_finite_count=bucket.non_finite_count,
                missing_value_count=bucket.missing_value_count,
                mean=None,
                median=None,
                stddev=None,
                cv_percent=None,
                minimum=None,
                maximum=None,
                q1=None,
                q3=None,
                p10=None,
                p90=None,
                total_value=float(len(values)),
                histogram_edges=(),
                histogram_counts=(),
                lower_whisker=None,
                upper_whisker=None,
                outlier_measurement_ids=(),
                outlier_values=(),
            )

        mean = self._stable_mean(values)
        stddev = self._safe_pstdev(values)
        histogram_edges, histogram_counts = self._histogram(sorted_values, q1=q1, q3=q3)
        iqr = q3 - q1
        lower_fence = q1 - (1.5 * iqr)
        upper_fence = q3 + (1.5 * iqr)
        inlier_values = [
            value
            for value in sorted_values
            if lower_fence <= value <= upper_fence
        ]
        outlier_entries = tuple(
            (measurement_id, value)
            for measurement_id, value in entries
            if value < lower_fence or value > upper_fence
        )
        cv_percent = None
        if mean is not None and mean > 0.0 and stddev is not None:
            candidate_cv = stddev / mean * 100.0
            if math.isfinite(candidate_cv):
                cv_percent = candidate_cv
        return MeasurementStatisticsSnapshot(
            scope=scope,
            metric=metric,
            unit=unit,
            total_count=bucket.total_count,
            valid_count=len(values),
            hard_failure_count=bucket.hard_failure_count,
            manual_review_count=bucket.manual_review_count,
            non_finite_count=bucket.non_finite_count,
            missing_value_count=bucket.missing_value_count,
            mean=mean,
            median=self._percentile(sorted_values, 0.50),
            stddev=stddev,
            cv_percent=cv_percent,
            minimum=sorted_values[0],
            maximum=sorted_values[-1],
            q1=q1,
            q3=q3,
            p10=self._percentile(sorted_values, 0.10),
            p90=self._percentile(sorted_values, 0.90),
            total_value=self._safe_fsum(values),
            histogram_edges=histogram_edges,
            histogram_counts=histogram_counts,
            lower_whisker=inlier_values[0] if inlier_values else None,
            upper_whisker=inlier_values[-1] if inlier_values else None,
            outlier_measurement_ids=tuple(item[0] for item in outlier_entries),
            outlier_values=tuple(item[1] for item in outlier_entries),
        )

    @staticmethod
    def _stable_mean(values: Sequence[float]) -> float | None:
        # Dividing before summing keeps the intermediate magnitude within the
        # finite input range, unlike statistics.fmean([1e308, 1e308]).
        try:
            result = math.fsum(value / len(values) for value in values)
        except (OverflowError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _safe_fsum(values: Sequence[float]) -> float | None:
        try:
            result = math.fsum(values)
        except (OverflowError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _safe_pstdev(values: Sequence[float]) -> float | None:
        if len(values) == 1:
            return 0.0
        try:
            result = statistics.pstdev(values)
        except (OverflowError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
        """Linear percentile equivalent to NumPy's default method."""

        if not sorted_values:
            raise ValueError("percentile requires at least one value")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be between 0 and 1")
        position = (len(sorted_values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(sorted_values[lower])
        weight = position - lower
        return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)

    def _histogram(
        self,
        sorted_values: Sequence[float],
        *,
        q1: float,
        q3: float,
    ) -> tuple[tuple[float, ...], tuple[int, ...]]:
        minimum = sorted_values[0]
        maximum = sorted_values[-1]
        if minimum == maximum:
            half_width = max(abs(minimum) * 0.01, 0.5)
            lower_edge = minimum - half_width
            upper_edge = maximum + half_width
            if not math.isfinite(lower_edge):
                lower_edge = minimum
            if not math.isfinite(upper_edge):
                upper_edge = maximum
            return (lower_edge, upper_edge), (len(sorted_values),)

        value_range = maximum - minimum
        iqr = q3 - q1
        if not math.isfinite(value_range) or not math.isfinite(iqr):
            return (), ()
        bin_width = 2.0 * iqr * (len(sorted_values) ** (-1.0 / 3.0)) if iqr > 0.0 else 0.0
        if bin_width > 0.0:
            bin_count = math.ceil(value_range / bin_width)
        else:
            bin_count = math.ceil(math.log2(len(sorted_values)) + 1.0)
        bin_count = max(1, min(self.MAX_HISTOGRAM_BINS, bin_count))
        actual_width = value_range / bin_count
        edges = tuple(
            minimum + (actual_width * index)
            for index in range(bin_count)
        ) + (maximum,)
        counts = [0] * bin_count
        for value in sorted_values:
            index = min(bin_count - 1, int((value - minimum) / actual_width))
            counts[index] += 1
        return edges, tuple(counts)
