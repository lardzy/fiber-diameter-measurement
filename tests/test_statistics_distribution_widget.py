from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QScrollArea

from fdm.models import Calibration, ImageDocument, Measurement, ProjectState
from fdm.services.measurement_statistics import MeasurementMetric, StatisticsScope
from fdm.ui.statistics_distribution import (
    DistributionRecordFilterRequest,
    StatisticsDistributionWidget,
    _DistributionTaskResult,
)
from fdm.ui.statistics_widgets import MeasurementStatisticsPanel


def _document(
    document_id: str,
    *,
    unit: str | None = None,
    category_values: tuple[tuple[str, str, tuple[float, ...]], ...] = (),
    count_categories: tuple[tuple[str, str, int], ...] = (),
) -> ImageDocument:
    calibration = (
        Calibration(
            mode="preset",
            pixels_per_unit=2.0,
            unit=unit,
            source_label="test",
        )
        if unit is not None
        else None
    )
    document = ImageDocument(
        id=document_id,
        path=f"/{document_id}.png",
        image_size=(100, 80),
        calibration=calibration,
    )
    for label, color, values in category_values:
        group = document.create_group(color=color, label=label)
        for index, value in enumerate(values):
            document.measurements.append(
                Measurement(
                    id=f"{document_id}_{label}_length_{index}",
                    image_id=document_id,
                    fiber_group_id=group.id,
                    mode="manual",
                    measurement_kind="line",
                    diameter_px=value,
                    diameter_unit=value,
                    status="manual",
                )
            )
    for label, color, count in count_categories:
        group = document.find_group_by_label(label) or document.create_group(
            color=color,
            label=label,
        )
        for index in range(count):
            document.measurements.append(
                Measurement(
                    id=f"{document_id}_{label}_count_{index}",
                    image_id=document_id,
                    fiber_group_id=group.id,
                    mode="count",
                    measurement_kind="count",
                    status="manual",
                )
            )
    document.rebuild_group_memberships()
    return document


class StatisticsDistributionWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _widget(self) -> StatisticsDistributionWidget:
        widget = StatisticsDistributionWidget()
        self.addCleanup(widget.close)
        return widget

    @staticmethod
    def _refresh_now(widget: StatisticsDistributionWidget) -> None:
        widget._refresh_timer.stop()
        widget._refresh_now()

    def test_project_context_separates_units_and_category_color_conflicts(self) -> None:
        first = _document(
            "first",
            category_values=(("棉", "#2A9D8F", (10.0, 20.0)),),
        )
        second = _document(
            "second",
            unit="um",
            category_values=(("棉", "#D79B45", (30.0,)), ("麻", "#6A8EDB", (40.0,))),
        )
        widget = self._widget()
        widget.set_context(
            ProjectState(version="test", documents=[first, second]),
            first,
            suggested_metric=MeasurementMetric.LENGTH,
        )
        widget.scope_combo.setCurrentIndex(
            widget.scope_combo.findData(StatisticsScope.PROJECT)
        )
        self._refresh_now(widget)

        self.assertEqual(
            [widget.unit_combo.itemData(index) for index in range(widget.unit_combo.count())],
            [None, "px", "um"],
        )
        self.assertTrue(widget.unit_combo.isEnabled())
        self.assertIsNone(widget.unit_combo.currentData())
        self.assertIn("请选择单位", widget.context_label.text())
        self.assertEqual(
            [label for label, _snapshots in widget.category_comparisons],
            ["棉", "麻"],
        )
        self.assertIn("棉", widget._color_conflicts)
        self.assertIn("多种单位", widget.context_label.text())
        self.assertIn("稳定配色", widget.context_label.text())

        unit_index = widget.unit_combo.findData("um")
        widget.unit_combo.setCurrentIndex(unit_index)
        widget.unit_combo.activated.emit(unit_index)
        self.assertEqual(widget.unit_combo.currentData(), "um")
        self.assertIn("· um ·", widget.context_label.text())

    def test_chart_selection_is_separate_from_explicit_record_filter(self) -> None:
        document = _document(
            "image",
            category_values=(
                ("棉", "#2A9D8F", (10.0, 20.0)),
                ("麻", "#D79B45", (30.0,)),
            ),
        )
        widget = self._widget()
        widget.set_context(
            ProjectState(version="test", documents=[document]),
            document,
            suggested_metric=MeasurementMetric.LENGTH,
        )
        self._refresh_now(widget)
        requests: list[DistributionRecordFilterRequest] = []
        widget.recordFilterRequested.connect(requests.append)

        widget._select_category_from_chart("棉")

        self.assertEqual(widget.target_combo.currentData(), "category")
        self.assertEqual(widget.category_combo.currentData(), "棉")
        self.assertEqual(requests, [])
        self.assertTrue(widget.filter_records_button.isEnabled())

        widget.filter_records_button.click()

        self.assertEqual(
            requests,
            [
                DistributionRecordFilterRequest(
                    document_id=document.id,
                    category_label="棉",
                    metric=MeasurementMetric.LENGTH,
                )
            ],
        )

    def test_count_uses_category_counts_without_fake_continuous_plots(self) -> None:
        document = _document(
            "counts",
            count_categories=(("棉", "#2A9D8F", 3), ("麻", "#D79B45", 2)),
        )
        widget = self._widget()
        widget.set_context(
            ProjectState(version="test", documents=[document]),
            document,
            suggested_metric=MeasurementMetric.COUNT,
        )
        self._refresh_now(widget)

        self.assertEqual(widget.active_metric(), MeasurementMetric.COUNT)
        self.assertFalse(widget.bar_metric_combo.isEnabled())
        self.assertEqual(widget.bar_metric_combo.currentData(), "valid_count")
        self.assertEqual(widget.histogram_canvas.snapshot.metric, MeasurementMetric.COUNT)
        self.assertEqual(
            [(item.label, item.value) for item in widget._category_series("valid_count")],
            [("棉", 3.0), ("麻", 2.0)],
        )
        self.assertIn("不绘制直方图和箱线图", widget.context_label.text())
        widget.resize(900, 520)
        widget.show()
        self.app.processEvents()
        self.assertFalse(widget.grab().isNull())

    def test_donut_legend_visibility_keeps_the_full_denominator(self) -> None:
        document = _document(
            "legend",
            category_values=(
                ("棉", "#2A9D8F", (10.0, 20.0)),
                ("麻", "#D79B45", (30.0,)),
            ),
        )
        widget = self._widget()
        widget.set_context(
            ProjectState(version="test", documents=[document]),
            document,
            suggested_metric=MeasurementMetric.LENGTH,
        )
        self._refresh_now(widget)
        self.assertEqual(widget.donut_canvas.denominator_total, 3.0)

        widget._toggle_category_visibility("棉")

        self.assertIn("棉", widget._hidden_categories)
        self.assertEqual(widget.donut_canvas.denominator_total, 3.0)

    def test_donut_legend_pages_make_later_categories_reachable(self) -> None:
        categories = tuple(
            (f"类别{index}", f"#{index + 1:02X}7799", (float(index + 1),))
            for index in range(8)
        )
        document = _document("many-categories", category_values=categories)
        widget = self._widget()
        widget.set_context(
            ProjectState(version="test", documents=[document]),
            document,
            suggested_metric=MeasurementMetric.LENGTH,
        )
        self._refresh_now(widget)
        self.assertEqual(
            widget.donut_canvas.visible_legend_labels(),
            tuple(f"类别{index}" for index in range(5)),
        )

        widget.donut_canvas._legend_offset = 3

        self.assertEqual(
            widget.donut_canvas.visible_legend_labels(),
            tuple(f"类别{index}" for index in range(3, 8)),
        )
        self.assertIn("N=1", widget.donut_canvas._category_value_text("类别0"))
        self.assertIn("占", widget.donut_canvas._category_value_text("类别0"))

    def test_responsive_breakpoints_match_dashboard_contract(self) -> None:
        self.assertEqual(StatisticsDistributionWidget.chart_columns_for_width(1200), 4)
        self.assertEqual(StatisticsDistributionWidget.chart_columns_for_width(1199), 2)
        self.assertEqual(StatisticsDistributionWidget.chart_columns_for_width(640), 2)
        self.assertEqual(StatisticsDistributionWidget.chart_columns_for_width(639), 1)
        self.assertEqual(StatisticsDistributionWidget.control_columns_for_width(1200), 7)
        self.assertEqual(StatisticsDistributionWidget.control_columns_for_width(900), 4)
        self.assertEqual(StatisticsDistributionWidget.control_columns_for_width(500), 2)

    def test_compact_dashboard_uses_one_scroll_page_and_reaches_final_card(self) -> None:
        widget = self._widget()
        widget.resize(600, 160)
        widget.show()
        self.app.processEvents()

        self.assertEqual(widget.findChildren(QScrollArea), [widget._scroll])
        for child in (
            widget.metric_combo,
            widget.context_label,
            widget.histogram_card,
            widget.bar_card,
        ):
            self.assertTrue(widget._scroll_content.isAncestorOf(child))
        self.assertEqual(widget._chart_columns, 1)
        self.assertEqual(widget._control_columns, 2)
        self.assertEqual(widget.bar_card.width(), widget._cards_container.width())

        scroll_bar = widget._scroll.verticalScrollBar()
        self.assertGreater(scroll_bar.maximum(), 0)
        scroll_bar.setValue(scroll_bar.maximum())
        self.app.processEvents()
        viewport = widget._scroll.viewport()
        final_card_top = widget.bar_card.mapTo(viewport, QPoint(0, 0)).y()
        final_card_bottom = widget.bar_card.mapTo(
            viewport,
            QPoint(0, widget.bar_card.height()),
        ).y()
        self.assertLess(final_card_top, viewport.height())
        self.assertLessEqual(final_card_bottom, viewport.height())

    def test_compact_dashboard_combo_wheel_scrolls_page_without_changing_value(self) -> None:
        widget = self._widget()
        widget.resize(600, 160)
        widget.show()
        self.app.processEvents()
        widget.metric_combo.setCurrentIndex(1)
        scroll_bar = widget._scroll.verticalScrollBar()
        self.assertGreater(scroll_bar.maximum(), 0)

        event = QWheelEvent(
            QPointF(5, 5),
            QPointF(5, 5),
            QPoint(0, 0),
            QPoint(0, -120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        QApplication.sendEvent(widget.metric_combo, event)

        self.assertEqual(widget.metric_combo.currentIndex(), 1)
        self.assertGreater(scroll_bar.value(), 0)

    def test_late_background_result_is_ignored_by_generation(self) -> None:
        document = _document(
            "generation",
            category_values=(("棉", "#2A9D8F", (10.0, 20.0)),),
        )
        widget = self._widget()
        widget.set_context(
            ProjectState(version="test", documents=[document]),
            document,
            suggested_metric=MeasurementMetric.LENGTH,
        )
        self._refresh_now(widget)
        completed = widget.snapshots
        stale_generation = widget._generation
        widget._generation += 1

        widget._on_async_ready(
            stale_generation,
            _DistributionTaskResult((), ()),
            None,
        )

        self.assertEqual(widget.snapshots, completed)

    def test_live_statistics_keeps_result_but_no_longer_owns_object_details(self) -> None:
        document = _document(
            "selected",
            category_values=(("棉", "#2A9D8F", (12.0,)),),
        )
        panel = MeasurementStatisticsPanel()
        self.addCleanup(panel.close)
        panel.set_context(
            ProjectState(version="test", documents=[document]),
            document,
            tool_mode="manual",
            selected_measurement=document.measurements[0],
        )

        self.assertIn("当前结果", panel.current_value_label.text())
        self.assertFalse(hasattr(panel, "_object_details_toggle"))
        self.assertFalse(hasattr(panel, "_object_details_label"))


if __name__ == "__main__":
    unittest.main()
