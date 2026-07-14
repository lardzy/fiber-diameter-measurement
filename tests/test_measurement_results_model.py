from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QStyleOptionViewItem, QTableView

from fdm.geometry import Point
from fdm.models import Calibration, FiberGroup, ImageDocument, Measurement
from fdm.ui.measurement_records import MeasurementRecordsController, MeasurementRecordsPane
from fdm.ui.measurement_results_model import (
    GROUP_ID_ROLE,
    MEASUREMENT_ID_ROLE,
    MeasurementGroupDelegate,
    MeasurementResultColumn,
    MeasurementResultsModel,
    MeasurementResultsProxyModel,
)


def _measurement(
    measurement_id: str,
    *,
    kind: str = "line",
    mode: str = "manual",
    status: str = "ready",
    value: float = 10.0,
    group_id: str | None = None,
) -> Measurement:
    return Measurement(
        id=measurement_id,
        image_id="image",
        fiber_group_id=group_id,
        mode=mode,
        measurement_kind=kind,
        diameter_px=value if kind in {"line", "polyline"} else None,
        diameter_unit=value if kind in {"line", "polyline"} else None,
        area_px=value if kind == "area" else None,
        area_unit=value if kind == "area" else None,
        confidence=0.75,
        status=status,
    )


def _document(measurements: list[Measurement]) -> ImageDocument:
    return ImageDocument(
        id="image",
        path="/tmp/results.png",
        image_size=(100, 80),
        fiber_groups=[
            FiberGroup(id="cotton", image_id="image", number=1, color="#1F7A8C", label="棉"),
            FiberGroup(id="flax", image_id="image", number=2, color="#D97706", label="麻"),
        ],
        measurements=measurements,
    )


class MeasurementResultsModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_model_exposes_twelve_columns_roles_and_editable_category(self) -> None:
        document = _document(
            [_measurement("meas_line", group_id="cotton", value=12.5)]
        )
        model = MeasurementResultsModel()
        model.set_document(document)

        headers = [
            model.headerData(column, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
            for column in range(model.columnCount())
        ]
        self.assertEqual(
            headers,
            [
                "纤维结果序号",
                "纤维类别结果序号",
                "纤维类别",
                "类型",
                "结果",
                "单位",
                "孔洞面积",
                "模式",
                "置信度",
                "状态",
                "创建时间",
                "ID",
            ],
        )
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.index(0, MeasurementResultColumn.GROUP).data(), "1 棉")
        self.assertEqual(model.index(0, MeasurementResultColumn.KIND).data(), "线段")
        self.assertEqual(model.index(0, MeasurementResultColumn.RESULT).data(), "12.5000")
        self.assertEqual(model.index(0, MeasurementResultColumn.CONFIDENCE).data(), "手工")
        self.assertEqual(model.index(0, MeasurementResultColumn.ID).data(MEASUREMENT_ID_ROLE), "meas_line")
        self.assertEqual(model.index(0, MeasurementResultColumn.GROUP).data(GROUP_ID_ROLE), "cotton")
        self.assertTrue(model.flags(model.index(0, MeasurementResultColumn.GROUP)) & Qt.ItemFlag.ItemIsEditable)
        self.assertFalse(model.flags(model.index(0, MeasurementResultColumn.KIND)) & Qt.ItemFlag.ItemIsEditable)

    def test_sequences_hole_area_and_created_time_reuse_export_semantics(self) -> None:
        first = _measurement("line_cotton_1", group_id="cotton")
        second = _measurement("line_cotton_2", group_id="cotton")
        area = _measurement("area_cotton", kind="area", group_id="cotton")
        rings = [
            [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)],
            [Point(2, 2), Point(4, 2), Point(4, 4), Point(2, 4)],
        ]
        area.replace_area_geometry(
            polygon_px=rings[0],
            area_rings_px=rings,
            exact_area_px=area.area_px,
            calibration=None,
        )
        area.created_at = "2026-01-02T03:04:05+00:00"
        third = _measurement("line_flax_1", group_id="flax")
        document = _document([first, second, area, third])
        document.calibration = Calibration(
            mode="manual",
            pixels_per_unit=2.0,
            unit="µm",
            source_label="test",
        )
        model = MeasurementResultsModel()
        model.set_document(document)

        self.assertEqual(
            [model.index(row, MeasurementResultColumn.RESULT_SEQUENCE).data() for row in range(4)],
            [1, 2, 1, 3],
        )
        self.assertEqual(
            [model.index(row, MeasurementResultColumn.CATEGORY_SEQUENCE).data() for row in range(4)],
            [1, 2, 1, 1],
        )
        self.assertEqual(model.index(0, MeasurementResultColumn.HOLE_AREA).data(), "—")
        self.assertEqual(model.index(2, MeasurementResultColumn.HOLE_AREA).data(), "1.0000")
        self.assertEqual(
            model.index(2, MeasurementResultColumn.HOLE_AREA).data(Qt.ItemDataRole.ToolTipRole),
            "1 µm²",
        )
        self.assertIn("2026-01-02", model.index(2, MeasurementResultColumn.CREATED_AT).data())
        self.assertEqual(
            model.index(2, MeasurementResultColumn.CREATED_AT).data(Qt.ItemDataRole.ToolTipRole),
            area.created_at,
        )

    def test_incremental_append_emits_one_insert_without_reset(self) -> None:
        first = _measurement("first", value=10.0)
        document = _document([first])
        model = MeasurementResultsModel()
        model.set_document(document)
        inserted: list[tuple[int, int]] = []
        reset_count = [0]
        model.rowsInserted.connect(lambda _parent, first_row, last_row: inserted.append((first_row, last_row)))
        model.modelReset.connect(lambda: reset_count.__setitem__(0, reset_count[0] + 1))

        second = _measurement("second", kind="count", mode="count", value=1.0)
        document.measurements.append(second)

        self.assertTrue(model.append_measurement(document, second))
        self.assertEqual(inserted, [(1, 1)])
        self.assertEqual(reset_count[0], 0)
        self.assertEqual(model.rowCount(), 2)
        self.assertEqual(model.source_row_for_id("second"), 1)

    def test_proxy_combines_text_kind_group_and_mutually_exclusive_status_filters(self) -> None:
        document = _document(
            [
                _measurement("valid_line", group_id="cotton", status="ready"),
                _measurement(
                    "review_polyline",
                    kind="polyline",
                    mode="snap",
                    status="manual_review",
                    group_id="cotton",
                ),
                _measurement(
                    "failed_area",
                    kind="area",
                    mode="magic_segment",
                    status="edge_pair_not_found",
                    group_id="flax",
                ),
                _measurement("valid_count", kind="count", mode="count", status="count"),
            ]
        )
        source = MeasurementResultsModel()
        source.set_document(document)
        proxy = MeasurementResultsProxyModel()
        proxy.setSourceModel(source)

        proxy.set_filters(kind="length")
        self.assertEqual(proxy.rowCount(), 2)
        proxy.set_filters(group="棉")
        self.assertEqual(proxy.rowCount(), 2)
        proxy.set_filters(status="valid")
        self.assertEqual(proxy.rowCount(), 2)
        proxy.set_filters(status="review")
        self.assertEqual(proxy.rowCount(), 1)
        self.assertEqual(proxy.index(0, 0).data(MEASUREMENT_ID_ROLE), "review_polyline")
        proxy.set_filters(status="failed")
        self.assertEqual(proxy.rowCount(), 1)
        self.assertEqual(proxy.index(0, 0).data(MEASUREMENT_ID_ROLE), "failed_area")
        proxy.set_filters(query="VALID_COUNT")
        self.assertEqual(proxy.rowCount(), 1)
        self.assertEqual(proxy.index(0, 0).data(MEASUREMENT_ID_ROLE), "valid_count")

    def test_proxy_numeric_sort_keeps_measurement_id_mapping(self) -> None:
        document = _document(
            [
                _measurement("large", value=100.0),
                _measurement("small", value=2.0),
                _measurement("middle", value=30.0),
            ]
        )
        source = MeasurementResultsModel()
        source.set_document(document)
        proxy = MeasurementResultsProxyModel()
        proxy.setSourceModel(source)

        proxy.sort(MeasurementResultColumn.RESULT, Qt.SortOrder.AscendingOrder)

        self.assertEqual(
            [proxy.index(row, 0).data(MEASUREMENT_ID_ROLE) for row in range(proxy.rowCount())],
            ["small", "middle", "large"],
        )
        source_index = proxy.mapToSource(proxy.index(0, MeasurementResultColumn.RESULT))
        self.assertEqual(source.measurement_id_at(source_index.row()), "small")

    def test_category_delegate_requests_domain_edit_by_measurement_id(self) -> None:
        document = _document([_measurement("editable", group_id="cotton")])
        source = MeasurementResultsModel()
        source.set_document(document)
        proxy = MeasurementResultsProxyModel()
        proxy.setSourceModel(source)
        view = QTableView()
        view.setModel(proxy)
        delegate = MeasurementGroupDelegate(view)
        requests: list[tuple[str, str | None]] = []
        source.groupChangeRequested.connect(
            lambda measurement_id, group_id: requests.append((measurement_id, group_id))
        )
        index = proxy.index(0, MeasurementResultColumn.GROUP)
        editor = delegate.createEditor(view, QStyleOptionViewItem(), index)
        self.assertIsInstance(editor, QComboBox)
        delegate.setEditorData(editor, index)
        self.assertEqual(editor.currentData(), "cotton")

        editor.setCurrentIndex(editor.findData("flax"))
        delegate.setModelData(editor, proxy, index)

        self.assertEqual(requests, [("editable", "flax")])
        view.close()

    def test_two_record_panes_share_filters_sort_and_selection_but_not_headers(self) -> None:
        document = _document(
            [
                _measurement("cotton", group_id="cotton", value=20.0),
                _measurement("flax", group_id="flax", value=5.0),
            ]
        )
        controller = MeasurementRecordsController()
        wide = MeasurementRecordsPane(controller, compact=False)
        compact = MeasurementRecordsPane(controller, compact=True)
        controller.set_document(document)

        self.assertIs(wide.table.model(), compact.table.model())
        self.assertIs(wide.table.selectionModel(), compact.table.selectionModel())
        source_label = str(
            controller.proxy_model.headerData(
                int(MeasurementResultColumn.RESULT_SEQUENCE),
                Qt.Orientation.Horizontal,
            )
        )
        self.assertEqual(source_label, "纤维结果序号")
        self.assertEqual(
            compact._header.display_label(
                int(MeasurementResultColumn.RESULT_SEQUENCE),
                source_label,
            ),
            "序号",
        )
        self.assertEqual(
            wide._header.display_label(
                int(MeasurementResultColumn.RESULT_SEQUENCE),
                source_label,
            ),
            "纤维结果序号",
        )
        wide.search_edit.setText("cotton")
        self.assertEqual(compact.search_edit.text(), "cotton")
        self.assertEqual(controller.proxy_model.rowCount(), 1)
        wide.search_edit.clear()

        controller.set_sort(MeasurementResultColumn.RESULT, Qt.SortOrder.AscendingOrder)
        self.assertEqual(
            controller.proxy_model.index(0, 0).data(MEASUREMENT_ID_ROLE),
            "flax",
        )
        self.assertTrue(controller.select_measurement_id("cotton"))
        self.assertEqual(controller.selected_measurement_ids(), ["cotton"])
        self.assertEqual(len(wide.table.selectionModel().selectedRows()), 1)
        self.assertEqual(len(compact.table.selectionModel().selectedRows()), 1)

        wide.table.setColumnHidden(int(MeasurementResultColumn.CREATED_AT), True)
        compact.table.setColumnHidden(int(MeasurementResultColumn.CREATED_AT), False)
        self.assertTrue(wide.table.isColumnHidden(int(MeasurementResultColumn.CREATED_AT)))
        self.assertFalse(compact.table.isColumnHidden(int(MeasurementResultColumn.CREATED_AT)))

        wide.close()
        compact.close()

    def test_record_header_state_rejects_legacy_unversioned_payload(self) -> None:
        controller = MeasurementRecordsController()
        pane = MeasurementRecordsPane(controller, compact=True)
        header = pane.table.horizontalHeader()
        legacy_state = bytes(header.saveState().toBase64()).decode("ascii")
        former_shared_schema_state = (
            f"{MeasurementRecordsPane.HEADER_STATE_SCHEMA}:{legacy_state}"
        )
        current_state = pane.save_header_state()
        self.assertTrue(
            current_state.startswith(f"{pane.header_state_schema}:")
        )

        pane.table.setColumnHidden(int(MeasurementResultColumn.ID), False)
        self.assertFalse(pane.restore_header_state(legacy_state))
        self.assertTrue(pane.table.isColumnHidden(int(MeasurementResultColumn.ID)))

        pane.table.setColumnWidth(int(MeasurementResultColumn.RESULT_SEQUENCE), 12)
        self.assertFalse(pane.restore_header_state(former_shared_schema_state))
        self.assertGreaterEqual(
            pane.table.columnWidth(int(MeasurementResultColumn.RESULT_SEQUENCE)),
            44,
        )

        pane.reset_columns()
        pane.table.setColumnHidden(int(MeasurementResultColumn.ID), False)
        self.assertTrue(pane.restore_header_state(current_state))
        self.assertTrue(pane.table.isColumnHidden(int(MeasurementResultColumn.ID)))
        pane.close()

    def test_record_category_filter_matches_unnamed_and_casefolded_chart_labels(self) -> None:
        unnamed = FiberGroup(
            id="unnamed",
            image_id="image",
            number=3,
            color="#2A9D8F",
            label="",
        )
        cotton = FiberGroup(
            id="named",
            image_id="image",
            number=4,
            color="#D97706",
            label="cotton",
        )
        document = ImageDocument(
            id="image",
            path="/tmp/category-filter.png",
            image_size=(100, 80),
            fiber_groups=[unnamed, cotton],
            measurements=[
                _measurement("unnamed-row", group_id=unnamed.id),
                _measurement("named-row", group_id=cotton.id),
            ],
        )
        controller = MeasurementRecordsController()
        controller.set_document(document)
        self.assertEqual(controller.group_labels(), ("3", "cotton"))

        controller.set_filters(group="3")
        self.assertEqual(controller.proxy_model.rowCount(), 1)
        self.assertEqual(
            controller.proxy_model.index(0, 0).data(MEASUREMENT_ID_ROLE),
            "unnamed-row",
        )

        controller.set_filters(group="Cotton")
        self.assertEqual(controller.proxy_model.rowCount(), 1)
        self.assertEqual(
            controller.proxy_model.index(0, 0).data(MEASUREMENT_ID_ROLE),
            "named-row",
        )


if __name__ == "__main__":
    unittest.main()
