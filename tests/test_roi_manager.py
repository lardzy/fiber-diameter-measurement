from __future__ import annotations

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QAbstractScrollArea,
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QSpinBox,
    )

    from fdm.project_roi import (
        EllipseRoiGeometry,
        ProjectRoi,
        ProjectRoiKind,
        RectangleRoiGeometry,
        RoiBooleanOperator,
    )
    from fdm.ui.roi_manager import (
        RoiBooleanRequest,
        RoiCreateFromAreaRequest,
        RoiCreateRequest,
        RoiDeleteRequest,
        RoiManagerPanel,
        RoiMetadataChangeRequest,
        RoiSelectionRequest,
    )

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


def _roi(
    roi_id: str,
    *,
    document_id: str = "doc-1",
    name: str | None = None,
    group: str | None = None,
    visible: bool = True,
    locked: bool = False,
    color: str = "#2A9D8F",
    revision: int = 0,
) -> ProjectRoi:
    return ProjectRoi(
        id=roi_id,
        document_id=document_id,
        name=name or roi_id,
        group=group,
        visible=visible,
        locked=locked,
        color=color,
        revision=revision,
        geometry=RectangleRoiGeometry(1, 2, 30, 40),
    )


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class RoiManagerPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.rois = (
            _roi(
                "roi-long",
                name="激光共聚焦超长超长超长超长超长超长 ROI 名称",
                group="孔洞复核组",
                color="#EF476F",
                revision=7,
            ),
            _roi(
                "roi-locked",
                name="已锁定区域",
                visible=False,
                locked=True,
                color="#FFD166",
                revision=2,
            ),
            _roi(
                "foreign",
                document_id="doc-2",
                name="其他图片 ROI",
            ),
        )
        self.panel = RoiManagerPanel()
        self.panel.resize(260, 420)
        self.panel.set_current_document("doc-1")
        self.panel.set_rois(self.rois)
        self.panel.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.panel.close()

    def _item(self, roi_id: str):
        for index in range(self.panel._tree.topLevelItemCount()):  # noqa: SLF001
            item = self.panel._tree.topLevelItem(index)  # noqa: SLF001
            if item.data(0, Qt.ItemDataRole.UserRole) == roi_id:
                return item
        self.fail(f"找不到 ROI 项：{roi_id}")

    def test_filters_to_current_document_and_long_name_does_not_widen_panel(
        self,
    ) -> None:
        tree = self.panel._tree  # noqa: SLF001
        self.assertEqual(tree.topLevelItemCount(), 2)
        self.assertEqual(self.panel._count_label.text(), "2 个")  # noqa: SLF001
        long_item = self._item("roi-long")
        self.assertIn("孔洞复核组", long_item.text(0))
        self.assertIn("激光共聚焦", long_item.toolTip(0))
        self.assertIn("分组：孔洞复核组", long_item.toolTip(0))
        self.assertEqual(
            tree.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertLessEqual(self.panel.minimumSizeHint().width(), 220)
        self.assertLessEqual(tree.columnWidth(0), 160)

        self.panel.set_current_document("doc-2")
        self.app.processEvents()
        self.assertEqual(tree.topLevelItemCount(), 1)
        self.assertEqual(tree.topLevelItem(0).text(0), "其他图片 ROI")

    def test_empty_and_search_states_remain_reachable_in_small_panel(self) -> None:
        self.panel.resize(220, 260)
        self.panel._search_edit.setText("不存在")  # noqa: SLF001
        self.app.processEvents()
        self.assertFalse(self.panel._tree.isVisible())  # noqa: SLF001
        self.assertTrue(self.panel._empty_label.isVisible())  # noqa: SLF001
        self.assertEqual(
            self.panel._empty_label.text(),  # noqa: SLF001
            "没有匹配的 ROI",
        )

        self.panel._search_edit.clear()  # noqa: SLF001
        self.app.processEvents()
        self.assertTrue(self.panel._tree.isVisible())  # noqa: SLF001
        scroll_areas = [
            child
            for child in self.panel.findChildren(QAbstractScrollArea)
            if child.objectName() != "qt_scrollarea_hcontainer"
            and type(child).__name__ != "QHeaderView"
        ]
        self.assertEqual(scroll_areas, [self.panel._tree])  # noqa: SLF001

    def test_create_and_create_from_area_emit_immutable_requests(self) -> None:
        creates: list[RoiCreateRequest] = []
        from_area: list[RoiCreateFromAreaRequest] = []
        self.panel.createRequested.connect(creates.append)
        self.panel.createFromAreaRequested.connect(from_area.append)

        self.assertTrue(self.panel.request_create(ProjectRoiKind.ELLIPSE))
        self.assertEqual(creates[0].document_id, "doc-1")
        self.assertEqual(creates[0].kind, ProjectRoiKind.ELLIPSE)
        with self.assertRaises(FrozenInstanceError):
            creates[0].document_id = "changed"  # type: ignore[misc]

        self.assertFalse(self.panel.request_create_from_area())
        self.panel.set_current_area_measurement("area-42")
        self.assertTrue(self.panel._from_area_button.isEnabled())  # noqa: SLF001
        self.assertTrue(self.panel.request_create_from_area())
        self.assertEqual(from_area[0].measurement_id, "area-42")
        self.assertEqual(from_area[0].document_id, "doc-1")

    def test_visibility_and_lock_are_controlled_metadata_requests(self) -> None:
        requests: list[RoiMetadataChangeRequest] = []
        self.panel.metadataChangeRequested.connect(requests.append)
        item = self._item("roi-long")
        original = self.rois[0]

        item.setCheckState(1, Qt.CheckState.Unchecked)
        self.app.processEvents()
        self.assertEqual(len(requests), 1)
        visibility = requests[-1]
        self.assertFalse(visibility.visible)
        self.assertFalse(visibility.locked)
        self.assertEqual(visibility.target.roi_id, "roi-long")
        self.assertEqual(visibility.target.expected_revision, 7)
        self.assertEqual(item.checkState(1), Qt.CheckState.Checked)

        item.setCheckState(2, Qt.CheckState.Checked)
        self.app.processEvents()
        self.assertEqual(len(requests), 2)
        self.assertTrue(requests[-1].locked)
        self.assertEqual(item.checkState(2), Qt.CheckState.Unchecked)
        self.assertEqual(self.rois[0], original)

    def test_rename_group_and_color_requests_keep_complete_metadata(self) -> None:
        requests: list[RoiMetadataChangeRequest] = []
        self.panel.metadataChangeRequested.connect(requests.append)
        self.panel.select_rois(("roi-long",))

        self.assertTrue(self.panel.request_rename("复核 ROI"))
        self.assertEqual(requests[-1].name, "复核 ROI")
        self.assertEqual(requests[-1].group, "孔洞复核组")
        self.assertEqual(requests[-1].color, "#EF476F")

        self.assertTrue(self.panel.request_group(""))
        self.assertIsNone(requests[-1].group)

        self.assertTrue(self.panel.request_color("#abcdef"))
        self.assertEqual(requests[-1].color, "#ABCDEF")
        self.assertEqual(self.rois[0].name, "激光共聚焦超长超长超长超长超长超长 ROI 名称")
        self.assertEqual(self.rois[0].group, "孔洞复核组")

    def test_multi_selection_boolean_delete_and_selection_are_structured(
        self,
    ) -> None:
        selections: list[RoiSelectionRequest] = []
        booleans: list[RoiBooleanRequest] = []
        deletions: list[RoiDeleteRequest] = []
        self.panel.selectionChanged.connect(selections.append)
        self.panel.booleanOperationRequested.connect(booleans.append)
        self.panel.deleteRequested.connect(deletions.append)

        self.panel.select_rois(("roi-long", "roi-locked"), emit_signal=True)
        self.assertEqual(set(selections[-1].roi_ids), {"roi-long", "roi-locked"})
        self.assertTrue(self.panel._boolean_button.isEnabled())  # noqa: SLF001
        self.assertTrue(self.panel.request_boolean(RoiBooleanOperator.XOR))
        self.assertEqual(booleans[0].operator, RoiBooleanOperator.XOR)
        self.assertEqual(
            {ref.roi_id for ref in booleans[0].operands},
            {"roi-long", "roi-locked"},
        )
        self.assertEqual(
            {ref.expected_revision for ref in booleans[0].operands},
            {7, 2},
        )

        self.assertTrue(self.panel.request_delete())
        self.assertEqual(
            {ref.roi_id for ref in deletions[0].targets},
            {"roi-long", "roi-locked"},
        )

    def test_difference_places_current_row_first(self) -> None:
        requests: list[RoiBooleanRequest] = []
        self.panel.booleanOperationRequested.connect(requests.append)
        self.panel.select_rois(("roi-locked", "roi-long"))
        self.assertTrue(self.panel.request_boolean(RoiBooleanOperator.DIFFERENCE))
        self.assertEqual(
            tuple(ref.roi_id for ref in requests[0].operands),
            ("roi-locked", "roi-long"),
        )

    def test_selection_and_activation_emit_without_changing_model(self) -> None:
        selections: list[RoiSelectionRequest] = []
        locations: list[RoiSelectionRequest] = []
        self.panel.selectionChanged.connect(selections.append)
        self.panel.locateRequested.connect(locations.append)
        item = self._item("roi-long")
        self.panel._tree.setCurrentItem(item)  # noqa: SLF001
        item.setSelected(True)
        self.panel._tree.itemActivated.emit(item, 0)  # noqa: SLF001
        self.app.processEvents()
        self.assertEqual(locations[-1].document_id, "doc-1")
        self.assertEqual(locations[-1].roi_ids, ("roi-long",))
        self.assertTrue(any("roi-long" in event.roi_ids for event in selections))
        self.assertEqual(self.rois[0].revision, 7)

    def test_no_dropdown_or_numeric_editor_can_consume_scroll(self) -> None:
        self.assertEqual(self.panel.findChildren(QComboBox), [])
        self.assertEqual(self.panel.findChildren(QSpinBox), [])
        self.assertEqual(self.panel.findChildren(QDoubleSpinBox), [])
        self.assertIsNotNone(self.panel._create_button.menu())  # noqa: SLF001
        self.assertIsNotNone(self.panel._boolean_button.menu())  # noqa: SLF001

    def test_runtime_update_preserves_selection_and_accepts_other_geometry(
        self,
    ) -> None:
        self.panel.select_rois(("roi-locked",))
        updated = ProjectRoi(
            id="roi-locked",
            document_id="doc-1",
            name="椭圆复核",
            geometry=EllipseRoiGeometry(5, 6, 20, 18),
            color="#118AB2",
            revision=3,
        )
        self.panel.set_rois((self.rois[0], updated, self.rois[2]))
        self.app.processEvents()
        self.assertEqual(self.panel.selected_roi_ids(), ("roi-locked",))
        item = self._item("roi-locked")
        self.assertIn("椭圆 ROI", item.toolTip(0))
        self.assertIn("#118AB2", item.toolTip(3))


if __name__ == "__main__":
    unittest.main()
