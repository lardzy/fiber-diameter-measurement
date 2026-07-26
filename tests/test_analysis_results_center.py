from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from PySide6.QtWidgets import QApplication, QComboBox, QScrollArea

    from fdm.analysis_artifacts import (
        AnalysisArtifact,
        AnalysisArtifactStatus,
        AnalysisAssetKind,
        AnalysisAssetReference,
        AnalysisCurve,
        AnalysisObjectKind,
        AnalysisObjectReference,
        AnalysisTable,
    )
    from fdm.services.analysis_asset_io import write_safe_analysis_npz
    from fdm.ui.analysis_results_center import (
        AnalysisActionRequest,
        AnalysisExportRequest,
        AnalysisLocateRequest,
        AnalysisResultsCenter,
        _load_bounded_asset_preview,
    )

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


class _FakeWheelEvent:
    def __init__(self) -> None:
        self.ignored = False
        self.accepted = False

    def ignore(self) -> None:
        self.ignored = True

    def accept(self) -> None:
        self.accepted = True


def _artifact(
    artifact_id: str,
    *,
    document_id: str = "doc_1",
    tool_id: str = "fdm.particle_analysis",
    status: AnalysisArtifactStatus = AnalysisArtifactStatus.CURRENT,
    category: str = "玻璃纤维",
    assets: tuple[AnalysisAssetReference, ...] = (),
) -> AnalysisArtifact:
    return AnalysisArtifact(
        id=artifact_id,
        source_document_id=document_id,
        source_pixel_revision=4,
        source_reference=AnalysisObjectReference(
            AnalysisObjectKind.ROI,
            f"roi_{artifact_id}",
            2,
        ),
        tool_id=tool_id,
        tool_version="1",
        parameters={"category_label": category, "connectivity": 8},
        scalars={"included_pixel_count": 3, "mean": 2.0},
        tables=(
            AnalysisTable(
                name="粒子明细",
                columns=("序号", "面积"),
                rows=((1, 12.5), (2, 9.0)),
            ),
        ),
        curves=(
            AnalysisCurve(
                name="面积分布",
                x=(1.0, 2.0, 3.0),
                y=(2.0, 5.0, 1.0),
                x_unit="µm²",
                y_unit="频数",
            ),
        ),
        assets=assets,
        status=status,
        stale_reason="来源像素已变化" if status is AnalysisArtifactStatus.STALE else None,
        created_at="2026-07-27T08:00:00+00:00",
    )


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class AnalysisResultsCenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.current = _artifact("current")
        self.stale = _artifact(
            "stale",
            document_id="doc_2",
            tool_id="fdm.histogram",
            status=AnalysisArtifactStatus.STALE,
            category="碳纤维",
        )
        self.dialog = AnalysisResultsCenter(
            (self.current, self.stale),
            document_names={"doc_1": "第一张图片", "doc_2": "第二张图片"},
            roi_names={
                "roi_current": "孔洞 ROI",
                "roi_stale": "历史 ROI",
            },
        )
        self.dialog.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.dialog.close()

    def test_non_modal_chinese_pages_status_and_stale_reason(self) -> None:
        self.assertFalse(self.dialog.isModal())
        self.assertEqual(self.dialog.windowTitle(), "分析结果中心")
        tabs = [
            self.dialog._tabs.tabText(index)  # noqa: SLF001
            for index in range(self.dialog._tabs.count())  # noqa: SLF001
        ]
        self.assertEqual(
            tabs,
            ["分析摘要", "参数与来源", "详细表格", "曲线 / 直方图", "标签图 / 资产"],
        )
        self.assertEqual(self.dialog._count_label.text(), "2 项结果")  # noqa: SLF001
        self.dialog._artifact_table.selectRow(1)  # noqa: SLF001
        self.app.processEvents()
        self.assertIn("已失效", self.dialog._selection_status.text())  # noqa: SLF001
        self.assertIn("来源像素已变化", self.dialog._selection_status.text())  # noqa: SLF001
        self.assertFalse(self.dialog._convert_button.isEnabled())  # noqa: SLF001

    def test_internal_scalar_keys_are_presented_with_chinese_labels(self) -> None:
        summary = self.dialog._scalar_summary(self.current)  # noqa: SLF001
        self.assertIn("纳入像素数=3", summary)
        self.assertIn("均值=2.0", summary)
        self.assertNotIn("included_pixel_count", summary)

    def test_document_roi_category_tool_and_status_filters(self) -> None:
        dialog = self.dialog
        doc_index = dialog._document_filter.findData("doc_2")  # noqa: SLF001
        dialog._document_filter.setCurrentIndex(doc_index)  # noqa: SLF001
        self.app.processEvents()
        self.assertEqual([item.id for item in dialog.filtered_artifacts()], ["stale"])

        dialog._document_filter.setCurrentIndex(0)  # noqa: SLF001
        status_index = dialog._status_filter.findData("current")  # noqa: SLF001
        dialog._status_filter.setCurrentIndex(status_index)  # noqa: SLF001
        self.app.processEvents()
        self.assertEqual([item.id for item in dialog.filtered_artifacts()], ["current"])

        dialog._status_filter.setCurrentIndex(0)  # noqa: SLF001
        category_index = dialog._category_filter.findData("碳纤维")  # noqa: SLF001
        dialog._category_filter.setCurrentIndex(category_index)  # noqa: SLF001
        self.app.processEvents()
        self.assertEqual([item.id for item in dialog.filtered_artifacts()], ["stale"])

    def test_locate_recalculate_convert_and_export_signals_are_structured(self) -> None:
        located: list[AnalysisLocateRequest] = []
        recalculated: list[AnalysisActionRequest] = []
        converted: list[AnalysisActionRequest] = []
        exported: list[AnalysisExportRequest] = []
        self.dialog.locateRequested.connect(located.append)
        self.dialog.recalculateRequested.connect(recalculated.append)
        self.dialog.convertToMeasurementRequested.connect(converted.append)
        self.dialog.exportRequested.connect(exported.append)

        self.dialog._artifact_table.selectRow(0)  # noqa: SLF001
        selected_item = self.dialog._artifact_table.item(0, 0)  # noqa: SLF001
        self.dialog._artifact_table.itemClicked.emit(selected_item)  # noqa: SLF001
        self.dialog._recalculate_button.click()  # noqa: SLF001
        self.dialog._convert_button.click()  # noqa: SLF001
        self.dialog._export_button.click()  # noqa: SLF001

        self.assertEqual(located[0].artifact_id, "current")
        self.assertEqual(located[0].object_id, "roi_current")
        self.assertEqual(recalculated[0].artifact_ids, ("current",))
        self.assertEqual(converted[0].artifact_ids, ("current",))
        self.assertEqual(exported[0].artifact_ids, ("current", "stale"))
        self.assertIsNone(exported[0].selected_table_name)

    def test_all_dropdowns_ignore_incidental_wheel(self) -> None:
        combos = self.dialog.findChildren(QComboBox)
        self.assertTrue(combos)
        for combo in combos:
            with self.subTest(combo=combo.objectName() or type(combo).__name__):
                before = combo.currentIndex()
                event = _FakeWheelEvent()
                combo.wheelEvent(event)
                self.assertEqual(combo.currentIndex(), before)
                self.assertTrue(event.ignored or event.accepted)

    def test_small_window_retains_scrollable_content(self) -> None:
        self.dialog.resize(self.dialog.minimumSize())
        self.app.processEvents()

        scroll = self.dialog.findChild(QScrollArea, "analysisResultsScroll")
        self.assertIsNotNone(scroll)
        self.assertTrue(
            scroll.horizontalScrollBar().maximum() > 0
            or scroll.verticalScrollBar().maximum() > 0
        )
        self.assertGreaterEqual(self.dialog._detail_table.rowCount(), 2)  # noqa: SLF001

    def test_session_asset_mapping_precedes_project_root_and_previews_skeleton(
        self,
    ) -> None:
        self.dialog.close()
        with TemporaryDirectory() as project_dir, TemporaryDirectory() as session_dir:
            relative = "analysis/live/skeleton.npz"
            source = Path(session_dir) / relative
            info = write_safe_analysis_npz(
                source,
                schema="fdm.skeleton-network.v1",
                arrays={
                    "skeleton": np.eye(32, dtype=np.uint8),
                    "endpoints_xy": np.asarray(
                        ((0.0, 0.0), (31.0, 31.0)),
                        dtype=np.float64,
                    ),
                    "branchpoints_xy": np.empty((0, 2), dtype=np.float64),
                    "branches": np.empty((0, 7), dtype=np.float64),
                },
            )
            reference = AnalysisAssetReference(
                kind=AnalysisAssetKind.GRAPH,
                path=relative,
                sha256=info.sha256,
                media_type="application/x-npz",
                metadata={
                    "schema": info.schema,
                    "allow_pickle": False,
                    "members": {
                        name: {"dtype": dtype, "shape": list(shape)}
                        for name, dtype, shape in info.members
                    },
                },
            )
            artifact = _artifact("mapped", assets=(reference,))
            dialog = AnalysisResultsCenter(
                (artifact,),
                asset_root=project_dir,
                asset_source_paths={relative: source},
            )
            dialog.show()
            self.app.processEvents()
            dialog._preview_thread_pool.waitForDone(3000)  # noqa: SLF001
            for _ in range(5):
                self.app.processEvents()

            self.assertEqual(
                dialog._asset_candidate(reference),  # noqa: SLF001
                source,
            )
            self.assertEqual(
                dialog._preview_thread_pool.maxThreadCount(),  # noqa: SLF001
                1,
            )
            self.assertFalse(dialog._asset_preview.pixmap().isNull())  # noqa: SLF001
            self.assertIn(
                "骨架网络",
                dialog._asset_preview_description.text(),  # noqa: SLF001
            )
            dialog.close()

    def test_known_heatmap_npz_is_loaded_with_bounded_safe_preview(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "thickness.npz"
            info = write_safe_analysis_npz(
                source,
                schema="fdm.local-thickness.v1",
                arrays={
                    "thickness_px": np.arange(
                        48 * 64,
                        dtype=np.float32,
                    ).reshape((48, 64)),
                    "maximal_circles": np.empty((0, 4), dtype=np.float64),
                },
            )
            reference = AnalysisAssetReference(
                kind=AnalysisAssetKind.OTHER,
                path="analysis/test/thickness.npz",
                sha256=info.sha256,
                media_type="application/x-npz",
                metadata={
                    "schema": info.schema,
                    "allow_pickle": False,
                    "members": {
                        name: {"dtype": dtype, "shape": list(shape)}
                        for name, dtype, shape in info.members
                    },
                },
            )

            rgb, description = _load_bounded_asset_preview(
                source,
                reference,
            )

            self.assertEqual(rgb.shape, (48, 64, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertIn("局部厚度热力图", description)
