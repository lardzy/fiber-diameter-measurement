from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.content_experiment import (
    CONTENT_EXPERIMENT_METADATA_KEY,
    ContentExperimentRecord,
    ContentExperimentSession,
    ContentFiberDefinition,
    ContentRecordKind,
    content_session_stats,
    write_session_to_project_metadata,
)
from fdm.geometry import Line, Point
from fdm.models import ProjectState, project_assets_root
from fdm.project_io import ProjectIO
from fdm.services import content_workbook
from fdm.services.content_workbook import ContentWorkbookService, content_template_path


class ContentExperimentTests(unittest.TestCase):
    def _session(self) -> ContentExperimentSession:
        cotton = ContentFiberDefinition(id="fiber_cotton", name="棉", color="#1F7A8C", density=1.54)
        lyocell = ContentFiberDefinition(id="fiber_lyocell", name="莱赛尔", color="#E07A5F", density=1.52)
        session = ContentExperimentSession(
            operator="张三",
            sample_id="S-001",
            sample_name="样品A",
            calibration_name="20x",
            calibration_pixels_per_unit=5.0,
            calibration_unit="um",
            fibers=[cotton, lyocell],
            current_fiber_id=cotton.id,
        )
        session.records.extend(
            [
                ContentExperimentRecord(id="rec_count_1", kind=ContentRecordKind.COUNT, fiber_id=cotton.id),
                ContentExperimentRecord(
                    id="rec_dia_1",
                    kind=ContentRecordKind.DIAMETER,
                    fiber_id=cotton.id,
                    source_mode="manual",
                    line_px=Line(Point(0, 0), Point(10, 0)),
                    diameter_px=10.0,
                    diameter_unit=2.0,
                ),
                ContentExperimentRecord(
                    id="rec_dia_2",
                    kind=ContentRecordKind.DIAMETER,
                    fiber_id=lyocell.id,
                    source_mode="snap",
                    line_px=Line(Point(0, 0), Point(12, 0)),
                    diameter_px=12.0,
                    diameter_unit=3.0,
                ),
            ]
        )
        return session

    def test_session_roundtrip_in_project_metadata(self) -> None:
        session = self._session()
        project = ProjectState.empty()
        write_session_to_project_metadata(project.metadata, session)

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "content.fdmproj"
            ProjectIO.save(project, path)
            loaded = ProjectIO.load(path)

        self.assertIn(CONTENT_EXPERIMENT_METADATA_KEY, loaded.metadata)
        loaded_session = ContentExperimentSession.from_dict(loaded.metadata[CONTENT_EXPERIMENT_METADATA_KEY])
        self.assertEqual(loaded_session.operator, "张三")
        self.assertEqual(loaded_session.sample_id, "S-001")
        self.assertEqual(loaded_session.calibration_name, "20x")
        self.assertEqual(loaded_session.calibration_pixels_per_unit, 5.0)
        self.assertEqual(loaded_session.calibration_unit, "um")
        self.assertEqual(len(loaded_session.fibers), 2)
        self.assertEqual(len(loaded_session.records), 3)
        self.assertEqual(loaded_session.records[1].source_mode, "manual")
        self.assertEqual(loaded_session.records[2].source_mode, "snap")

    def test_content_stats_use_counts_plus_measured_roots(self) -> None:
        session = self._session()
        stats = {item.fiber.name: item for item in content_session_stats(session)}

        self.assertEqual(stats["棉"].count, 1)
        self.assertEqual(stats["棉"].measured, 1)
        self.assertEqual(stats["棉"].total_roots, 2)
        self.assertAlmostEqual(stats["棉"].average_diameter or 0.0, 2.0)
        self.assertAlmostEqual(stats["莱赛尔"].average_diameter or 0.0, 3.0)
        self.assertIsNotNone(stats["棉"].content_percent)
        self.assertIsNotNone(stats["莱赛尔"].content_percent)

    def test_workbook_snapshot_writes_expected_cells(self) -> None:
        try:
            from openpyxl import load_workbook
        except ModuleNotFoundError as exc:  # pragma: no cover
            self.skipTest(f"openpyxl unavailable: {exc}")

        session = self._session()
        service = ContentWorkbookService()
        with TemporaryDirectory() as tmp_dir:
            project_path = Path(tmp_dir) / "content.fdmproj"
            snapshot = service.save_snapshot(session, project_path)
            workbook = load_workbook(snapshot, data_only=False)

            self.assertEqual(snapshot.parent, project_assets_root(project_path) / "content_experiments")
            self.assertEqual(workbook["特种毛(报出)"]["D2"].value, "样品A")
            self.assertEqual(workbook["特种毛(报出)"]["D3"].value, "S-001")
            self.assertEqual(workbook["特种毛(报出)"]["D7"].value, "张三")
            self.assertEqual(workbook["原始数据"]["D6"].value, "棉")
            self.assertEqual(workbook["原始数据"]["E6"].value, "莱赛尔")
            self.assertEqual(workbook["原始数据"]["D7"].value, 1.54)
            self.assertEqual(workbook["原始数据"]["D8"].value, 1)
            self.assertEqual(workbook["原始数据"]["D25"].value, 2.0)
            self.assertEqual(workbook["原始数据"]["E25"].value, 3.0)

    def test_windows_excel_mode_opens_sheet_template_when_com_available(self) -> None:
        session = self._session()
        service = ContentWorkbookService()

        class FakeWorkbooks:
            def __init__(self) -> None:
                self.add_template = ""

            def Add(self, *, Template: str):
                self.add_template = Template
                return object()

        class FakeExcel:
            def __init__(self) -> None:
                self.Visible = False
                self.Workbooks = FakeWorkbooks()

        fake_excel = FakeExcel()
        fake_client = types.ModuleType("win32com.client")
        fake_client.Dispatch = lambda name: fake_excel  # type: ignore[attr-defined]
        fake_win32com = types.ModuleType("win32com")
        fake_win32com.client = fake_client  # type: ignore[attr-defined]

        with (
            patch.object(content_workbook.sys, "platform", "win32"),
            patch.dict(sys.modules, {"win32com": fake_win32com, "win32com.client": fake_client}),
            patch.object(service, "sync_session", return_value=None),
        ):
            mode = service.open_session(session)

        self.assertEqual(mode, "excel")
        self.assertTrue(fake_excel.Visible)
        self.assertEqual(fake_excel.Workbooks.add_template, str(content_template_path()))
        self.assertEqual(service.last_warning, "")

    def test_windows_excel_mode_falls_back_to_xlsx_with_warning_when_com_fails(self) -> None:
        session = self._session()
        service = ContentWorkbookService()
        with (
            patch.object(content_workbook.sys, "platform", "win32"),
            patch.dict(sys.modules, {"win32com": types.ModuleType("win32com")}),
        ):
            mode = service.open_session(session)

        self.assertEqual(mode, "xlsx")
        self.assertIn("xlsx 快照模式", service.last_warning)
