from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.models import ImageDocument, ProjectState, new_id
from fdm.settings import AppSettings, RawRecordTemplate
from fdm.services.export_service import ExportScope, ExportSelection, ExportService, PlannedExportFile
from fdm.ui.export_controller import ExportController
from fdm.ui.project_session_controller import ProjectSessionController


PROJECT_VERSION = "test"


class _ProjectHost:
    def __init__(self, tmp_dir: Path) -> None:
        document = ImageDocument(id=new_id("image"), path=str(tmp_dir / "first.png"), image_size=(100, 80))
        document.initialize_runtime_state()
        self.project = ProjectState(version=PROJECT_VERSION, documents=[document])
        self._project_path = None
        self._load_thread = None
        self._pending_project_load_snapshot = False
        self._app_settings = AppSettings(recent_project_dir="")
        self.tmp_dir = tmp_dir
        self.default_save_path: Path | None = None
        self.remembered_directory: Path | None = None
        self.saved = False
        self.updated = False
        self.status_message = ""
        self.stopped_preview = False
        self.reset = False
        self.refreshed_presets = False
        self.open_requests: list[tuple[str, ImageDocument | None]] = []
        self.missing_paths: list[str] = []
        self.repaired_paths: list[str] = []

    def _show_project_information(self, title: str, message: str) -> None:
        raise AssertionError((title, message))

    def _show_project_warning(self, title: str, message: str) -> None:
        raise AssertionError((title, message))

    def _select_project_save_path(self, default_path: Path) -> str:
        self.default_save_path = default_path
        return str(self.tmp_dir / "named_project.fdmproj")

    def _preferred_dialog_directory(self, *, recent_dir: str = "") -> Path:
        del recent_dir
        return self.tmp_dir

    def _normalize_dialog_save_path(self, selected_path: str, default_filename: str) -> Path:
        del default_filename
        return Path(selected_path)

    def _remember_recent_directory(self, *, setting_name: str, directory: Path, context: str) -> None:
        del setting_name, context
        self.remembered_directory = directory

    def _project_asset_image_for_save(self, document: ImageDocument):
        del document
        return None

    def _mark_project_saved(self) -> None:
        self.saved = True

    def _update_ui_for_current_document(self) -> None:
        self.updated = True

    def _show_status_message(self, message: str, timeout_ms: int = 0) -> None:
        del timeout_ms
        self.status_message = message

    def _select_project_open_path(self) -> str:
        return ""

    def _confirm_close_documents(self, documents: list[ImageDocument]) -> bool:
        del documents
        return True

    def _merge_legacy_calibration_presets(self, presets: list[object]) -> int:
        del presets
        return 0

    def _reset_workspace(self) -> None:
        self.reset = True
        self.project = ProjectState.empty()

    def _refresh_preset_combo(self, *, selected_name: str | None = None) -> None:
        del selected_name
        self.refreshed_presets = True

    def _open_image_requests(
        self,
        requests: list[tuple[str, ImageDocument | None]],
        *,
        context_label: str,
        missing_paths: list[str] | None = None,
        repaired_paths: list[str] | None = None,
    ) -> None:
        del context_label
        self.open_requests = requests
        self.missing_paths = list(missing_paths or [])
        self.repaired_paths = list(repaired_paths or [])

    def stop_live_preview(self) -> None:
        self.stopped_preview = True


class _DialogCode:
    Accepted = 1


class _ExportDialog:
    DialogCode = _DialogCode

    def __init__(self, selection: ExportSelection) -> None:
        self._selection = selection

    def exec(self) -> int:
        return self.DialogCode.Accepted

    def selection(self) -> ExportSelection:
        return self._selection


class _ProgressDialog:
    def __init__(self) -> None:
        self.shown = False
        self.closed = False

    def show(self) -> None:
        self.shown = True

    def raise_(self) -> None:
        return

    def activateWindow(self) -> None:
        return


class _RecordingExportService(ExportService):
    def __init__(self, planned_outputs: list[PlannedExportFile]) -> None:
        self._planned_outputs = planned_outputs
        self.output_dir: Path | None = None
        self.single_output_path: Path | None = None
        self.documents: list[ImageDocument] = []
        self.selection: ExportSelection | None = None

    def planned_outputs(
        self,
        documents: list[ImageDocument],
        selection: ExportSelection | None = None,
    ) -> list[PlannedExportFile]:
        self.documents = list(documents)
        self.selection = selection
        return list(self._planned_outputs)

    def export_project(
        self,
        project: ProjectState,
        output_dir: str | Path,
        *,
        selection: ExportSelection | None = None,
        documents: list[ImageDocument] | None = None,
        overlay_renderer=None,
        single_output_path: str | Path | None = None,
        raw_record_template: RawRecordTemplate | None = None,
        progress_callback=None,
    ) -> dict[str, object]:
        del project, overlay_renderer, raw_record_template
        self.output_dir = Path(output_dir)
        self.single_output_path = Path(single_output_path) if single_output_path is not None else None
        self.documents = list(documents or [])
        self.selection = selection
        if progress_callback is not None:
            progress_callback(1, max(1, len(self._planned_outputs)), "完成", self.single_output_path)
        if self.single_output_path is not None:
            return {"xlsx": self.single_output_path}
        return {"files": [self.output_dir / item.filename for item in self._planned_outputs]}


class _ExportHost:
    def __init__(
        self,
        missing_template: Path | None = None,
        *,
        project: ProjectState | None = None,
        selection: ExportSelection | None = None,
        planned_outputs: list[PlannedExportFile] | None = None,
        default_dir: Path | None = None,
    ) -> None:
        self.project = project or ProjectState.empty()
        self._app_settings = AppSettings(
            raw_record_templates=(
                [RawRecordTemplate(name="缺失模板", path=str(missing_template))]
                if missing_template is not None
                else []
            ),
            last_raw_record_template_path=str(missing_template) if missing_template is not None else "",
        )
        self.selection = selection or ExportSelection(include_excel=True)
        self.export_service = _RecordingExportService(planned_outputs or [])
        self.default_dir = default_dir or Path.cwd()
        self.save_response = ""
        self.directory_response = ""
        self.save_requests: list[tuple[Path, str]] = []
        self.directory_requests: list[Path] = []
        self.information_messages: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []
        self.saved_contexts: list[str] = []
        self.remembered_directory: Path | None = None
        self.progress = _ProgressDialog()

    def current_document(self) -> ImageDocument | None:
        return self.project.documents[0] if self.project.documents else None

    def _create_export_options_dialog(self, preset: ExportSelection) -> _ExportDialog:
        del preset
        return _ExportDialog(self.selection)

    def _show_export_information(self, title: str, message: str) -> None:
        self.information_messages.append((title, message))

    def _show_export_warning(self, title: str, message: str) -> None:
        self.warnings.append((title, message))

    def _preferred_dialog_directory(self, *, recent_dir: str = "") -> Path:
        del recent_dir
        return self.default_dir

    def _select_export_save_path(self, default_path: Path, file_filter: str) -> str:
        self.save_requests.append((default_path, file_filter))
        return self.save_response

    def _select_export_directory(self, default_dir: Path) -> str:
        self.directory_requests.append(default_dir)
        return self.directory_response

    def _single_export_dialog_filter(self, filename: str) -> str:
        return f"filter:{Path(filename).suffix.lower()}"

    def _normalize_dialog_save_path(self, selected_path: str, default_filename: str) -> Path:
        path = Path(selected_path)
        if not path.suffix:
            path = path.with_suffix(Path(default_filename).suffix)
        return path

    def _create_blocking_progress_dialog(self, *, title: str, label_text: str, maximum: int) -> _ProgressDialog:
        del title, label_text, maximum
        return self.progress

    def _update_blocking_progress_dialog(
        self,
        progress,
        *,
        completed_steps: int,
        total_steps: int,
        label: str,
        path: Path | None,
    ) -> None:
        del progress, completed_steps, total_steps, label, path
        return

    def _pump_modal_progress_events(self) -> None:
        return

    def _close_progress_dialog(self, progress) -> None:
        progress.closed = True

    def _render_overlay_image(self, document: ImageDocument, kind: str, output_path: Path, render_mode: str) -> None:
        del document, kind, output_path, render_mode
        return

    def _format_export_failure_message(self, exc: Exception, *, export_path: Path | None) -> str:
        del export_path
        return str(exc)

    def _remember_recent_directory(self, *, setting_name: str, directory: Path, context: str) -> None:
        del setting_name, context
        self.remembered_directory = directory

    def _save_app_settings(self, *, context: str = "") -> None:
        self.saved_contexts.append(context)


class ProjectAndExportControllerTests(unittest.TestCase):
    def test_save_project_uses_first_default_path_and_remembers_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            host = _ProjectHost(Path(tmp))
            controller = ProjectSessionController(host)
            with patch("fdm.ui.project_session_controller.ProjectIO.save") as save_mock:
                self.assertTrue(controller.save_project())

            self.assertEqual(host.default_save_path, Path(tmp) / "fiber_measurement.fdmproj")
            self.assertEqual(save_mock.call_args.args[1], Path(tmp) / "named_project.fdmproj")
            self.assertEqual(host.remembered_directory, Path(tmp))
            self.assertTrue(host.saved)
            self.assertTrue(host.updated)
            self.assertIn("项目已保存", host.status_message)

    def test_load_project_repairs_missing_absolute_path_from_project_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_path = root / "moved_project.fdmproj"
            repaired_image = root / "first.png"
            repaired_image.write_bytes(b"fake")
            missing_absolute = root / "missing" / "first.png"
            document = ImageDocument(
                id=new_id("image"),
                path="first.png",
                absolute_path=str(missing_absolute),
                image_size=(100, 80),
            )
            document.initialize_runtime_state()
            loaded_project = ProjectState(version=PROJECT_VERSION, documents=[document])
            host = _ProjectHost(root)
            controller = ProjectSessionController(host)

            with patch("fdm.ui.project_session_controller.ProjectIO.load", return_value=loaded_project):
                controller.load_project_from_path(project_path)

            self.assertTrue(host.stopped_preview)
            self.assertTrue(host.reset)
            self.assertTrue(host.refreshed_presets)
            self.assertEqual(host.open_requests, [(str(repaired_image.resolve()), document)])
            self.assertEqual(host.missing_paths, [])
            self.assertEqual(document.path, str(repaired_image.resolve()))
            self.assertEqual(document.absolute_path, str(repaired_image.resolve()))
            self.assertEqual(host.repaired_paths, [f"{missing_absolute} -> {repaired_image.resolve()}"])
            self.assertIn("已自动修复 1 张图片路径", host.status_message)

    def test_export_controller_single_output_uses_save_dialog_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            document = ImageDocument(id=new_id("image"), path=str(root / "image.png"), image_size=(100, 80))
            document.initialize_runtime_state()
            project = ProjectState(version=PROJECT_VERSION, documents=[document])
            selection = ExportSelection(include_excel=True, scope=ExportScope.CURRENT)
            planned = [PlannedExportFile("xlsx", "纤维测量结果.xlsx")]
            host = _ExportHost(
                project=project,
                selection=selection,
                planned_outputs=planned,
                default_dir=root / "exports",
            )
            host.save_response = str(root / "custom_name")

            ExportController(host).export_results(selection)

            self.assertEqual(host.save_requests, [(root / "exports" / "纤维测量结果.xlsx", "filter:.xlsx")])
            self.assertEqual(host.directory_requests, [])
            self.assertEqual(host.export_service.single_output_path, root / "custom_name.xlsx")
            self.assertEqual(host.export_service.output_dir, root)
            self.assertEqual(host.remembered_directory, root)
            self.assertTrue(host.progress.shown)
            self.assertTrue(host.progress.closed)
            self.assertEqual(host.information_messages[0][0], "导出完成")

    def test_export_controller_multiple_outputs_uses_directory_dialog(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            document = ImageDocument(id=new_id("image"), path=str(root / "image.png"), image_size=(100, 80))
            document.initialize_runtime_state()
            project = ProjectState(version=PROJECT_VERSION, documents=[document])
            selection = ExportSelection(include_csv=True, scope=ExportScope.ALL_OPEN)
            planned = [
                PlannedExportFile("image_summary_csv", "图片汇总.csv"),
                PlannedExportFile("measurement_details_csv", "测量明细.csv"),
            ]
            output_dir = root / "chosen_exports"
            host = _ExportHost(
                project=project,
                selection=selection,
                planned_outputs=planned,
                default_dir=root / "exports",
            )
            host.directory_response = str(output_dir)

            ExportController(host).export_results(selection)

            self.assertEqual(host.save_requests, [])
            self.assertEqual(host.directory_requests, [root / "exports"])
            self.assertEqual(host.export_service.single_output_path, None)
            self.assertEqual(host.export_service.output_dir, output_dir)
            self.assertEqual(host.remembered_directory, output_dir)

    def test_export_controller_clears_missing_raw_record_template(self) -> None:
        with TemporaryDirectory() as tmp:
            missing_template = Path(tmp) / "missing_template.xlsm"
            host = _ExportHost(missing_template)
            selection = ExportSelection(
                include_excel=True,
                raw_record_template_path=str(missing_template),
            )

            template = ExportController(host).prepare_raw_record_template_for_export(selection)

            self.assertIsNone(template)
            self.assertEqual(selection.raw_record_template_path, "")
            self.assertEqual(host._app_settings.last_raw_record_template_path, "")
            self.assertEqual(host.saved_contexts, ["导出结果"])
            self.assertEqual(len(host.warnings), 1)
            self.assertIn("找不到已选择的原始记录模板", host.warnings[0][1])


if __name__ == "__main__":
    unittest.main()
