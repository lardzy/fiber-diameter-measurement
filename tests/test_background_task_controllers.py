from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from PySide6.QtCore import QThread
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication

    from fdm.models import ImageDocument, ProjectState, new_id
    from fdm.settings import AppSettings
    from fdm.ui.area_inference_worker import AreaInferenceRequest
    from fdm.ui.background_task_controller import BackgroundTaskController, BatchLoadState
    from fdm.ui.image_loader import ImageLoadRequest
    from fdm.ui.preview_analysis_task_controller import PreviewAnalysisTaskController
    from fdm.ui.thread_task_manager import ThreadTaskManager

    QT_CONTROLLER_AVAILABLE = True
except ModuleNotFoundError:
    QThread = None
    QColor = None
    QImage = None
    QApplication = None
    ImageDocument = None
    ProjectState = None
    AppSettings = None
    AreaInferenceRequest = None
    BackgroundTaskController = None
    BatchLoadState = None
    ImageLoadRequest = None
    PreviewAnalysisTaskController = None
    ThreadTaskManager = None
    new_id = None
    QT_CONTROLLER_AVAILABLE = False


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _spin_until(predicate, *, timeout_s: float = 2.0) -> None:
    app = _app()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


class _FakeProgress:
    def __init__(self) -> None:
        self.canceled = _FakeSignal()
        self.shown = False
        self.closed = False
        self.deleted = False
        self.value = 0
        self.maximum = 0
        self.label = ""

    def show(self) -> None:
        self.shown = True

    def setMaximum(self, value: int) -> None:
        self.maximum = value

    def setValue(self, value: int) -> None:
        self.value = value

    def setLabelText(self, value: str) -> None:
        self.label = value

    def close(self) -> None:
        self.closed = True

    def deleteLater(self) -> None:
        self.deleted = True


class _BackgroundHost:
    def __init__(self, project: ProjectState | None = None) -> None:
        self.project = project or ProjectState.empty()
        self._app_settings = AppSettings()
        self._pending_project_load_snapshot = False
        self.progresses: list[_FakeProgress] = []
        self.loaded: list[str] = []
        self.summaries: list[BatchLoadState] = []
        self.project_marked_saved = False
        self.applied_area: list[tuple[str, list[object]]] = []
        self.warnings: list[str] = []
        self.status_messages: list[str] = []
        self.loaded_callback_threads: list[QThread] = []
        self.area_callback_threads: list[QThread] = []

    def _create_progress_dialog(self, *, title: str, label_text: str, maximum: int):
        del title, label_text, maximum
        progress = _FakeProgress()
        self.progresses.append(progress)
        return progress

    def _add_loaded_document(self, request: ImageLoadRequest, image: QImage) -> None:
        del image
        self.loaded_callback_threads.append(QThread.currentThread())
        self.loaded.append(request.path)

    def _show_batch_load_summary(self, state: BatchLoadState) -> None:
        self.summaries.append(state)

    def _mark_project_saved(self) -> None:
        self.project_marked_saved = True

    def _apply_area_inference_result(
        self,
        document: ImageDocument,
        instances: list[object],
        *,
        global_group_labels: list[str] | None,
        model_name: str,
        update_project_group_templates: bool,
    ) -> None:
        del global_group_labels, model_name, update_project_group_templates
        self.area_callback_threads.append(QThread.currentThread())
        self.applied_area.append((document.id, instances))

    def _show_area_inference_warning(self, message: str) -> None:
        self.warnings.append(message)

    def _show_status_message(self, message: str, timeout_ms: int = 0) -> None:
        del timeout_ms
        self.status_messages.append(message)

    def _on_prompt_segmentation_succeeded(self, document_id: str, request_id: int, result: object) -> None:
        del document_id, request_id, result

    def _on_prompt_segmentation_failed(self, document_id: str, request_id: int, reason: str) -> None:
        del document_id, request_id, reason

    def _on_fiber_quick_geometry_succeeded(self, document_id: str, request_id: int, result: object) -> None:
        del document_id, request_id, result

    def _on_fiber_quick_geometry_failed(self, document_id: str, request_id: int, reason: str) -> None:
        del document_id, request_id, reason

    def _on_fiber_quick_commit_geometry_succeeded(self, document_id: str, request_id: int, result: object) -> None:
        del document_id, request_id, result

    def _on_fiber_quick_commit_geometry_failed(self, document_id: str, request_id: int, reason: str) -> None:
        del document_id, request_id, reason

    def _on_reference_instance_succeeded(self, document_id: str, request_id: int, result: object) -> None:
        del document_id, request_id, result

    def _on_reference_instance_failed(self, document_id: str, request_id: int, reason: str) -> None:
        del document_id, request_id, reason


class _FakeDialog:
    def __init__(self) -> None:
        self.shown = False
        self.closed = False
        self.statuses: list[str] = []
        self.busy: list[tuple[bool, str]] = []

    def show(self) -> None:
        self.shown = True

    def raise_(self) -> None:
        return

    def activateWindow(self) -> None:
        return

    def close_silently(self) -> None:
        self.closed = True

    def set_status(self, text: str) -> None:
        self.statuses.append(text)

    def set_busy(self, active: bool, text: str) -> None:
        self.busy.append((active, text))


class _FakeSignal:
    def __init__(self) -> None:
        self.emitted: list[object] = []
        self._callbacks: list[object] = []

    def connect(self, callback, *args) -> None:
        del args
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        self.emitted.append(args if args else "emit")
        for callback in list(self._callbacks):
            callback(*args)


class _PreviewHost:
    def __init__(self) -> None:
        self.dialog = _FakeDialog()
        self.frame_requests: list[int] = []
        self.status_messages: list[str] = []
        self.action_updates = 0
        self.synced_buttons = 0

    def _selected_capture_device(self):
        return type("Device", (), {"id": "microview:0", "name": "Microview #1"})()

    def _clear_magic_segment_sessions(self) -> None:
        return

    def _create_preview_analysis_dialog(self, mode: str) -> _FakeDialog:
        del mode
        return self.dialog

    def _analysis_mode_label(self, mode: str) -> str:
        return "景深合成" if mode == "focus_stack" else "地图构建"

    def _preview_analysis_finalize_message(self, mode: str) -> str:
        return f"正在完成{self._analysis_mode_label(mode)}，请稍候…"

    def _current_focus_stack_render_config(self):
        return None

    def _preview_analysis_interval_ms(self, mode: str) -> int:
        del mode
        return 50

    def _request_capture_analysis_frame(self, request_id: int) -> bool:
        self.frame_requests.append(request_id)
        return True

    def _on_preview_analysis_worker_preview(self, payload: object) -> None:
        del payload

    def _on_preview_analysis_worker_finished(self, payload: object) -> None:
        del payload

    def _on_preview_analysis_worker_failed(self, message: str) -> None:
        del message

    def _sync_preview_analysis_buttons(self) -> None:
        self.synced_buttons += 1

    def _update_action_states(self) -> None:
        self.action_updates += 1

    def _show_status_message(self, message: str, timeout_ms: int = 0) -> None:
        del timeout_ms
        self.status_messages.append(message)


@unittest.skipUnless(QT_CONTROLLER_AVAILABLE, "requires PySide6")
class BackgroundTaskControllerTests(unittest.TestCase):
    def test_batch_load_controller_closes_progress_and_preserves_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.png"
            missing_path = Path(tmp) / "missing.png"
            image = QImage(32, 24, QImage.Format.Format_RGB32)
            image.fill(QColor("#FFFFFF"))
            image.save(str(image_path))
            host = _BackgroundHost()
            manager = ThreadTaskManager(parent=_app())
            controller = BackgroundTaskController(host, manager)

            controller.start_batch_image_load(
                [
                    ImageLoadRequest(path=str(image_path)),
                    ImageLoadRequest(path=str(missing_path)),
                ],
                context_label="打开图片",
                skipped_count=1,
            )
            _spin_until(lambda: controller.load_state is None)

            self.assertEqual(host.loaded, [str(image_path)])
            self.assertEqual(len(host.summaries), 1)
            self.assertEqual(host.summaries[0].loaded_count, 1)
            self.assertEqual(host.summaries[0].failed_count, 1)
            self.assertTrue(host.progresses[0].closed)

    def test_batch_load_callbacks_run_on_gui_thread(self) -> None:
        app = _app()
        with TemporaryDirectory() as tmp:
            paths: list[Path] = []
            for index in range(2):
                image_path = Path(tmp) / f"image-{index}.png"
                image = QImage(32, 24, QImage.Format.Format_RGB32)
                image.fill(QColor("#FFFFFF"))
                image.save(str(image_path))
                paths.append(image_path)
            host = _BackgroundHost()
            manager = ThreadTaskManager(parent=app)
            controller = BackgroundTaskController(host, manager, parent=app)

            controller.start_batch_image_load(
                [ImageLoadRequest(path=str(path)) for path in paths],
                context_label="打开文件夹",
                skipped_count=0,
            )
            _spin_until(lambda: controller.load_state is None)

            self.assertEqual(host.loaded, [str(path) for path in paths])
            self.assertTrue(host.loaded_callback_threads)
            self.assertTrue(all(thread is app.thread() for thread in host.loaded_callback_threads))

    def test_area_inference_controller_applies_success_and_reports_failure(self) -> None:
        document = ImageDocument(id=new_id("image"), path="/tmp/area-controller.png", image_size=(80, 60))
        document.initialize_runtime_state()
        host = _BackgroundHost(ProjectState(version="test", documents=[document]))
        manager = ThreadTaskManager(parent=_app())
        controller = BackgroundTaskController(host, manager)

        class _Result:
            instances = ["ok"]

        def fake_infer_image(**kwargs):
            if kwargs["image_path"].endswith("bad.png"):
                raise RuntimeError("bad image")
            return _Result()

        requests = [
            AreaInferenceRequest(document_id=document.id, image_path="/tmp/good.png", model_name="棉", model_file="m.pth"),
            AreaInferenceRequest(document_id=document.id, image_path="/tmp/bad.png", model_name="棉", model_file="m.pth"),
        ]
        with patch("fdm.ui.area_inference_worker.AreaInferenceService.infer_image", side_effect=fake_infer_image):
            controller.start_area_inference_batch(requests, model_name="棉")
            _spin_until(lambda: controller.area_infer_state is None)

        self.assertEqual(host.applied_area, [(document.id, ["ok"])])
        self.assertEqual(len(host.warnings), 1)
        self.assertIn("bad image", host.warnings[0])
        self.assertTrue(host.progresses[0].closed)

    def test_area_inference_callbacks_run_on_gui_thread(self) -> None:
        app = _app()
        document = ImageDocument(id=new_id("image"), path="/tmp/area-thread.png", image_size=(80, 60))
        document.initialize_runtime_state()
        host = _BackgroundHost(ProjectState(version="test", documents=[document]))
        manager = ThreadTaskManager(parent=app)
        controller = BackgroundTaskController(host, manager, parent=app)

        class _Result:
            instances = ["ok"]

        request = AreaInferenceRequest(document_id=document.id, image_path="/tmp/good.png", model_name="棉", model_file="m.pth")
        with patch("fdm.ui.area_inference_worker.AreaInferenceService.infer_image", return_value=_Result()):
            controller.start_area_inference_batch([request], model_name="棉")
            _spin_until(lambda: controller.area_infer_state is None)

        self.assertEqual(host.applied_area, [(document.id, ["ok"])])
        self.assertTrue(host.area_callback_threads)
        self.assertTrue(all(thread is app.thread() for thread in host.area_callback_threads))

    def test_persistent_workers_are_ensured_and_shutdown(self) -> None:
        host = _BackgroundHost()
        manager = ThreadTaskManager(parent=_app())
        controller = BackgroundTaskController(host, manager)

        prompt = controller.ensure_prompt_segmentation_worker()
        fiber = controller.ensure_fiber_quick_geometry_worker()
        reference = controller.ensure_reference_instance_worker()

        self.assertIs(prompt, controller.ensure_prompt_segmentation_worker())
        self.assertIsNotNone(fiber)
        self.assertIsNotNone(reference)

        controller.shutdown_all(document_ids=["doc-1"], commit_document_ids=["doc-1"])

        self.assertIsNone(controller.worker("prompt_segmentation"))
        self.assertIsNone(controller.worker("fiber_quick_geometry"))
        self.assertIsNone(controller.worker("reference_instance"))


@unittest.skipUnless(QT_CONTROLLER_AVAILABLE, "requires PySide6")
class PreviewAnalysisTaskControllerTests(unittest.TestCase):
    def test_preview_analysis_start_frame_failed_and_cancel_flow(self) -> None:
        host = _PreviewHost()
        manager = ThreadTaskManager(parent=_app())
        controller = PreviewAnalysisTaskController(host, manager, parent=_app())

        controller.start_session("focus_stack")
        _spin_until(lambda: bool(host.frame_requests))
        controller.on_frame_failed(host.frame_requests[-1], "frame failed")
        controller.cancel(message="cancelled")

        self.assertEqual(controller.mode, "none")
        self.assertTrue(host.dialog.shown)
        self.assertTrue(host.dialog.closed)
        self.assertIn("frame failed", host.dialog.statuses)
        self.assertIn("cancelled", host.status_messages[-1])

    def test_preview_analysis_frame_ready_and_finalize_use_worker_signals(self) -> None:
        host = _PreviewHost()
        manager = ThreadTaskManager(parent=_app())
        controller = PreviewAnalysisTaskController(host, manager, parent=_app())
        frame_signal = _FakeSignal()
        finalize_signal = _FakeSignal()
        worker = type(
            "Worker",
            (),
            {
                "frameSubmitted": frame_signal,
                "finalizeRequested": finalize_signal,
            },
        )()
        controller.mode = "focus_stack"
        controller.dialog = host.dialog
        controller.worker = worker
        controller.request_id = 3
        controller.request_pending = True

        image = QImage(16, 12, QImage.Format.Format_RGB32)
        image.fill(QColor("#FFFFFF"))
        controller.on_frame_ready(3, image)
        controller.finalize()

        self.assertEqual(len(frame_signal.emitted), 1)
        self.assertEqual(finalize_signal.emitted, ["emit"])
        self.assertTrue(controller.finalizing)
        self.assertEqual(host.dialog.busy, [(True, "正在完成景深合成，请稍候…")])


if __name__ == "__main__":
    unittest.main()
