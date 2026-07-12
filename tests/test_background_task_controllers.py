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
    from fdm.ui.area_inference_worker import AreaBatchInferenceWorker, AreaInferenceRequest
    from fdm.ui.background_task_controller import (
        AreaInferenceBatchState,
        BackgroundTaskController,
        BatchLoadState,
    )
    from fdm.ui.image_loader import ImageLoadRequest
    from fdm.ui.preview_analysis_task_controller import PreviewAnalysisTaskController
    from fdm.ui.preview_analysis_worker import MapBuildSessionWorker
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
    AreaBatchInferenceWorker = None
    BackgroundTaskController = None
    AreaInferenceBatchState = None
    BatchLoadState = None
    ImageLoadRequest = None
    PreviewAnalysisTaskController = None
    MapBuildSessionWorker = None
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
        self.unresolved_requests: list[tuple[str, str]] = []

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

    def _register_unresolved_project_document(self, request: ImageLoadRequest, reason: str) -> None:
        document_id = str(getattr(request.document, "id", ""))
        self.unresolved_requests.append((document_id, reason))

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
        self.busy: list[tuple[bool, str, bool]] = []

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

    def set_busy(self, active: bool, text: str, *, allow_cancel: bool = False) -> None:
        self.busy.append((active, text, allow_cancel))


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
    def test_area_progress_reports_real_completed_count_and_timeout(self) -> None:
        host = _BackgroundHost()
        controller = BackgroundTaskController(host, ThreadTaskManager(parent=_app()))
        request = AreaInferenceRequest(
            document_id="area-progress",
            image_path="/tmp/current.png",
            model_name="棉",
            model_file="model.pth",
            request_id="progress-request",
            generation=4,
        )
        controller._area_infer_state = AreaInferenceBatchState(
            total=1,
            pending_requests={request.request_id: request},
            generation=4,
        )
        progress = _FakeProgress()
        controller._area_infer_progress_dialog = progress

        controller._on_area_inference_progress(
            1,
            1,
            request.image_path,
            request.request_id,
            request.generation,
        )

        self.assertEqual(progress.maximum, 1)
        self.assertEqual(progress.value, 0)
        self.assertIn("已完成 0/1", progress.label)
        self.assertIn("最长等待 180 秒", progress.label)

    def test_area_result_requires_matching_request_id_and_generation(self) -> None:
        document = ImageDocument(id="area-current", path="/tmp/current.png", image_size=(20, 10))
        document.initialize_runtime_state()
        host = _BackgroundHost(ProjectState(version="test", documents=[document]))
        controller = BackgroundTaskController(host, ThreadTaskManager(parent=_app()))
        request = AreaInferenceRequest(
            document_id=document.id,
            image_path=document.path,
            model_name="棉",
            model_file="model.pth",
            request_id="current-request",
            generation=9,
        )
        controller._area_infer_state = AreaInferenceBatchState(
            total=1,
            model_name="棉",
            failures=[],
            pending_requests={request.request_id: request},
            generation=9,
        )

        controller._on_area_inference_succeeded(
            document.id,
            ["stale-request"],
            "old-request",
            9,
        )
        controller._on_area_inference_succeeded(
            document.id,
            ["stale-generation"],
            request.request_id,
            8,
        )
        self.assertEqual(host.applied_area, [])

        controller._on_area_inference_succeeded(
            document.id,
            ["current"],
            request.request_id,
            9,
        )
        controller._on_area_inference_succeeded(
            document.id,
            ["duplicate-late"],
            request.request_id,
            9,
        )

        self.assertEqual(host.applied_area, [(document.id, ["current"])])
        self.assertEqual(controller.area_infer_state.completed_count, 1)

    def test_area_worker_cancelled_during_infer_emits_no_success(self) -> None:
        request = AreaInferenceRequest(
            document_id="cancel-area",
            image_path="/tmp/cancel.png",
            model_name="棉",
            model_file="model.pth",
        )
        worker = AreaBatchInferenceWorker([request], settings=AppSettings())
        succeeded: list[object] = []
        finished: list[tuple[bool, int, int, int]] = []
        worker.succeeded.connect(lambda *args: succeeded.append(args))
        worker.finished.connect(lambda *args: finished.append(args))

        class Result:
            instances = ["late"]

        def infer(**_kwargs):
            worker.cancel()
            return Result()

        with patch(
            "fdm.ui.area_inference_worker.AreaInferenceService.infer_image",
            side_effect=infer,
        ):
            worker.run()

        self.assertEqual(succeeded, [])
        self.assertEqual(finished, [(True, 0, 0, 0)])

    def test_cancel_preserves_each_same_path_project_document_by_request_id(self) -> None:
        host = _BackgroundHost()
        controller = BackgroundTaskController(host, ThreadTaskManager(parent=_app()))
        first = ImageDocument(id="same-source-a", path="/tmp/shared.png", image_size=(20, 10))
        second = ImageDocument(id="same-source-b", path="/tmp/shared.png", image_size=(20, 10))
        first.initialize_runtime_state()
        second.initialize_runtime_state()
        first_request = ImageLoadRequest(path=first.path, document=first, generation=7)
        second_request = ImageLoadRequest(path=second.path, document=second, generation=7)
        controller._load_state = BatchLoadState(  # noqa: SLF001
            context_label="打开项目",
            total=2,
            pending_requests={
                first_request.request_id: first_request,
                second_request.request_id: second_request,
            },
            generation=7,
        )
        image = QImage(20, 10, QImage.Format.Format_RGB32)
        image.fill(QColor("#FFFFFF"))

        controller._on_batch_load_loaded(first_request, image, generation=7)  # noqa: SLF001
        controller._on_batch_load_finished(True, 1, 0, 0, generation=7)  # noqa: SLF001

        self.assertEqual(host.loaded, [first.path])
        self.assertEqual(host.unresolved_requests, [(second.id, "加载已取消")])

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

        _spin_until(lambda: controller.worker("prompt_segmentation") is None)
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
        self.assertEqual(host.dialog.busy, [(True, "正在完成景深合成，请稍候…", True)])

    def test_preview_analysis_waits_for_worker_ack_before_requesting_next_frame(self) -> None:
        host = _PreviewHost()
        manager = ThreadTaskManager(parent=_app())
        controller = PreviewAnalysisTaskController(host, manager, parent=_app())
        submitted = _FakeSignal()
        worker = type(
            "Worker",
            (),
            {"frameSubmittedWithId": submitted},
        )()
        controller.mode = "focus_stack"
        controller.dialog = host.dialog
        controller.worker = worker
        controller.request_id = 4
        controller.request_pending = True

        image = QImage(16, 12, QImage.Format.Format_RGB32)
        image.fill(QColor("#FFFFFF"))
        controller.on_frame_ready(4, image)
        controller.request_frame()

        self.assertTrue(controller.analysis_pending)
        self.assertEqual(len(submitted.emitted), 1)
        self.assertEqual(host.frame_requests, [])

        controller.on_frame_processed(4)
        controller.request_frame()

        self.assertFalse(controller.analysis_pending)
        self.assertEqual(host.frame_requests, [5])

    def test_preview_analysis_watchdog_releases_stuck_capture_request(self) -> None:
        host = _PreviewHost()
        manager = ThreadTaskManager(parent=_app())
        controller = PreviewAnalysisTaskController(
            host,
            manager,
            parent=_app(),
            frame_watchdog_ms=10,
        )
        controller.mode = "focus_stack"
        controller.dialog = host.dialog
        controller.worker = object()

        controller.request_frame()
        _spin_until(lambda: not controller.request_pending)

        self.assertFalse(controller.request_pending)
        self.assertIn("超时", host.status_messages[-1])

    def test_preview_analysis_cancel_while_finalizing_calls_worker_cancel(self) -> None:
        host = _PreviewHost()
        manager = ThreadTaskManager(parent=_app())
        controller = PreviewAnalysisTaskController(host, manager, parent=_app())

        class Worker:
            def __init__(self) -> None:
                self.cancelled = 0

            def cancel(self) -> None:
                self.cancelled += 1

        worker = Worker()
        controller.mode = "map_build"
        controller.dialog = host.dialog
        controller.worker = worker
        controller.finalizing = True

        controller.cancel(message="cancelled")

        self.assertEqual(worker.cancelled, 1)
        self.assertEqual(controller.mode, "none")
        self.assertFalse(controller.finalizing)
        self.assertTrue(host.dialog.closed)
        self.assertIn("cancelled", host.status_messages[-1])

    def test_map_build_worker_suppresses_finished_when_cancelled_during_finalize(self) -> None:
        worker_holder: dict[str, object] = {}

        class FakeAnalyzer:
            def __init__(self, *, device_id: str, device_name: str) -> None:
                del device_id, device_name

            def finalize(self) -> object:
                worker = worker_holder["worker"]
                worker.cancel()
                return object()

        with patch("fdm.ui.preview_analysis_worker.MapBuildAnalyzer", FakeAnalyzer):
            worker = MapBuildSessionWorker(device_id="microview:0", device_name="Microview #1")
            worker_holder["worker"] = worker
            finished: list[object] = []
            worker.finished.connect(lambda payload: finished.append(payload))

            worker.finalize()

        self.assertEqual(finished, [])


if __name__ == "__main__":
    unittest.main()
