from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtWidgets import QApplication

from fdm.image_processing_models import (
    ImageOperationSpec,
    ImageProcessingRecipe,
)
from fdm.services.image_batch import (
    BatchProgressPhase,
    BatchRasterInput,
    execute_batch_recipe,
)
from fdm.services.raster_io import numpy_to_raster_plane
from fdm.ui.image_batch_controller import ImageBatchTaskController


_PLENTY_OF_DISK = 10 << 30


def _recipe() -> ImageProcessingRecipe:
    return ImageProcessingRecipe.from_operations(
        (ImageOperationSpec("mean_filter", {"radius": 1}),)
    )


def _inputs(document_id: str = "doc") -> tuple[BatchRasterInput, ...]:
    return (
        BatchRasterInput(
            document_id=document_id,
            display_name=f"图片 {document_id}",
            raster=numpy_to_raster_plane(
                np.arange(256, dtype=np.uint8).reshape(16, 16)
            ),
        ),
    )


class ImageBatchTaskControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _wait_until(self, predicate, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        self.fail("等待异步图像批处理条件超时")

    def test_controller_emits_ordered_progress_and_real_result(self) -> None:
        controller = ImageBatchTaskController()
        progress = []
        ready = []
        busy = []
        controller.progressChanged.connect(progress.append)
        controller.batchReady.connect(ready.append)
        controller.busyChanged.connect(busy.append)
        try:
            request = controller.start(
                recipe=_recipe(),
                inputs=_inputs(),
                available_disk_bytes=_PLENTY_OF_DISK,
            )
            self._wait_until(lambda: bool(ready))

            self.assertEqual(ready[0].request_id, request.request_id)
            self.assertEqual(ready[0].generation, request.generation)
            self.assertEqual(
                [update.phase for update in progress],
                [
                    BatchProgressPhase.PREFLIGHT,
                    BatchProgressPhase.PROCESSING,
                    BatchProgressPhase.PROCESSING,
                    BatchProgressPhase.PACKAGING,
                ],
            )
            self.assertEqual(busy, [True, False])
            self.assertFalse(controller.is_busy())
        finally:
            controller.close()
            controller.wait_for_done()

    def test_latest_request_cancels_old_and_never_runs_two_workers(self) -> None:
        first_started = threading.Event()
        release_first = threading.Event()
        active = 0
        maximum_active = 0
        lock = threading.Lock()

        def executor(request, token, progress, limits):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                if request.generation == 1:
                    first_started.set()
                    release_first.wait(2.0)
                token.raise_if_cancelled()
                return execute_batch_recipe(
                    request,
                    cancellation_token=token,
                    progress_callback=progress,
                    limits=limits,
                )
            finally:
                with lock:
                    active -= 1

        controller = ImageBatchTaskController(executor=executor)
        ready = []
        progress = []
        busy = []
        controller.batchReady.connect(ready.append)
        controller.progressChanged.connect(progress.append)
        controller.busyChanged.connect(busy.append)
        try:
            first = controller.start(
                recipe=_recipe(),
                inputs=_inputs("first"),
                available_disk_bytes=_PLENTY_OF_DISK,
            )
            self.assertTrue(first_started.wait(1.0))
            second = controller.start(
                recipe=_recipe(),
                inputs=_inputs("second"),
                available_disk_bytes=_PLENTY_OF_DISK,
            )
            release_first.set()
            self._wait_until(lambda: len(ready) == 1)

            self.assertNotEqual(first.request_id, second.request_id)
            self.assertEqual(ready[0].request_id, second.request_id)
            self.assertEqual(maximum_active, 1)
            self.assertTrue(
                all(
                    update.request_id == second.request_id
                    for update in progress
                )
            )
            self.assertEqual(busy, [True, False])
        finally:
            release_first.set()
            controller.close()
            controller.wait_for_done()

    def test_cancelled_task_never_emits_success(self) -> None:
        started = threading.Event()

        def executor(request, token, _progress, _limits):
            started.set()
            while not token.is_cancelled:
                time.sleep(0.002)
            token.raise_if_cancelled()
            raise AssertionError("取消后不得继续")  # pragma: no cover

        controller = ImageBatchTaskController(executor=executor)
        ready = []
        cancelled = []
        controller.batchReady.connect(ready.append)
        controller.taskCancelled.connect(cancelled.append)
        try:
            request = controller.start(
                recipe=_recipe(),
                inputs=_inputs(),
                available_disk_bytes=_PLENTY_OF_DISK,
            )
            self.assertTrue(started.wait(1.0))
            controller.cancel()
            self._wait_until(lambda: not controller.is_busy())

            self.assertEqual(ready, [])
            self.assertEqual(cancelled, [request.request_id])
        finally:
            controller.close()
            controller.wait_for_done()

    def test_mismatched_result_identity_is_reported_as_failure(self) -> None:
        def executor(request, token, progress, limits):
            result = execute_batch_recipe(
                request,
                cancellation_token=token,
                progress_callback=progress,
                limits=limits,
            )
            return replace(result, generation=request.generation + 1)

        controller = ImageBatchTaskController(executor=executor)
        ready = []
        failures = []
        controller.batchReady.connect(ready.append)
        controller.taskFailed.connect(
            lambda request_id, message: failures.append(
                (request_id, message)
            )
        )
        try:
            request = controller.start(
                recipe=_recipe(),
                inputs=_inputs(),
                available_disk_bytes=_PLENTY_OF_DISK,
            )
            self._wait_until(lambda: bool(failures))

            self.assertEqual(ready, [])
            self.assertEqual(failures[0][0], request.request_id)
            self.assertIn("request_id/generation", failures[0][1])
        finally:
            controller.close()
            controller.wait_for_done()


if __name__ == "__main__":
    unittest.main()
