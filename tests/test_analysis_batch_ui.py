from __future__ import annotations

import os
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QRunnable, QThreadPool, Qt
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from fdm.services.advanced_analysis_registry import AdvancedAnalysisInvocation
from fdm.services.advanced_image_analysis import AdvancedAnalysisKind
from fdm.services.analysis_batch import (
    AnalysisBatchItemResult,
    AnalysisBatchProgress,
    AnalysisBatchRequest,
    AnalysisBatchResult,
    AnalysisInvocation,
    AnalysisRecipe,
)
from fdm.services.raster_io import numpy_to_raster_plane
from fdm.ui.analysis_batch_controller import AnalysisBatchController
from fdm.ui.analysis_batch_dialog import AnalysisBatchDialog


def _request() -> AnalysisBatchRequest:
    recipe = AnalysisRecipe(
        "direction-v2",
        "方向性 v2",
        AdvancedAnalysisKind.DIRECTIONALITY,
    )
    invocation = AnalysisInvocation(
        "item-1",
        "图像 1",
        AdvancedAnalysisInvocation(
            AdvancedAnalysisKind.DIRECTIONALITY,
            request_id="item-request",
            generation=1,
            plane=numpy_to_raster_plane(np.zeros((8, 8), dtype=np.uint8)),
        ),
    )
    return AnalysisBatchRequest(
        "batch-ui",
        1,
        recipe,
        (invocation,),
    )


def test_dialog_displays_only_final_batch_item_results() -> None:
    app = QApplication.instance() or QApplication([])
    request = _request()
    dialog = AnalysisBatchDialog()
    try:
        dialog.set_recipes((request.recipe,))
        dialog.set_invocations(request.invocations)
        assert dialog.selected_item_ids() == ("item-1",)
        run_spy = QSignalSpy(dialog.runRequested)
        export_spy = QSignalSpy(dialog.exportRequested)
        assert not dialog.export_button.isEnabled()

        dialog.run_button.click()
        assert run_spy.count() == 1
        dialog.set_busy(True)
        assert dialog.cancel_button.isEnabled()
        dialog.update_progress(
            AnalysisBatchProgress("batch-ui", 1, 1, 1, "item-1")
        )
        assert "整批结束后提交" in dialog.summary_label.text()

        final_result = AnalysisBatchResult(
                "batch-ui",
                1,
                request.recipe.recipe_id,
                (
                    AnalysisBatchItemResult(
                        "item-1",
                        "图像 1",
                        success=False,
                        error_type="ValueError",
                        error_message="测试失败",
                    ),
                ),
        )
        dialog.show_result(final_result)
        app.processEvents()
        assert "失败 1 项" in dialog.summary_label.text()
        assert "测试失败" in dialog.items_table.item(0, 2).text()
        assert dialog.export_button.isEnabled()
        dialog.export_button.click()
        assert export_spy.count() == 1
        assert export_spy.at(0)[0] is final_result

        dialog.items_table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
        assert dialog.selected_item_ids() == ()
        assert not dialog.run_button.isEnabled()
    finally:
        dialog.close()


def test_controller_runs_one_worker_and_emits_final_result() -> None:
    app = QApplication.instance() or QApplication([])
    request = _request()

    def executor(batch_request, token, progress):
        token.raise_if_cancelled()
        progress(
            AnalysisBatchProgress(
                batch_request.request_id,
                batch_request.generation,
                1,
                1,
                "item-1",
            )
        )
        return AnalysisBatchResult(
            batch_request.request_id,
            batch_request.generation,
            batch_request.recipe.recipe_id,
            (
                AnalysisBatchItemResult(
                    "item-1",
                    "图像 1",
                    success=True,
                ),
            ),
        )

    controller = AnalysisBatchController(executor=executor)
    ready_spy = QSignalSpy(controller.batchReady)
    progress_spy = QSignalSpy(controller.progressChanged)

    assert controller.start(request)
    assert not controller.start(request)
    assert controller.wait_for_done()
    for _ in range(10):
        app.processEvents()
        if ready_spy.count():
            break

    assert progress_spy.count() == 1
    assert ready_spy.count() == 1
    assert ready_spy.at(0)[0].request_id == "batch-ui"


def test_controller_wait_does_not_wait_for_unrelated_global_pool_work() -> None:
    app = QApplication.instance() or QApplication([])
    release = Event()
    started = Event()

    class BlockingGlobalTask(QRunnable):
        def run(self) -> None:
            started.set()
            release.wait(5.0)

    global_pool = QThreadPool.globalInstance()
    global_pool.start(BlockingGlobalTask())
    assert started.wait(1.0)

    def executor(batch_request, _token, _progress):
        return AnalysisBatchResult(
            batch_request.request_id,
            batch_request.generation,
            batch_request.recipe.recipe_id,
            (),
        )

    controller = AnalysisBatchController(executor=executor)
    ready_spy = QSignalSpy(controller.batchReady)
    try:
        assert controller._thread_pool is not global_pool
        assert controller._thread_pool.maxThreadCount() == 1
        assert controller.start(_request())
        assert controller.wait_for_done(1_000)
        for _ in range(10):
            app.processEvents()
            if ready_spy.count():
                break
        assert ready_spy.count() == 1
    finally:
        release.set()
        global_pool.waitForDone(5_000)


def test_controller_cancellation_never_emits_batch_ready() -> None:
    app = QApplication.instance() or QApplication([])

    def executor(_batch_request, token, _progress):
        token.raise_if_cancelled()
        token.wait(1.0)
        token.raise_if_cancelled()
        raise AssertionError("cancelled executor must not finish normally")

    controller = AnalysisBatchController(executor=executor)
    cancelled_spy = QSignalSpy(controller.batchCancelled)
    ready_spy = QSignalSpy(controller.batchReady)

    assert controller.start(_request())
    assert controller.cancel()
    assert controller.wait_for_done()
    for _ in range(10):
        app.processEvents()
        if cancelled_spy.count():
            break

    assert cancelled_spy.count() == 1
    assert ready_spy.count() == 0
