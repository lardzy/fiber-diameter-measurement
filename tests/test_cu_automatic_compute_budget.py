from __future__ import annotations

import threading
import time
import unittest

from fdm.services.compute_budget import ComputeBudget, ComputeBusyError, HEAVY_COMPUTE_BUDGET


class ComputeBudgetTests(unittest.TestCase):
    def test_fifo_and_cancelled_waiter_release(self):
        budget = ComputeBudget()
        owner = budget.request("area")
        self.assertTrue(owner.try_acquire())
        first = budget.request("automatic")
        second = budget.request("segmentation")
        self.assertFalse(first.try_acquire())
        first.cancel()
        self.assertFalse(second.try_acquire())
        owner.release()
        self.assertTrue(second.try_acquire())
        self.assertEqual(budget.active_kind, "segmentation")
        second.release()
        self.assertIsNone(budget.active_kind)

    def test_background_wait_preserves_cancel_exception(self):
        budget = ComputeBudget()
        owner = budget.request("automatic")
        owner.try_acquire()
        cancelled = threading.Event()
        finished = threading.Event()
        errors = []

        class OriginalCancellation(Exception):
            pass

        def checkpoint():
            if cancelled.is_set():
                raise OriginalCancellation("same exception")

        def waiting():
            try:
                with budget.slot("segmentation", checkpoint):
                    self.fail("cancelled work entered compute slot")
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=waiting)
        thread.start()
        cancelled.set()
        self.assertTrue(finished.wait(1))
        thread.join()
        self.assertIsInstance(errors[0], OriginalCancellation)
        owner.release()
        with budget.slot("next"):
            self.assertEqual(budget.active_kind, "next")

    def test_main_thread_never_waits_for_busy_permit(self):
        budget = ComputeBudget()
        owner = budget.request("area")
        owner.try_acquire()
        started = time.monotonic()
        with self.assertRaises(ComputeBusyError):
            with budget.slot("segmentation"):
                pass
        self.assertLess(time.monotonic() - started, 0.1)
        owner.release()
        self.assertEqual(len(budget._queue), 0)

    def test_nested_service_calls_reuse_permit(self):
        budget = ComputeBudget()
        with budget.slot("outer"):
            with budget.slot("inner"):
                self.assertEqual(budget.active_kind, "outer")
        self.assertIsNone(budget.active_kind)

    def test_actual_automatic_waits_without_starting_child(self):
        import numpy as np
        from fdm.services.cu_automatic_worker import CuAutomaticWorker
        owner = HEAVY_COMPUTE_BUDGET.request("segmentation")
        self.assertTrue(owner.try_acquire())
        try:
            with CuAutomaticWorker() as worker:
                image = np.full((64, 64), 220, np.uint8)
                identifier = worker.submit(image)
                self.assertEqual(worker.state, "queued")
                self.assertIsNone(worker._process)
                self.assertEqual(worker.poll(), [])
                owner.release()
                end = time.monotonic() + 10
                response = []
                while time.monotonic() < end and not response:
                    response = worker.poll()
                    time.sleep(0.01)
                self.assertEqual(response[0]["task_id"], identifier)
                self.assertEqual(response[0]["result"]["status"], "empty")
        finally:
            owner.release()
