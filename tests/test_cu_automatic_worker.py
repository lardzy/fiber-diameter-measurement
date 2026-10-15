from __future__ import annotations

import time
import unittest
from multiprocessing import shared_memory

import numpy as np

from fdm.services.cu_automatic_worker import CuAutomaticWorker


def frame():
    image = np.full((240, 320, 3), 220, np.uint8)
    image[116:124, 40:280] = 40
    return image


def wait_result(worker, timeout=12):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        responses = worker.poll()
        if responses:
            return responses[0]
        time.sleep(0.01)
    raise AssertionError("worker did not return a real result")


class CuAutomaticWorkerTests(unittest.TestCase):
    def test_real_spawn_reuses_process_and_releases_shared_frame(self):
        with CuAutomaticWorker() as worker:
            for algorithm in ["traditional", "multiscale"]:
                identifier = worker.submit(frame(), algorithm)
                name = worker._segment.name
                end = time.monotonic() + 3
                while worker.state == "queued" and time.monotonic() < end:
                    time.sleep(0.01)
                pid = worker._process.pid
                response = wait_result(worker)
                self.assertEqual(response["task_id"], identifier)
                self.assertIsNone(response["error"])
                self.assertEqual(response["result"]["status"], "ok")
                self.assertGreater(response["result"]["candidates"][0]["width_px"], 6)
                with self.assertRaises(FileNotFoundError):
                    shared_memory.SharedMemory(name=name)
                if algorithm == "traditional":
                    original_pid = pid
                else:
                    self.assertEqual(original_pid, pid)

    def test_busy_cancel_and_recovery(self):
        with CuAutomaticWorker() as worker:
            identifier = worker.submit(frame())
            with self.assertRaisesRegex(RuntimeError, "automatic_busy"):
                worker.submit(frame())
            self.assertFalse(worker.cancel("other"))
            self.assertTrue(worker.cancel(identifier))
            self.assertEqual(wait_result(worker)["result"]["status"], "cancelled")
            worker.submit(frame())
            self.assertEqual(wait_result(worker)["result"]["status"], "ok")

    def test_watchdog_kills_without_polling_and_recovers(self):
        with CuAutomaticWorker(timeout_seconds=0.001) as worker:
            worker.submit(frame())
            time.sleep(0.15)
            self.assertFalse(worker.busy)
            self.assertEqual(worker.poll()[0]["result"]["status"], "timeout")
            worker._timeout = 30.0
            worker.submit(frame())
            self.assertEqual(wait_result(worker)["result"]["status"], "ok")

    def test_crash_is_explicit_and_next_job_rebuilds(self):
        with CuAutomaticWorker() as worker:
            worker.submit(frame())
            end = time.monotonic() + 3
            while worker.state == "queued" and time.monotonic() < end:
                time.sleep(0.01)
            worker._process.kill()
            self.assertEqual(wait_result(worker)["result"]["status"], "worker_crashed")
            worker.submit(frame())
            self.assertEqual(wait_result(worker)["result"]["status"], "ok")

    def test_close_is_idempotent_and_prevents_submission(self):
        worker = CuAutomaticWorker()
        worker.close()
        worker.close()
        with self.assertRaises(RuntimeError):
            worker.submit(frame())
