"""Wait for the public page lifecycle without making production navigation synchronous."""
import time
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication


def wait_workbench_ready(hub, timeout=3):
    deadline = time.perf_counter() + timeout
    while hub._page_load_state == 'loading':
        QApplication.processEvents()
        if time.perf_counter() > deadline:
            raise AssertionError('Trial page did not finish preparing')
        QTest.qWait(1)
    assert hub._page_load_state == 'ready', hub.loading_label.text()
    QApplication.processEvents()
    return hub._current_trial()
