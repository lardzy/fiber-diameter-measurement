import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from threading import Event
from PySide6.QtWidgets import QApplication
from fdm.ui.measurement_commit_queue import MeasurementCommitQueue


def test_confirmations_keep_order_and_undo_cancels_only_the_latest():
    app = QApplication.instance() or QApplication([])
    queue = MeasurementCommitQueue()
    gate = Event()
    document = object()
    applied = []

    def first():
        gate.wait(2)
        return 1

    queue.submit(document, first, applied.append)
    queue.submit(document, lambda: 2, applied.append)
    queue.submit(document, lambda: 3, applied.append)
    assert queue.cancel_last(document)
    assert queue.pending(document)
    gate.set()
    queue.flush()
    assert applied == [1, 2]
    assert not queue.pending()
    queue.deleteLater()
    app.processEvents()
