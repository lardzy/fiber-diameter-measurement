"""Small sidecar publication isolated from both the GUI and registration."""
from collections import OrderedDict
import multiprocessing
from pathlib import Path
from time import monotonic

from PySide6.QtCore import QObject, QTimer


def _copy(source, target):
    from fdm.atomic_io import atomic_copy_file
    atomic_copy_file(source, target)


class _Publisher(QObject):
    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self._pending = OrderedDict()
        self._process = None
        self._active = None
        self._started = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._poll)

    def enqueue(self, source, target):
        self._pending[str(target)] = (str(source), str(target))
        self._timer.start()
        self._poll()

    def _poll(self):
        if self._process is not None:
            timed_out = monotonic() - self._started > 30
            if self._process.is_alive() and not timed_out:
                return
            if timed_out:
                self._process.terminate()
                self._process.join(timeout=.2)
                if self._process.is_alive():
                    self._process.kill()
                    return
            code = self._process.exitcode
            self._process.join(timeout=0)
            self._process.close()
            self._process = None
            if code != 0 or timed_out:
                self._window.statusBar().showMessage(
                    f"原始切片已保存；拼接修复文件尚未发布，已保留本机结果：{self._active[0]}。再次检查可重试发布。", 15000)
        if self._pending:
            _, self._active = self._pending.popitem(last=False)
            self._process = multiprocessing.get_context("spawn").Process(target=_copy, args=self._active, daemon=True)
            self._started = monotonic()
            try:
                self._process.start()
            except Exception:
                self._process = None
                self._window.statusBar().showMessage(f"拼接修复文件未发布，本机结果：{self._active[0]}", 10000)
        else:
            self._timer.stop()

    def shutdown(self):
        self._timer.stop()
        self._pending.clear()
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=.3)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=.3)
            if not self._process.is_alive():
                self._process.close()
            self._process = None


def publish_sidecar(window, source: Path, target: Path):
    publisher = getattr(window, "_stitch_publisher", None)
    if publisher is None:
        publisher = _Publisher(window)
        window._stitch_publisher = publisher
    publisher.enqueue(source, target)
