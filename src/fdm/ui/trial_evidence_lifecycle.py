"""Nonblocking cleanup presentation and leases for asynchronous image readers."""
from PySide6.QtCore import QObject, QTimer, Signal

from fdm.live_experiment.evidence_resources import get_evidence_resources


def install_trial_evidence_shutdown(application):
    if application is None or getattr(application, "_trial_evidence_shutdown_installed", False):
        return
    application._trial_evidence_shutdown_installed = True
    application.aboutToQuit.connect(_stop_trial_evidence_cleanup)


def _stop_trial_evidence_cleanup():
    from fdm.live_experiment import evidence_resources
    resources = evidence_resources._resources
    if resources is not None:
        resources.stop_cleanup()


def submit_evidence_read(executor, callback, *args, path):
    lease = get_evidence_resources().pin_paths([path])
    try:
        resolved = lease.resolve(path)
        if resolved is None:
            from concurrent.futures import Future
            future = Future()
            future.set_result((None, "此临时图片已随试验释放。"))
            lease.close()
            return future
        if str(resolved) != str(path):
            if args and isinstance(args[0], str):
                args = (str(resolved), *args[1:])
            elif args and hasattr(args[0], "path"):
                from dataclasses import replace
                args = (replace(args[0], path=str(resolved)), *args[1:])
        future = executor.submit(callback, *args)
    except BaseException:
        lease.close()
        raise
    future.add_done_callback(lambda _: lease.close())
    return future


class TrialEvidenceCleanup(QObject):
    statusChanged = Signal(str, bool)

    def __init__(self, host):
        super().__init__(host)
        self.future = None
        self._resources = None
        self._baseline = None
        self._shown_completed = -1
        self.last_message = "仅清理无引用的程序临时图片；当前试验及项目图片保留。"
        self._poll = QTimer(self)
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._completed)

    def request(self):
        if self.future is not None:
            return
        self.statusChanged.emit("正在后台清理；仍在使用的图片将保留。", True)
        self._resources = get_evidence_resources()
        self._baseline = self._resources.cleanup_progress()
        self._shown_completed = self._baseline.completed
        self.future = self._resources.request_cleanup()
        self._poll.start()

    def _completed(self):
        if self.future is None:
            return
        try:
            progress = self._resources.cleanup_progress()
            if progress.busy and progress.completed == self._shown_completed:
                return
            from fdm.live_experiment.evidence_resources import CleanupReport
            latest = progress.last_report
            report = CleanupReport(
                deleted_files=progress.deleted_files - self._baseline.deleted_files,
                deleted_bytes=progress.deleted_bytes - self._baseline.deleted_bytes,
                # In-use is an observation, not an additive event count. A
                # pinned image can appear in several deferred passes.
                in_use=latest.in_use,
                skipped=progress.skipped - self._baseline.skipped,
                deferred=progress.busy,
                stopped=latest.stopped,
                errors=list(latest.errors),
            )
            self._shown_completed = progress.completed
            self.last_message = report.message()
            if progress.busy:
                self.statusChanged.emit(self.last_message, True)
                return
        except Exception as exc:
            self.last_message = f"部分临时图片暂未清理，不影响测量：{exc}"
        self.future = self._resources = self._baseline = None
        self._poll.stop()
        self.statusChanged.emit(self.last_message, False)


def trial_evidence_cleanup(host):
    cleanup = getattr(host, "_trial_evidence_cleanup", None)
    if cleanup is None:
        cleanup = TrialEvidenceCleanup(host)
        host._trial_evidence_cleanup = cleanup
    return cleanup
