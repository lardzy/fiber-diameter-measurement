"""Application-owned trial workbooks and explicit final synchronization barriers."""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout


def live_excel_manager(host):
    manager = getattr(host, "_trial_live_excel", None)
    if manager is None or getattr(manager, "_closed", False):
        from fdm.ui.live_excel import LiveExcelManager
        manager = host._trial_live_excel = LiveExcelManager(parent=host)
        host._trial_excel_sessions = {}
    return manager


def ensure_live_excel(host, session):
    manager = live_excel_manager(host)
    host._trial_excel_sessions[session.id] = session
    manager.ensure(session)


def _wait(parent, message, attempt, *, retry=None, force=None, cancellable=True):
    """Keep Qt, collection and COM replies running while the operator decides."""
    if attempt():
        return True
    class WaitDialog(QDialog):
        def reject(self):
            if cancellable:
                super().reject()
        def closeEvent(self, event):
            if cancellable:
                super().closeEvent(event)
            else:
                event.ignore()
    dialog = WaitDialog(parent)
    dialog.setWindowTitle("Excel 最终同步")
    dialog.setMinimumWidth(430)
    layout = QVBoxLayout(dialog)
    label = QLabel(dialog)
    label.setWordWrap(True)
    layout.addWidget(label)
    buttons = QDialogButtonBox(parent=dialog)
    if cancellable:
        buttons.setStandardButtons(QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消，保留当前试验")
        buttons.rejected.connect(dialog.reject)
    if retry is not None:
        button = buttons.addButton("重试同步", QDialogButtonBox.ButtonRole.ActionRole)
        button.clicked.connect(retry)
    if force is not None:
        button = buttons.addButton("结束关联（未同步数据仍在 FDM）", QDialogButtonBox.ButtonRole.ActionRole)
        button.clicked.connect(lambda: (force(), dialog.accept()))
    layout.addWidget(buttons)
    timer = QTimer(dialog)
    timer.setInterval(80)
    def poll():
        label.setText(message())
        if attempt():
            dialog.accept()
    timer.timeout.connect(poll)
    label.setText(message())
    timer.start()
    try:
        return dialog.exec() == QDialog.DialogCode.Accepted
    finally:
        timer.stop()


def prepare_trial_excel(host, session, *, parent=None):
    manager = getattr(host, "_trial_live_excel", None)
    if manager is None or session.id not in getattr(host, "_trial_excel_sessions", {}):
        return True
    status = manager.status(session.id)
    # Missing Excel/template has no associated workbook to finalize. A book
    # closed explicitly by its user is likewise not reopened at shutdown.
    if (status.get("state") in {"stopped", "detached"}
            or (not status.get("ever_connected") and status.get("state") in {"error", "unsupported"})):
        host._trial_excel_force = getattr(host, "_trial_excel_force", {})
        host._trial_excel_force[session.id] = session.revision
        return True
    host._trial_excel_force = getattr(host, "_trial_excel_force", {})
    if host._trial_excel_force.get(session.id) == session.revision:
        return True
    accepted = _wait(parent or host,
        lambda: "正在将本试验的最终测量数据写入 Excel；报表计算和文件保存由 Excel 管理。\n\n" + manager.status(session.id).get("message", ""),
        lambda: manager.flush(session), retry=lambda: manager.retry(session),
        force=lambda: host._trial_excel_force.__setitem__(session.id, session.revision))
    if not accepted:
        host._trial_excel_force.pop(session.id, None)
    return accepted


def detach_trial_excel(host, session, *, parent=None):
    manager = getattr(host, "_trial_live_excel", None)
    if manager is None or session.id not in getattr(host, "_trial_excel_sessions", {}):
        return True
    force = getattr(host, "_trial_excel_force", {}).get(session.id) == session.revision
    if not _wait(parent or host,
        lambda: "正在解除 Excel 关联。工作簿将留在 Excel 中供您保存。\n\n" + manager.status(session.id).get("message", ""),
        lambda: manager.detach(session, force=force), cancellable=False):
        return False
    host._trial_excel_sessions.pop(session.id, None)
    getattr(host, "_trial_excel_force", {}).pop(session.id, None)
    return True


def prepare_all_trial_excel(host):
    accepted = all(prepare_trial_excel(host, session) for session in
                   tuple(getattr(host, "_trial_excel_sessions", {}).values()))
    if not accepted:
        host._trial_excel_force = {}
    return accepted


def detach_all_trial_excel(host):
    return all(detach_trial_excel(host, session) for session in
               tuple(getattr(host, "_trial_excel_sessions", {}).values()))


def close_trial_excel_service(host):
    manager = getattr(host, "_trial_live_excel", None)
    return manager is None or _wait(host, lambda: "正在释放 Excel 同步进程；已打开的工作簿继续保留。\n"
        + getattr(manager, "close_message", ""), manager.close, cancellable=False)
