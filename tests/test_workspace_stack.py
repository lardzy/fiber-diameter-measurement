from PySide6.QtCore import QCoreApplication, QEvent, QSize
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from fdm.ui.workspace_stack import WorkspaceStack


def settle():
    for _ in range(4):
        QApplication.processEvents()


def dispose(window):
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_retained_large_page_cannot_enlarge_unrelated_workspace(desktop_application):
    window = QWidget()
    layout = QVBoxLayout(window)
    outer = WorkspaceStack()
    inner = WorkspaceStack()
    layout.addWidget(outer)
    outer.addWidget(inner)
    current = QWidget()
    current.setMinimumSize(200, 100)
    inner.addWidget(current)
    # Models a retained SDK display whose physical resolution is fixed.
    native_page = QWidget()
    native_page.setFixedSize(2400, 1800)
    inner.addWidget(native_page)
    window.resize(640, 480)
    window.show()
    settle()
    try:
        assert window.size() == QSize(640, 480)
        assert inner.minimumSizeHint() == QSize(200, 100)
        assert outer.minimumSizeHint() == QSize(200, 100)
        native_page.setFixedSize(3840, 2160)
        settle()
        assert window.size() == QSize(640, 480)
        # Visible content still imposes its real constraints: the stack must
        # not mask a genuine inaccessible page by always returning zero.
        current.setMinimumHeight(520)
        settle()
        assert current.height() >= 520
        assert window.height() >= 520
        current.setMinimumHeight(100)
        settle()
        window.resize(640, 480)
        settle()
        assert window.size() == QSize(640, 480)
    finally:
        dispose(window)


def test_replacing_current_page_invalidates_parent_constraints(desktop_application):
    window = QWidget()
    layout = QVBoxLayout(window)
    stack = WorkspaceStack()
    layout.addWidget(stack)
    small = QWidget()
    small.setMinimumSize(200, 100)
    tall = QWidget()
    tall.setMinimumSize(200, 600)
    stack.addWidget(small)
    stack.addWidget(tall)
    window.resize(640, 480)
    window.show()
    settle()
    try:
        stack.setCurrentWidget(tall)
        settle()
        assert tall.height() >= 600
        assert window.height() >= 600
        stack.removeWidget(tall)
        settle()
        window.resize(640, 480)
        settle()
        assert stack.currentWidget() is small
        assert window.size() == QSize(640, 480)
        assert stack.minimumSizeHint() == QSize(200, 100)
    finally:
        dispose(window)


def test_hidden_word_wrap_does_not_change_current_page_height_policy(desktop_application):
    stack = WorkspaceStack()
    fixed_page = QWidget()
    wrapped_page = QLabel("A long status message " * 30)
    wrapped_page.setWordWrap(True)
    stack.addWidget(fixed_page)
    stack.addWidget(wrapped_page)
    try:
        assert not stack.hasHeightForWidth()
        assert stack.heightForWidth(200) == -1
        stack.setCurrentWidget(wrapped_page)
        assert stack.hasHeightForWidth()
        assert stack.heightForWidth(200) > stack.heightForWidth(400)
    finally:
        dispose(stack)
