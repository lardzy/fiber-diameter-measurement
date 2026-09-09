from __future__ import annotations

from PySide6.QtWidgets import QLayout, QWidget


def detach_widget_from_layout(widget: QWidget) -> bool:
    """Remove a widget's old layout item before transferring the widget.

    Implicit removal during Qt reparenting can leave a live PySide wrapper for
    an already deleted QWidgetItem (observed with PySide 6.10.2). Calling the
    binding's removeWidget explicitly invalidates that wrapper. The widget
    itself stays alive and keeps its parent until the destination adopts it.
    """
    parent = widget.parentWidget()
    layout = parent.layout() if parent is not None else None
    return layout is not None and _remove_widget(layout, widget)


def _remove_widget(layout: QLayout, widget: QWidget) -> bool:
    if layout.indexOf(widget) >= 0:
        layout.removeWidget(widget)
        return True
    # Follow QObject-owned child layouts rather than materializing itemAt()
    # wrappers for unrelated widgets. QMainWindow can delete its internal
    # layout items during restoreState/fullscreen without notifying Shiboken.
    # Nested button rows are QObject children of their containing layout.
    for child in layout.children():
        if isinstance(child, QLayout) and _remove_widget(child, widget):
            return True
    return False
