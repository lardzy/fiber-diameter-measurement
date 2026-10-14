from __future__ import annotations

import sys

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFocusEvent, QHideEvent, QKeyEvent, QKeySequence
from PySide6.QtWidgets import QKeySequenceEdit, QLineEdit, QWidget

from fdm.screenshot_settings import HotkeyBinding


class ScreenshotHotkeyEdit(QKeySequenceEdit):
    """Record one combination, retaining Windows VK identity when available.

    QKeySequenceEdit normally finishes after a timer and can miss PrintScreen
    when Windows delivers only its release. Recording here is explicit; focus
    owns the pause lease, independently of whether a combination was entered.
    """

    recordingChanged = Signal(bool)
    recordingRejected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._recording = False
        self._pressed_key: int | None = None
        self._binding_enabled = False
        self._sequence_text = ""
        self._native_virtual_key: int | None = None
        self._native_modifiers: int | None = None
        self._native_scan_code: int | None = None
        self._native_sequence_text: str | None = None
        self.setMaximumSequenceLength(1)
        self.setClearButtonEnabled(True)
        self.keySequenceChanged.connect(self._sequence_changed)
        # The internal line edit can receive events instead of the composite
        # widget, depending on Qt version and focus routing.
        self._line_edit = self.findChild(QLineEdit)
        if self._line_edit is not None:
            self._line_edit.installEventFilter(self)

    @property
    def is_recording(self) -> bool:
        return self._recording

    def set_binding(self, binding: HotkeyBinding) -> None:
        normalized = binding.normalized()
        self.setKeySequence(QKeySequence(normalized.sequence))
        self._sequence_text = self.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
        self._binding_enabled = bool(normalized.enabled and self._sequence_text)
        self._native_virtual_key = normalized.native_virtual_key
        self._native_modifiers = normalized.native_modifiers
        self._native_scan_code = None
        self._native_sequence_text = (
            self.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
            if normalized.native_virtual_key is not None else None
        )

    def binding(self) -> HotkeyBinding:
        sequence = self.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
        if sequence != self._sequence_text:
            self._sequence_changed(self.keySequence())
        if self._native_virtual_key is not None and sequence != self._native_sequence_text:
            self._clear_native_identity()
        return HotkeyBinding(
            sequence=sequence,
            enabled=bool(self._binding_enabled and sequence),
            native_virtual_key=self._native_virtual_key,
            native_modifiers=self._native_modifiers,
        ).normalized()

    def _sequence_changed(self, sequence: QKeySequence) -> None:
        # QKeySequenceEdit emits this signal again from focusOutEvent even if
        # the text did not change. A completed edit must retain its Windows
        # identity; only an actual text change invalidates that identity.
        text = sequence.toString(QKeySequence.SequenceFormat.PortableText)
        if text != self._sequence_text:
            self._sequence_text = text
            self._binding_enabled = bool(text)
        if text != self._native_sequence_text:
            self._clear_native_identity()

    def _clear_native_identity(self) -> None:
        self._native_virtual_key = None
        self._native_modifiers = None
        self._native_scan_code = None
        self._native_sequence_text = None

    def _set_recording(self, active: bool) -> None:
        if self._recording != active:
            self._recording = active
            self._pressed_key = None
            self.recordingChanged.emit(active)

    def focusInEvent(self, event: QFocusEvent) -> None:
        self._set_recording(True)
        super().focusInEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        self._set_recording(False)
        super().focusOutEvent(event)

    def hideEvent(self, event: QHideEvent) -> None:
        self._set_recording(False)
        super().hideEvent(event)

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if watched is getattr(self, "_line_edit", None):
            if event.type() == QEvent.Type.KeyPress:
                self.keyPressEvent(event)
                return event.isAccepted()
            if event.type() == QEvent.Type.KeyRelease:
                self.keyReleaseEvent(event)
                return event.isAccepted()
            if event.type() == QEvent.Type.ShortcutOverride:
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.isAutoRepeat():
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab) and not (
            event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)
        ):
            self.focusNextPrevChild(event.key() != Qt.Key.Key_Backtab)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self.clearFocus()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Backspace and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self.clear()
            event.accept()
            return
        if self._record_key(event):
            self._pressed_key = self._canonical_key(event.key())
        event.accept()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        key = self._canonical_key(event.key())
        # Windows/Qt can deliver PrintScreen only on release. If a press was
        # seen, retain its modifier snapshot even if modifiers are released first.
        if key == Qt.Key.Key_Print and self._pressed_key != key and not event.isAutoRepeat():
            self._record_key(event)
        if self._pressed_key == key:
            self._pressed_key = None
        event.accept()

    @staticmethod
    def _canonical_key(key: int) -> int:
        return int(Qt.Key.Key_Print) if key == Qt.Key.Key_SysReq else key

    def _record_key(self, event: QKeyEvent) -> bool:
        key = self._canonical_key(event.key())
        if key in {
            0, Qt.Key.Key_unknown, Qt.Key.Key_Control, Qt.Key.Key_Shift,
            Qt.Key.Key_Alt, Qt.Key.Key_Meta, Qt.Key.Key_AltGr,
        }:
            return False
        if event.modifiers() & Qt.KeyboardModifier.GroupSwitchModifier:
            self.recordingRejected.emit("AltGr 字符不作为全局组合键录入，请使用 Ctrl、Alt 或 Shift 配合其它键。")
            return False
        modifiers = event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.MetaModifier
            | Qt.KeyboardModifier.KeypadModifier
        )
        # Qt does not emit keySequenceChanged when a freshly recorded key has
        # the same display text. Its native identity can nevertheless differ
        # after a layout change, or be unavailable on another platform.
        self._clear_native_identity()
        self.setKeySequence(QKeySequence(modifiers.value | key))
        # An actual key entry explicitly configures the binding, even when it
        # has the same text as a previously disabled persisted combination.
        self._sequence_text = self.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
        self._binding_enabled = bool(self._sequence_text)
        # nativeVirtualKey is platform-specific. Cocoa/X11 values must never
        # be serialized as Windows virtual keys.
        if sys.platform == "win32" and event.nativeVirtualKey():
            native_modifiers = 0
            for qt_modifier, win_modifier in (
                (Qt.KeyboardModifier.AltModifier, 0x01),
                (Qt.KeyboardModifier.ControlModifier, 0x02),
                (Qt.KeyboardModifier.ShiftModifier, 0x04),
                (Qt.KeyboardModifier.MetaModifier, 0x08),
            ):
                if modifiers & qt_modifier:
                    native_modifiers |= win_modifier
            candidate = HotkeyBinding(
                self.keySequence().toString(QKeySequence.SequenceFormat.PortableText),
                True, int(event.nativeVirtualKey()), native_modifiers,
            ).normalized()
            self._native_virtual_key = candidate.native_virtual_key
            self._native_modifiers = candidate.native_modifiers
            self._native_scan_code = int(event.nativeScanCode())
            self._native_sequence_text = (
                self.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
                if candidate.native_virtual_key is not None else None
            )
        self.editingFinished.emit()
        return True
