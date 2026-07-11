from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from PySide6.QtCore import QSize, Qt
    from PySide6.QtGui import QAction, QFont, QIcon, QPixmap
    from PySide6.QtWidgets import QApplication, QMenu, QToolButton, QWidget

    from fdm.ui.widgets import OverlayToolSplitButton, ToolStripActionButton

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is required for UI tests")
class OverlayToolSplitButtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_uses_qt_standard_split_button_behavior(self) -> None:
        button = OverlayToolSplitButton()
        primary_button = ToolStripActionButton(QAction("选择"))
        try:
            self.assertIsInstance(button, QToolButton)
            self.assertEqual(button.popupMode(), QToolButton.ToolButtonPopupMode.MenuButtonPopup)
            self.assertTrue(button.isCheckable())
            self.assertEqual(button.focusPolicy(), Qt.FocusPolicy.StrongFocus)
            self.assertNotIn("paintEvent", OverlayToolSplitButton.__dict__)
            self.assertNotIn("keyPressEvent", OverlayToolSplitButton.__dict__)
            self.assertTrue(button.property("primaryTool"))
            self.assertTrue(button.property("splitTool"))
            self.assertEqual(button.focusPolicy(), primary_button.focusPolicy())
        finally:
            button.close()
            primary_button.close()

    def test_current_tool_and_accessible_name_follow_visible_selection(self) -> None:
        button = OverlayToolSplitButton()
        menu = QMenu()
        icon = QIcon(QPixmap(QSize(16, 16)))
        try:
            button.setText("连续测量")
            button.setCurrentTool("continuous_manual", icon)
            button.setMenu(menu)

            self.assertEqual(button.text(), "连续测量")
            self.assertEqual(button.currentToolKind(), "continuous_manual")
            self.assertFalse(button.currentToolIcon().isNull())
            self.assertIs(button.menu(), menu)
            self.assertIn("连续测量", button.accessibleName())
            self.assertIn("展开", button.accessibleName())

            button.setText("手动线段")
            self.assertIn("手动线段", button.accessibleName())
            self.assertNotIn("连续测量", button.accessibleName())
        finally:
            button.close()
            menu.close()

    def test_primary_action_and_compact_mode_keep_compatibility_contract(self) -> None:
        parent = QWidget()
        parent_font = QFont(parent.font())
        parent_font.setPointSize(parent_font.pointSize() + 1)
        parent.setFont(parent_font)
        button = OverlayToolSplitButton(parent)
        button.setText("多边形面积")
        triggered: list[str] = []
        button.primaryTriggered.connect(lambda: triggered.append("primary"))
        try:
            button.click()
            self.assertEqual(triggered, ["primary"])
            self.assertEqual(button.font(), parent_font)
            self.assertEqual(button.toolButtonStyle(), Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            expanded_width = button.sizeHint().width()

            button.setCompactMode(True)

            self.assertTrue(button.isCompactMode())
            self.assertTrue(button.property("compactTool"))
            self.assertEqual(button.toolButtonStyle(), Qt.ToolButtonStyle.ToolButtonIconOnly)
            self.assertEqual(button.sizeHint().width(), button.compactWidthHint())
            self.assertLess(button.sizeHint().width(), expanded_width)
            self.assertGreaterEqual(button.menuAreaWidth(), 28)
        finally:
            parent.close()


if __name__ == "__main__":
    unittest.main()
