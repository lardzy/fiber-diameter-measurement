from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication

from fdm.ui import icons
from fdm.ui.dialogs import ShortcutHelpDialog


class FullscreenUiAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_fullscreen_and_navigator_icons_have_distinct_fallbacks(self) -> None:
        names = ("fullscreen", "exit_fullscreen", "navigator")
        self.assertTrue(all(name in icons.QT_AWESOME_NAMES for name in names))

        with patch.object(icons, "qta", None):
            rendered = {
                name: icons.themed_icon(name, color="#2A9D8F", size=24).pixmap(24, 24)
                for name in names
            }

        self.assertTrue(all(not pixmap.isNull() for pixmap in rendered.values()))
        fingerprints = {
            name: tuple(
                pixmap.toImage().pixel(x, y)
                for y in range(pixmap.height())
                for x in range(pixmap.width())
            )
            for name, pixmap in rendered.items()
        }
        self.assertEqual(len(set(fingerprints.values())), len(names))

    def test_shortcut_help_explains_fullscreen_and_escape_precedence(self) -> None:
        dialog = ShortcutHelpDialog()
        self.addCleanup(dialog.close)
        text = dialog._content.toPlainText()  # noqa: SLF001

        self.assertIn("F11  进入或退出全屏测量", text)
        self.assertIn("Esc  优先取消当前绘制", text)
        self.assertIn("优先于退出全屏", text)


if __name__ == "__main__":
    unittest.main()
