from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.cancellation import CancellationError, CancellationSource, CancellationTokenSource


class CancellationTests(unittest.TestCase):
    def test_source_exposes_read_only_token_and_cancel_is_idempotent(self) -> None:
        source = CancellationTokenSource()
        self.assertIs(CancellationSource, CancellationTokenSource)

        self.assertFalse(source.token.is_cancelled)
        self.assertFalse(source.token.wait(0))
        self.assertTrue(source.cancel())
        self.assertTrue(source.token.is_cancelled)
        self.assertTrue(source.token.wait(0))
        self.assertFalse(source.cancel())

    def test_token_raises_after_cancellation(self) -> None:
        source = CancellationSource()
        source.token.raise_if_cancelled()

        source.cancel()

        with self.assertRaises(CancellationError):
            source.token.raise_if_cancelled()


if __name__ == "__main__":
    unittest.main()
