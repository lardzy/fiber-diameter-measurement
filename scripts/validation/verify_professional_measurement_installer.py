"""Verify the isolated professional-measurement installer and its actual EXE."""
from __future__ import annotations

import verify_slide_brightness_installer as verifier

verifier.EXPECTED_ROOT = r"C:\tmp\fdm-professional-measurement-release-20261027"
verifier.EXPECTED_VERSION = "0.4.13"

if __name__ == "__main__":
    raise SystemExit(verifier.main())
