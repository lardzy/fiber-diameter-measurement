"""Verify the isolated professional-precision installer and its actual EXE."""
from __future__ import annotations

import verify_slide_brightness_installer as verifier

verifier.EXPECTED_ROOT = r"C:\tmp\fdm-professional-precision-release-20261027"
verifier.EXPECTED_VERSION = "0.4.14"

if __name__ == "__main__":
    raise SystemExit(verifier.main())
