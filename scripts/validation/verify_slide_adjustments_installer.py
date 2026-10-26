"""Verify the isolated brightness/exposure installer payload and frozen self-check."""
from __future__ import annotations

import verify_slide_brightness_installer as verifier

# Reuse the exact inventory, camera hash and subprocess self-check contracts.
# Only this task's isolated root and release version differ.
verifier.EXPECTED_ROOT = r"C:\tmp\fdm-slide-adjustments-release-20261026"
verifier.EXPECTED_VERSION = "0.4.12"

if __name__ == "__main__":
    raise SystemExit(verifier.main())
