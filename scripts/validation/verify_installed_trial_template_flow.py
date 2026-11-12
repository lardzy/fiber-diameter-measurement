"""Source convenience wrapper; use the EXE flag for actual installed acceptance.

Source: python scripts/validation/verify_installed_trial_template_flow.py --source-template SOURCE --output NEW_DIRECTORY
Installed: FiberDiameterMeasurement.exe --verify-trial-template-flow --require-installed --source-template SOURCE --output NEW_DIRECTORY
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fdm.validation.trial_template_flow import main


if __name__ == "__main__":
    raise SystemExit(main())
