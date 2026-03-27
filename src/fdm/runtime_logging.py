from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import sys


def runtime_log_path() -> Path:
    local_app_data = Path.home()
    if sys.platform.startswith("win"):
        app_data = Path(
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or str(Path.home())
        )
        local_app_data = app_data
    log_dir = local_app_data / "FiberDiameterMeasurement" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "startup.log"


def append_runtime_log(title: str, details: str = "") -> None:
    try:
        log_path = runtime_log_path()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {title}\n")
            if details:
                handle.write(details.rstrip())
                handle.write("\n")
            handle.write("\n")
    except OSError:
        return

