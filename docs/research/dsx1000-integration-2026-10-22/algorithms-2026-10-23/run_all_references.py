"""Rebuild synthetic checks in isolated temporary copies; no vendor code/device I/O."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SUITES = [
    ("surface", "validate_reference.py", ["reference.py"], "reference-validation.json"),
    ("rendering", "validate_reference.py", ["reference.py"], "reference-validation.json"),
    ("height-contract", "validate_reference.py", ["reference.py"], "reference-validation.json"),
    ("efi-3d", "validate_reference.py", ["reference_efi.py"], "reference-validation/results.json"),
    ("stabilization", "run_validation.py", ["nmv_reference.py", "fft-sizes.json"], "validation/results.json"),
    ("stabilization/hdr", "run_validation.py", ["hamal8_reference.py", "native-tables.json"], "validation/results.json"),
    ("stitching", "run_reference_tests.py", ["reference_prototype.py"], "test-results.json"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy, scipy, cv2
    output = ROOT / "validation-suite"
    output.mkdir(exist_ok=True)
    runs = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="dsx-algorithm-replay-") as temp:
        for folder, script, sources, result_name in SUITES:
            copied = Path(temp) / folder
            copied.mkdir(parents=True)
            identities = {}
            for name in [script] + sources:
                original = ROOT / folder / name
                identities[name] = sha(original)
                shutil.copy2(original, copied / name)
            start = time.monotonic()
            process = subprocess.run([sys.executable, str(copied / script)], cwd=copied,
                                     env=env, capture_output=True, text=True, timeout=300)
            result_path = copied / result_name
            detail = json.loads(result_path.read_text()) if result_path.exists() else {}
            checks = detail.get("checks", detail.get("tests", detail.get("results", [])))
            passed = sum(bool(c.get("passed", c.get("pass", False))) for c in checks)
            name = folder.replace("/", "-")
            (output / (name + ".json")).write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n")
            run = dict(suite=folder, exit_code=process.returncode, checks=len(checks), passed=passed,
                       success=process.returncode == 0 and len(checks) > 0 and passed == len(checks),
                       elapsed_seconds=time.monotonic()-start, source_sha256=identities,
                       stdout=process.stdout, stderr=process.stderr,
                       result_path="validation-suite/" + name + ".json")
            runs.append(run)
            print(f"{folder}: {passed}/{len(checks)}, exit={process.returncode}", flush=True)
    result = dict(schema="dsx-independent-algorithm-replay-v1",
                  utc=datetime.now(timezone.utc).isoformat(),
                  scope="Fresh temporary source copies and generated synthetic fixtures; no vendor DLL or hardware execution",
                  python=platform.python_version(), platform=platform.platform(),
                  dependencies=dict(numpy=numpy.__version__, scipy=scipy.__version__, opencv=cv2.__version__),
                  runner_sha256=sha(Path(__file__)), suites=runs,
                  checks=sum(r["checks"] for r in runs), passed=sum(r["passed"] for r in runs),
                  status="passed" if all(r["success"] for r in runs) else "failed")
    (ROOT / "validation-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"TOTAL {result['passed']}/{result['checks']}: {result['status']}")
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
