from __future__ import annotations
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/validation'))
sys.path.insert(0, str(ROOT / 'scripts'))
import seventh_round_windows as adapter

OLD = adapter.WINDOWS_ROOT
NEW = r'C:\tmp\fdm-cu-live-explicit-gestures-20261021'
base_adapters = adapter.adapters

def scoped_adapters():
    result = {}
    for name, (blob, detail) in base_adapters().items():
        data = blob.decode('utf-8-sig').replace(OLD, NEW)
        if name == 'scripts/validation/seventh_round_remote.py':
            old = '''"-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
         "$ProgressPreference='SilentlyContinue'; " + code'''
            new = '''"-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand",
         base64.b64encode(("$ProgressPreference='SilentlyContinue'; " + code).encode("utf-16-le")).decode("ascii")'''
            if data.count(old) != 1:
                raise RuntimeError('Expected one short command transport')
            data = data.replace(old, new)
        if name == 'scripts/validation/build_acquisition_seventh_round.ps1':
            old_runtime = 'scripts/validation/validate_acquisition_seventh_round.py --phase quick --scenario quick --platform windows'
            new_runtime = 'scripts/validation/validate_explicit_gestures_runtime.py --phase quick --scenario quick --platform windows --base-device-scale 2'
            if data.count(old_runtime) != 1:
                raise RuntimeError('Expected one pre-package runtime gate')
            data = data.replace(old_runtime, new_runtime)
        blob = (('\ufeff' if name.endswith('.ps1') else '') + data).encode('utf-8')
        detail = dict(detail, sha256=hashlib.sha256(blob).hexdigest(), size=len(blob), scope_override=NEW)
        result[name] = blob, detail
    runtime = Path(__file__).with_name('validate_explicit_gestures_runtime.py')
    blob = runtime.read_bytes()
    source = ROOT / 'scripts/validation/validate_acquisition_seventh_round.py'
    target = 'scripts/validation/validate_explicit_gestures_runtime.py'
    result[target] = (blob, dict(source=source.relative_to(ROOT).as_posix(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), path=target,
        sha256=hashlib.sha256(blob).hexdigest(), size=len(blob),
        adaptation='Round 7 actual-navigation harness plus explicit manual gesture menu and real manual submissions'))
    return result

adapter.WINDOWS_ROOT = NEW
adapter.adapters = scoped_adapters
if __name__ == '__main__':
    raise SystemExit(adapter.main())
