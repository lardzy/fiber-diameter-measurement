#!/usr/bin/env python3
"""Run third-round installed/extracted GUI checks in an exclusive checkpoint.

Does not manage camera sharing, VM snapshots, installation, or existing profiles.
The caller provides a checkpoint with host camera sharing temporarily disabled.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import uuid
import run_acquisition_second_round_frozen as shared

ROOT = Path(__file__).resolve().parents[2]
WINDOWS_ROOT = r"C:\tmp\fdm-cu-live-third-round-20261019"
shared.WINDOWS_ROOT = WINDOWS_ROOT
# shared.run_powershell only uses this path for bounded, owned temporary scripts.
# The legacy subdirectory is intentionally not used for this task.
_original_windows_path = shared.windows_path
shared.windows_path = lambda relative: _original_windows_path(relative.replace("build/second-round/", "build/third-round/"))
MARKER = "FDM_THIRD_GUI_RESULT="
PROBE = "scripts/validation/probe_acquisition_third_round_frozen.ps1"


def win(relative: str) -> str:
    return str(PureWindowsPath(WINDOWS_ROOT) / relative)


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def cases():
    result = [dict(name="empty-light-1600x900", theme="light", width=1600, height=900, populated=False)]
    result += [dict(name=f"records-{theme}-1600x900", theme=theme, width=1600, height=900, populated=True)
               for theme in ("light", "dark", "navy", "warm")]
    result += [dict(name=f"records-light-{w}x{h}", theme="light", width=w, height=h, populated=True)
               for w, h in ((1366, 768), (1067, 600))]
    result.append(dict(name="excel-fixed-entry-light-1600x900", theme="light", width=1600, height=900, populated=True, export=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vm", default="Windows 11")
    parser.add_argument("--executable", type=shared.candidate_executable)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/acquisition-third-round-20261019/windows/frozen-gui")
    parser.add_argument("--case", choices=[case["name"] for case in cases()])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ-") + uuid.uuid4().hex[:8]
    selected = [case for case in cases() if args.case is None or case["name"] == args.case]
    remote = win(f"build/third-round/frozen-gui/{run_id}")
    plan = dict(schema=1, run_id=run_id, cases=selected, windows_output=remote, executable_override=args.executable,
                real_hardware_tested=False, camera_sharing_managed=False, installation_performed=False,
                snapshot_management_performed=False)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    local = args.output.resolve() / run_id
    local.mkdir(parents=True)
    write(local / "plan.json", plan)
    sha = hashlib.sha256((ROOT / PROBE).read_bytes()).hexdigest()
    q = shared.ps_quote
    preflight = f"""$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$identity=[IO.File]::ReadAllText({q(win('build/third-round/package/installer-identity.json'))})|ConvertFrom-Json
if(-not $identity.passed) {{throw 'Package verification must pass first'}}
$payload=Join-Path $identity.payload 'FiberDiameterMeasurement.exe'
$exe={q(args.executable) if args.executable else '$payload'}
if((Get-FileHash -LiteralPath $exe).Hash -ne (Get-FileHash -LiteralPath $payload).Hash) {{throw 'Executable differs from verified payload'}}
if((Get-FileHash -LiteralPath {q(win(PROBE))}).Hash.ToLowerInvariant() -ne '{sha}') {{throw 'Probe differs from host source'}}
[void](New-Item -ItemType Directory -Path {q(remote)})
[Console]::WriteLine('FDM_PREFLIGHT='+(@{{identity=$identity;executable=$exe;probe_sha256='{sha}'}}|ConvertTo-Json -Depth 15 -Compress))
"""
    output = shared.run_powershell(args.vm, preflight, timeout=90)
    lines = [line[len('FDM_PREFLIGHT='):] for line in output.stdout.splitlines() if line.startswith('FDM_PREFLIGHT=')]
    if len(lines) != 1:
        raise RuntimeError('Preflight did not produce one report')
    verified = json.loads(lines[0])
    summary = dict(**plan, preflight=verified, groups=[], passed=False)
    write(local / "summary.json", summary)
    for case in selected:
        group_dir = str(PureWindowsPath(remote) / case['name'])
        populated = '$true' if case['populated'] else '$false'
        export = '$true' if case.get('export') else '$false'
        code = f"""$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$out={q(group_dir)}
[void](New-Item -ItemType Directory -Path $out)
& {q(win(PROBE))} -Executable {q(verified['executable'])} -OutputRoot $out -Theme {case['theme']} -ClientWidth {case['width']} -ClientHeight {case['height']} -Populated:{populated} -ExportReport:{export} *> (Join-Path $out 'probe.log')
$code=$LASTEXITCODE
$result=Get-ChildItem -LiteralPath $out -Filter result.json -Recurse|Select-Object -First 1 -ExpandProperty FullName
if(-not $result) {{throw 'Probe did not write result.json'}}
[Console]::WriteLine('{MARKER}'+[IO.File]::ReadAllText($result).Replace("`r",'').Replace("`n",''))

"""
        print(json.dumps(dict(event='start', **case)), flush=True)
        try:
            completed = shared.run_powershell(args.vm, code, timeout=600)
            (local / (case['name'] + '.stdout.log')).write_text(completed.stdout + completed.stderr, encoding='utf-8')
            reports = [line[len(MARKER):] for line in completed.stdout.splitlines() if line.startswith(MARKER)]
            decoded = [json.loads(row) for row in reports]
            if not decoded or any(row != decoded[0] for row in decoded[1:]):
                raise RuntimeError('Probe did not return a consistent result')
            result = decoded[0]
            report = dict(name=case['name'], passed=bool(result['ok']), result=result)
        except Exception as error:
            report = dict(name=case['name'], passed=False, error=str(error))
        summary['groups'].append(report)
        write(local / 'summary.json', summary)
        print(json.dumps(dict(event='finished', name=case['name'], passed=report['passed'])), flush=True)
        if not report['passed']:
            break
    summary['passed'] = len(summary['groups']) == len(selected) and all(g['passed'] for g in summary['groups'])
    write(local / 'summary.json', summary)
    print(json.dumps(dict(passed=summary['passed'], output=str(local), windows_output=remote)), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
