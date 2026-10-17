#!/usr/bin/env python3
"""Run the second-round candidate's isolated Windows GUI acceptance from macOS.

This does not install an application, manage VM snapshots, or touch the existing
installation. Use only during an exclusive Windows GUI validation window. The
PowerShell probe copies the old fixture before editing it and owns each process
and profile it creates. ``--dry-run`` performs no Windows commands.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from sync_cu_live_windows import powershell, ps_quote

WINDOWS_ROOT = r"C:\tmp\fdm-cu-live-second-round-20261017"
LEGACY_RELATIVE = (
    "docs/validation/ui-redesign-20261016/windows/evidence-final/"
    "frozen-legacy-upgrade/20261016-182409-9cf7eca8528d48e586723ec65c87c70c/"
    "save-lifecycle-probe.fdmproj"
)
PROBE_RELATIVE = "scripts/validation/probe_workflow_redesign_frozen.ps1"
MARKER = "FDM_SECOND_ROUND_JSON="


def windows_path(relative: str) -> str:
    return str(PureWindowsPath(WINDOWS_ROOT) / PureWindowsPath(relative))


def candidate_executable(value: str) -> str:
    path = PureWindowsPath(value)
    allowed = PureWindowsPath(WINDOWS_ROOT) / "dist/windows"
    if ".." in path.parts or not path.is_absolute() or path.name.lower() != "fiberdiametermeasurement.exe":
        raise argparse.ArgumentTypeError("Expected the candidate FiberDiameterMeasurement.exe")
    if not path.is_relative_to(allowed):
        raise argparse.ArgumentTypeError(f"Executable must be under {allowed}")
    return str(path)


def cases() -> list[dict]:
    clean = [dict(name=f"clean-{theme}-1366x690", theme=theme, width=1366, height=690,
                  legacy=False, save_rounds=0) for theme in ("dark", "light", "navy", "warm")]
    return clean + [
        dict(name="clean-light-1280x670", theme="light", width=1280, height=670, legacy=False, save_rounds=0),
        dict(name="legacy-049-dark-1366x690", theme="dark", width=1366, height=690, legacy=True, save_rounds=20),
    ]


def parse_report(output: str) -> dict:
    lines = [line[len(MARKER):] for line in output.splitlines() if line.startswith(MARKER)]
    if len(lines) != 1:
        raise RuntimeError("Windows command did not return exactly one acceptance report")
    return json.loads(lines[0])


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_remote_bytes(vm: str, path: str, payload: bytes) -> None:
    """Bound every EncodedCommand below the guest command-line limit."""
    temporary = path + "." + uuid.uuid4().hex + ".upload"
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    for offset in range(0, len(encoded), 512):
        chunk = encoded[offset:offset + 512]
        mode = "CreateNew" if offset == 0 else "Append"
        code = ("$ErrorActionPreference='Stop'; $s=[IO.File]::Open(" + ps_quote(temporary)
                + f",[IO.FileMode]::{mode},[IO.FileAccess]::Write); try {{$b=[Convert]::FromBase64String('{chunk}');"
                + "$s.Write($b,0,$b.Length)} finally {$s.Dispose()}")
        powershell(vm, code, timeout=60)
    code = ("$ErrorActionPreference='Stop'; if((Get-FileHash -LiteralPath " + ps_quote(temporary)
            + f" -Algorithm SHA256).Hash.ToLowerInvariant() -ne '{digest}'){{throw 'Transferred validation data SHA mismatch'}}; "
            + "Move-Item -LiteralPath " + ps_quote(temporary) + " -Destination " + ps_quote(path) + " -Force")
    powershell(vm, code, timeout=60)


def run_powershell(vm: str, code: str, *, timeout: int):
    # prlctl reports misleading guest-session errors for overlong command lines.
    # Short probes stay inline; longer trusted validation code is staged inside
    # this task's existing build directory and removed after execution.
    if len(base64.b64encode(code.encode("utf-16-le"))) <= 2800:
        return powershell(vm, code, timeout=timeout)
    script = windows_path("build/second-round/runner-command-" + uuid.uuid4().hex + ".ps1")
    write_remote_bytes(vm, script, b"\xef\xbb\xbf" + code.encode("utf-8"))
    invoke = ("$ErrorActionPreference='Stop'; try { & " + ps_quote(script)
              + " } finally { Remove-Item -LiteralPath " + ps_quote(script) + " -Force }")
    return powershell(vm, invoke, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vm", default="Windows 11")
    parser.add_argument("--executable", type=candidate_executable,
                        help="Optional actual installed candidate, confined to this task's dist/windows")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/validation/acquisition-second-round-20261017/windows/frozen-gui")
    parser.add_argument("--dry-run", action="store_true", help="Print the six-case plan; never contact the VM")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ-") + uuid.uuid4().hex[:8]
    remote_output = windows_path(f"build/second-round/frozen-gui/{run_id}")
    identity_path = windows_path("build/second-round/package/installer-identity.json")
    fixture_bytes = (ROOT / LEGACY_RELATIVE).read_bytes()
    if json.loads(fixture_bytes)["version"] != "0.4.9":
        parser.error("The recorded legacy fixture must have been saved by version 0.4.9")
    fixture_sha = hashlib.sha256(fixture_bytes).hexdigest()
    fixture_path = windows_path(f"build/second-round/fixtures/legacy-049-{fixture_sha[:12]}.fdmproj")
    probe_path = windows_path(PROBE_RELATIVE)
    probe_sha = hashlib.sha256((ROOT / PROBE_RELATIVE).read_bytes()).hexdigest()
    plan = dict(schema=1, run_id=run_id, vm=args.vm, identity=identity_path,
                executable_override=args.executable, legacy_project=fixture_path,
                legacy_source=str(ROOT / LEGACY_RELATIVE), legacy_source_sha256=fixture_sha,
                windows_output=remote_output, probe_sha256=probe_sha, cases=cases(),
                real_hardware_tested=False, installation_performed=False,
                snapshot_management_performed=False)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    local_output = args.output.resolve() / run_id
    local_output.mkdir(parents=True)
    write_json(local_output / "plan.json", plan)
    # Historical evidence is excluded by some source snapshots. Copy the exact
    # recorded 0.4.9 fixture into an owned validation directory, never an install
    # or user-data directory; the GUI probe makes another per-run editable copy.
    fixture_prepare = f"""$ErrorActionPreference='Stop'
$path={ps_quote(fixture_path)}
[void](New-Item -ItemType Directory -Force -Path (Split-Path $path))
if(Test-Path -LiteralPath $path) {{
    if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne '{fixture_sha}') {{ throw 'Existing validation fixture hash mismatch' }}
    [Console]::WriteLine('fixture-present')
}}
"""
    prepared = run_powershell(args.vm, fixture_prepare, timeout=60)
    if "fixture-present" not in prepared.stdout:
        write_remote_bytes(args.vm, fixture_path, fixture_bytes)
    # The caller syncs validation scripts after the candidate build. Refuse an
    # older probe rather than silently measuring a different navigation path.
    preflight = f"""$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$identity=Get-Content -LiteralPath {ps_quote(identity_path)} -Raw -Encoding UTF8 | ConvertFrom-Json
if(-not $identity.passed) {{ throw 'Candidate package verification did not pass' }}
$payloadExe=Join-Path $identity.payload 'FiberDiameterMeasurement.exe'
$exe={ps_quote(args.executable) if args.executable else '$payloadExe'}
$exe=[IO.Path]::GetFullPath($exe)
$allowed={ps_quote(windows_path('dist/windows') + chr(92))}
if(-not $exe.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)) {{ throw 'Executable escaped task dist/windows' }}
if((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $payloadExe -Algorithm SHA256).Hash) {{ throw 'Candidate executable differs from verified payload' }}
$manifest=Get-Content -LiteralPath (Join-Path (Split-Path $exe) 'release-manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if($manifest.build_id -ne $identity.build_id) {{ throw 'Executable manifest build ID differs from verified package' }}
if((Get-FileHash -LiteralPath {ps_quote(probe_path)} -Algorithm SHA256).Hash.ToLowerInvariant() -ne '{probe_sha}') {{ throw 'Validation probe differs from host; sync the validation script before running' }}
$fixture=Get-Content -LiteralPath {ps_quote(fixture_path)} -Raw -Encoding UTF8 | ConvertFrom-Json
if($fixture.version -ne '0.4.9' -or $fixture.live_experiments[0].sample_name -ne 'save-lifecycle-probe') {{ throw 'Expected the recorded 0.4.9 save-lifecycle fixture' }}
[void](New-Item -ItemType Directory -Path {ps_quote(remote_output)})
$report=@{{identity=$identity; executable=$exe; executable_sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant(); legacy_sha256=(Get-FileHash -LiteralPath {ps_quote(fixture_path)} -Algorithm SHA256).Hash.ToLowerInvariant(); probe_sha256='{probe_sha}'}}
[IO.File]::WriteAllText((Join-Path {ps_quote(remote_output)} 'preflight.json'),($report|ConvertTo-Json -Depth 20),[Text.UTF8Encoding]::new($false))
[Console]::WriteLine('{MARKER}'+($report|ConvertTo-Json -Depth 20 -Compress))
"""
    try:
        preflight_result = run_powershell(args.vm, preflight, timeout=120)
    except Exception as error:
        (local_output / "preflight.error.log").write_text(str(error), encoding="utf-8")
        raise
    (local_output / "preflight.stdout.log").write_text(preflight_result.stdout + preflight_result.stderr, encoding="utf-8")
    verified = parse_report(preflight_result.stdout)
    summary = dict(**plan, preflight=verified, groups=[], passed=False)
    write_json(local_output / "summary.json", summary)
    for case in plan["cases"]:
        print(json.dumps(dict(event="start", **case), ensure_ascii=False), flush=True)
        group_output = str(PureWindowsPath(remote_output) / case["name"])
        save_arg = "$true" if case["legacy"] else "$false"
        legacy_arg = " -LegacyProject " + ps_quote(fixture_path) if case["legacy"] else ""
        code = f"""$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$out={ps_quote(group_output)}
[void](New-Item -ItemType Directory -Path $out)
$log=Join-Path $out 'probe.stdout.log'
$LASTEXITCODE=0
& {ps_quote(probe_path)} -Executable {ps_quote(verified['executable'])} -OutputRoot $out -Theme {case['theme']} -ClientWidth {case['width']} -ClientHeight {case['height']} -ExerciseBackgroundSaving:{save_arg}{legacy_arg} *> $log
$probeExit=$LASTEXITCODE
$resultPath=Get-ChildItem -LiteralPath $out -Filter result.json -Recurse | Select-Object -First 1 -ExpandProperty FullName
if(-not $resultPath) {{ throw ('No probe result; inspect '+$log) }}
$result=Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
$report=@{{name={ps_quote(case['name'])}; probe_exit=$probeExit; result_path=$resultPath; stdout_path=$log; result=$result}}
[IO.File]::WriteAllText((Join-Path $out 'group-summary.json'),($report|ConvertTo-Json -Depth 30),[Text.UTF8Encoding]::new($false))
Get-Content -LiteralPath $log
[Console]::WriteLine('{MARKER}'+($report|ConvertTo-Json -Depth 30 -Compress))
"""
        try:
            completed = run_powershell(args.vm, code, timeout=900)
            (local_output / (case["name"] + ".stdout.log")).write_text(completed.stdout + completed.stderr, encoding="utf-8")
            report = parse_report(completed.stdout)
            result = report["result"]
            report["passed"] = bool(result.get("ok") and report["probe_exit"] == 0)
            if case["legacy"]:
                report["passed"] &= bool(result.get("background_save_verified") and len(result.get("saved_count_checkpoints", [])) == 20)
        except Exception as error:
            report = dict(name=case["name"], passed=False, error=str(error))
            (local_output / (case["name"] + ".error.log")).write_text(str(error), encoding="utf-8")
        summary["groups"].append(report)
        write_json(local_output / "summary.json", summary)
        print(json.dumps(dict(event="finished", name=case["name"], passed=report["passed"])), flush=True)
        # A broken/ambiguous owned-process cleanup must be inspected before the
        # next GUI group. Never launch another application over an unknown state.
        if not report["passed"]:
            break
    summary["passed"] = len(summary["groups"]) == len(plan["cases"]) and all(g["passed"] for g in summary["groups"])
    summary["local_output"] = str(local_output)
    write_json(local_output / "summary.json", summary)
    # Detailed group results already live on the guest. Assemble them there;
    # transferring six complete UIA result trees back again is unnecessary.
    compact = dict(summary, preflight=None,
                   groups=[{k: v for k, v in group.items() if k in {"name", "passed", "error"}}
                           for group in summary["groups"]])
    encoded = base64.b64encode(json.dumps(compact, ensure_ascii=False, allow_nan=False).encode("utf-8")).decode("ascii")
    archive = f"""$ErrorActionPreference='Stop'
$out={ps_quote(remote_output)}
$summary=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded}')) | ConvertFrom-Json
$summary.preflight=Get-Content -LiteralPath (Join-Path $out 'preflight.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$groups=@()
foreach($group in $summary.groups) {{
    $path=Join-Path (Join-Path $out $group.name) 'group-summary.json'
    if(Test-Path -LiteralPath $path) {{
        $saved=Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
        $saved | Add-Member -NotePropertyName passed -NotePropertyValue $group.passed -Force
        $groups+=,$saved
    }} else {{ $groups+=,$group }}
}}
$summary.groups=$groups
[IO.File]::WriteAllText((Join-Path $out 'summary.json'),($summary|ConvertTo-Json -Depth 30),[Text.UTF8Encoding]::new($false))
"""
    run_powershell(args.vm, archive, timeout=120)
    print(json.dumps(dict(passed=summary["passed"], local_output=str(local_output), windows_output=remote_output,
                          groups_completed=len(summary["groups"])), ensure_ascii=False), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
