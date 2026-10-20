"""Run one final, unmodified Windows presentation case in the isolated checkout.

Run baseline and guard separately for each interval. This deliberately keeps
every result, including a failed visibility trace, and refuses to overwrite it.
The three-second recovery tail never contributes to the load FPS or frame age.
"""
from __future__ import annotations

import argparse
import re

from fifth_round_remote import run_powershell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vm", default="Windows 11")
    parser.add_argument("--mode", choices=("baseline", "guard"), required=True)
    parser.add_argument("--measure-ms", type=int, choices=(150, 1000), required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.snapshot_sha256):
        parser.error("snapshot-sha256 must be the lowercase SHA256 of the frozen source manifest")
    script = r"""$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root='C:\tmp\fdm-cu-live-fifth-round-20261020'
$manifest=Join-Path $root '.fdm_snapshot.json'
if((Get-FileHash -LiteralPath $manifest).Hash.ToLowerInvariant() -ne '__SNAPSHOT__'){throw 'Frozen source identity changed'}
$snapshot=[IO.File]::ReadAllText($manifest)|ConvertFrom-Json
foreach($row in $snapshot.files){
 if($row.path.StartsWith('src/') -or $row.path -in @('scripts/validation/benchmark_candidate_preview.py','scripts/validation/validate_candidate_presentation.py')){
  if((Get-FileHash -LiteralPath (Join-Path $root $row.path)).Hash.ToLowerInvariant() -ne $row.sha256){throw ('Source changed: '+$row.path)}
 }
}
$out=Join-Path $root 'build\fifth-round\final-performance'
[void](New-Item -ItemType Directory -Force -Path $out)
$stem='__MODE__-__INTERVAL__ms'
$result=Join-Path $out ($stem+'.json')
$launch=Join-Path $out ($stem+'-launch.json')
if((Test-Path $result) -or (Test-Path $launch)){throw 'Preserve existing performance evidence'}
Set-Location $root
$env:PYTHONPATH=Join-Path $root 'src'
$env:QT_QPA_PLATFORM='windows'
$env:QT_SCALE_FACTOR='0.5'
$env:PYTHONIOENCODING='utf-8'
foreach($name in @('FDM_TEST_MICROVIEW_DLL','FDM_MICROVIEW_DLL_DIR','FDM_TEST_MICROVIEW_FIXED_PERIOD','FDM_TEST_MICROVIEW_REUSE_PROCESSED')){Remove-Item "Env:$name" -ErrorAction SilentlyContinue}
$python='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$context=[ordered]@{started_at=(Get-Date).ToString('o');source_snapshot_sha256='__SNAPSHOT__';mode='__MODE__';measure_interval_ms=__INTERVAL__;load_seconds=30;recovery_seconds=3;qt_platform=$env:QT_QPA_PLATFORM;qt_scale_factor=$env:QT_SCALE_FACTOR;python=$python;gallery=$true;prototype=$false;real_camera=$false}
[IO.File]::WriteAllText($launch,($context|ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false))
$ErrorActionPreference='Continue'
& $python scripts/validation/validate_candidate_presentation.py --mode '__MODE__' --duration 30 --recovery-seconds 3 --measure-ms __INTERVAL__ --gallery --output $result *> (Join-Path $out ($stem+'.log'))
$code=$LASTEXITCODE
$ErrorActionPreference='Stop'
if(-not(Test-Path $result)){throw ('Performance process did not produce a result; exit='+$code)}
$context.finished_at=(Get-Date).ToString('o')
$context.exit_code=$code
$context.result_sha256=(Get-FileHash -LiteralPath $result).Hash.ToLowerInvariant()
[IO.File]::WriteAllText($launch,($context|ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false))
$report=[IO.File]::ReadAllText($result)|ConvertFrom-Json
[ordered]@{result=$result;exit_code=$code;passed=$report.presentation_trace.passed;fps=$report.fps;p95_frame_age_ms=$report.p95_frame_age_ms;transitions=$report.presentation_trace.visibility_transition_count;accepted=$report.accepted;saved=$report.saved;remaining_budget_bytes=$report.remaining_budget_bytes;post_load_recovery=$report.post_load_recovery;display_context=$report.display_context;sha256=$context.result_sha256}|ConvertTo-Json -Depth 12
"""
    for key, value in {"__SNAPSHOT__": args.snapshot_sha256, "__MODE__": args.mode,
                       "__INTERVAL__": str(args.measure_ms)}.items():
        script = script.replace(key, value)
    print(run_powershell(args.vm, script, timeout=180).stdout)


if __name__ == "__main__":
    main()
