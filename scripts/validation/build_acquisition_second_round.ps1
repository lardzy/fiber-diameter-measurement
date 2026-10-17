param([switch]$SkipTests)
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $root.StartsWith('C:\tmp\fdm-cu-live-')) { throw 'Use an isolated validation checkout.' }
Set-Location $root
$out = Join-Path $root 'build\second-round'
New-Item -ItemType Directory -Force $out | Out-Null
$python = 'C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$env:PYTHONPATH = "$root\src;$root\validation_vendor"
$env:QT_QPA_PLATFORM = 'windows'
$env:QT_SCALE_FACTOR = '0.5'
$env:PYTHONIOENCODING = 'utf-8'
foreach ($name in @('FDM_TEST_MICROVIEW_DLL','FDM_MICROVIEW_DLL_DIR','FDM_TEST_MICROVIEW_FIXED_PERIOD','FDM_TEST_MICROVIEW_REUSE_PROCESSED','PSExecutionPolicyPreference')) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
if (-not $SkipTests) {
    & $python -m pytest -q tests/test_acquisition_second_round.py tests/test_acquisition_workflow_redesign.py tests/test_acquisition_compact_client.py tests/test_acquisition_workbench.py tests/test_measurement_feedback.py tests/test_live_experiment_reminders.py tests/test_quantity_target_dialog.py tests/test_excel_template_runtime.py tests/test_cu_template_contract.py tests/test_live_experiment_content_workspace.py tests/test_live_display_preferences.py --disable-warnings *> "$out\windows-tests.log"
    $testExit = $LASTEXITCODE
    "exit=$testExit" | Set-Content "$out\windows-tests-status.txt"
    if ($testExit -ne 0) { exit $testExit }
}
& $python scripts/build_windows_installer.py --exclude-area-models *> "$out\installer-build.log"
$buildExit = $LASTEXITCODE
"exit=$buildExit" | Set-Content "$out\installer-build-status.txt"
if ($buildExit -ne 0) { exit $buildExit }
& $python scripts/validation/verify_second_round_installer.py --root $root --installer "$root\dist\installer\fiber-diameter-measurement-setup-0.4.10-no-area-models.exe" --extractor 'C:\tmp\fdm-cu-live-final-r4-20261015\build\validation-tools\innounp.exe' *> "$out\package-verify.log"
$verifyExit = $LASTEXITCODE
"exit=$verifyExit" | Set-Content "$out\package-verify-status.txt"
exit $verifyExit
