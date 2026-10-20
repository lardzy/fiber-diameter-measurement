param([switch]$SkipTests, [switch]$TestsOnly, [string[]]$AdditionalTests=@())
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ($root -ne 'C:\tmp\fdm-cu-live-fourth-round-20261019') { throw 'Use the isolated fourth-round checkout.' }
Set-Location $root
$out = Join-Path $root 'build\fourth-round'
New-Item -ItemType Directory -Force $out | Out-Null
$python = 'C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$extractor = 'C:\tmp\fdm-cu-live-final-r4-20261015\build\validation-tools\innounp.exe'
$installer = Join-Path $root 'dist\installer\fiber-diameter-measurement-setup-0.4.10-no-area-models.exe'
if (Test-Path -LiteralPath $installer) { throw 'A candidate already exists; preserve it before a new build.' }
foreach ($path in @($python, $extractor, "$root\.fdm_snapshot.json")) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing build prerequisite: $path" }
}
$env:PYTHONPATH = "$root\src;$root\validation_vendor"
$env:QT_QPA_PLATFORM = 'windows'
$env:QT_SCALE_FACTOR = '0.5'
$env:PYTHONIOENCODING = 'utf-8'
foreach ($name in @('FDM_TEST_MICROVIEW_DLL','FDM_MICROVIEW_DLL_DIR','FDM_TEST_MICROVIEW_FIXED_PERIOD','FDM_TEST_MICROVIEW_REUSE_PROCESSED','PSExecutionPolicyPreference')) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
# Native stderr may contain harmless build warnings on PowerShell 5.1.
$ErrorActionPreference = 'Continue'
& $python -c "from fdm.version import __version__; assert __version__ == '0.4.10', __version__"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$tests = @(
    'tests/test_candidate_controller.py', 'tests/test_cu_automatic_worker.py',
    'tests/test_live_performance_diagnostics.py', 'tests/test_trial_history_shortcuts.py',
    'tests/test_live_measurement_annotations.py', 'tests/test_live_measurement_candidate_canvas.py',
    'tests/test_measurement_evidence_view.py', 'tests/test_trial_assistance_ui.py',
    'tests/test_acquisition_fourth_round_visual.py',
    'tests/test_live_experiment_crop_evidence.py', 'tests/test_live_experiment_evidence.py',
    'tests/test_live_experiment_content_project_assets.py', 'tests/test_live_experiment_save_rebase.py',
    'tests/test_acquisition_third_round.py', 'tests/test_trial_records.py',
    'tests/test_trial_compact_layout.py', 'tests/test_trial_records_integration.py',
    'tests/test_theme_readability.py', 'tests/test_theme_and_fluent_icons.py',
    'tests/test_export_options_dialog.py', 'tests/test_acquisition_second_round.py',
    'tests/test_acquisition_workflow_redesign.py', 'tests/test_acquisition_compact_client.py',
    'tests/test_acquisition_workbench.py', 'tests/test_live_experiment_content_workspace.py',
    'tests/test_live_display_preferences.py', 'tests/test_trial_feedback_history.py',
    'tests/test_excel_template_runtime.py', 'tests/test_cu_template_contract.py'
) + $AdditionalTests
$tests | ConvertTo-Json | Set-Content "$out\test-selection.json" -Encoding UTF8
if (-not $SkipTests) {
    & $python -m pytest -q @tests --disable-warnings -o faulthandler_timeout=180 *> "$out\windows-tests.log"
    $testExit = $LASTEXITCODE
    "exit=$testExit" | Set-Content "$out\windows-tests-status.txt"
    if ($testExit -ne 0) { exit $testExit }
}
if ($TestsOnly) { exit 0 }
& $python scripts/build_windows_installer.py --exclude-area-models *> "$out\installer-build.log"
$buildExit = $LASTEXITCODE
"exit=$buildExit" | Set-Content "$out\installer-build-status.txt"
if ($buildExit -ne 0) { exit $buildExit }
& $python scripts/validation/verify_fourth_round_installer.py --root $root --installer $installer --extractor $extractor *> "$out\package-verify.log"
$verifyExit = $LASTEXITCODE
"exit=$verifyExit" | Set-Content "$out\package-verify-status.txt"
exit $verifyExit
