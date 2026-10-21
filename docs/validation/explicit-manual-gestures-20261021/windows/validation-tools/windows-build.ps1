$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root='C:\tmp\fdm-cu-live-explicit-gestures-20261021'
$out=Join-Path $root 'build\seventh-round'
$tests=@('tests/test_live_explicit_manual_gesture.py','tests/test_live_measurement_mixed_canvas.py','tests/test_live_mixed_presented_frame.py','tests/test_trial_assistance_ui.py','tests/test_round7_canvas.py','tests/test_round6_manual_start.py')
$selection=Join-Path $out 'patch-test-selection.txt'
[IO.File]::WriteAllLines($selection,$tests,[Text.UTF8Encoding]::new($false))
& (Join-Path $root 'scripts\validation\build_acquisition_seventh_round.ps1') -TestSelectionFile $selection
$code=$LASTEXITCODE
if($code -ne 0){exit $code}
$gate=[ordered]@{schema=1;created_at=(Get-Date).ToString('o');delivery_kind='explicit-manual-gestures';quick_tests_passed=$true;package_verified=$true;full_regression_complete=$false;hardware_capture_tested=$false;version='0.4.10';note='User-authorized explicit two-point versus drag gesture contracts. Mixed candidate short clicks remain enabled. Previous patch and round 7 rollback packages preserved.'}
[IO.File]::WriteAllText((Join-Path $out 'build-validation-gate.json'),($gate|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
