$ErrorActionPreference='Continue'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new()
$root='C:\tmp\fdm-cu-live-ui-redesign-20261016'
$out=Join-Path $root 'build\ui-redesign'
$python='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $out,(Join-Path $root 'build\test-sdk') | Out-Null
Set-Location $root
$env:PYTHONPATH="$root\src;$root\validation_vendor"
$env:QT_QPA_PLATFORM='windows'
$env:QT_SCALE_FACTOR='0.5'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$batch=Join-Path $out 'compile-test-sdk.cmd'
$lines=@('@echo off','call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat"','if errorlevel 1 exit /b %errorlevel%',"cd /d `"$root`"",'cl /nologo /std:c++17 /EHsc /LD tests\native\microview_test_sdk.cpp /Fo:build\test-sdk\sdk.obj /Fe:build\test-sdk\MVAPI.dll /link user32.lib','exit /b %errorlevel%')
[IO.File]::WriteAllLines($batch,$lines,[Text.Encoding]::ASCII)
& cmd /c $batch *> "$out\test-sdk-build.log"
if($LASTEXITCODE -ne 0) {throw 'Test SDK compilation failed'}
$env:FDM_TEST_MICROVIEW_DLL="$root\build\test-sdk\MVAPI.dll"
& $python -m pytest -q tests/test_trial_workflow_redesign.py tests/test_acquisition_workflow_redesign.py tests/test_acquisition_workbench.py tests/test_live_experiment_canvas_navigation.py tests/test_live_preview_corrections.py tests/test_live_display_preferences.py tests/test_microview_native_windows.py tests/test_microview_display_routing.py tests/test_live_experiment_save_rebase.py tests/test_live_experiment_project_lifecycle.py tests/test_project_background_save.py tests/test_live_experiment_transition_confirmation.py tests/test_workflow_project_preferences.py tests/test_export_options_dialog.py tests/test_professional_workspace.py tests/test_command_bar_redesign.py tests/test_theme_and_fluent_icons.py tests/test_workspace_stack.py tests/test_export_service.py tests/test_project_export_controllers.py *> "$out\windows-regressions.log"
$regressions=$LASTEXITCODE
"regressions_exit=$regressions" | Set-Content "$out\tests-status.txt"
Get-Content "$out\windows-regressions.log" -Tail 35
if($regressions -ne 0) {throw 'Windows regression failure'}
& $python scripts/validation/validate_microview_workbench.py --sdk "$root\build\test-sdk\MVAPI.dll" --output "$out\native-workbench" *> "$out\native-workbench.log"
$native=$LASTEXITCODE
"native_workbench_exit=$native" | Add-Content "$out\tests-status.txt"
Get-Content "$out\native-workbench.log" -Tail 20
if($native -ne 0) {throw 'Native workbench validation failed'}
