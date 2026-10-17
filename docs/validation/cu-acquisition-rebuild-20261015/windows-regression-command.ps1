$ErrorActionPreference='Continue'
$root='C:\tmp\fdm-cu-live-rebuild-20261015'
Set-Location $root
$env:PYTHONPATH="$root\src;$root\validation_vendor"
$env:FDM_TEST_MICROVIEW_DLL="$root\build\test-sdk\MVAPI.dll"
$env:QT_QPA_PLATFORM='offscreen'
& 'C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe' -m pytest -q tests/test_microview_native_windows.py tests/test_live_experiment_sources.py tests/test_cu_template_contract.py tests/test_trial_input_contract.py tests/test_capture_and_cu_scale.py tests/test_live_experiment_content_workspace.py tests/test_live_experiment_cu_transition.py tests/test_live_experiment_post_classification.py tests/test_cu_live_self_check.py tests/test_acquisition_workbench.py tests/test_main_window_view_experience.py --disable-warnings *> build\cu-live-validation\final-tests.txt
$code=$LASTEXITCODE
Write-Output ((Get-Content build\cu-live-validation\final-tests.txt -Tail 65) -join "`n")
exit $code
