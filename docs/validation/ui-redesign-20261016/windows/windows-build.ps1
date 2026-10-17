$ErrorActionPreference='Continue'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new()
$root='C:\tmp\fdm-cu-live-ui-redesign-20261016'
$out=Join-Path $root 'build\ui-redesign'
Set-Location $root
$env:PYTHONPATH="$root\src;$root\validation_vendor"
$env:QT_QPA_PLATFORM='windows'; $env:QT_SCALE_FACTOR='0.5'; $env:PYTHONIOENCODING='utf-8'
foreach($name in @('FDM_TEST_MICROVIEW_DLL','FDM_MICROVIEW_DLL_DIR','FDM_TEST_MICROVIEW_FIXED_PERIOD','FDM_TEST_MICROVIEW_REUSE_PROCESSED')) {Remove-Item "Env:$name" -ErrorAction SilentlyContinue}
& 'C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe' scripts/build_windows_installer.py --exclude-area-models *> "$out\installer-build.log"
$code=$LASTEXITCODE
"installer_exit=$code" | Set-Content "$out\installer-status.txt"
Get-Content "$out\installer-build.log" -Tail 35
exit $code
