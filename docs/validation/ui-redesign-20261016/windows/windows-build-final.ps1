$ErrorActionPreference='Continue'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new()
$root='C:\tmp\fdm-cu-live-ui-redesign-20261016'
$out=Join-Path $root 'build\ui-redesign-final'
New-Item -ItemType Directory -Force $out | Out-Null
$old=Join-Path $root 'dist\installer\fiber-diameter-measurement-setup-0.4.9-no-area-models.exe'
$archive=Join-Path $root 'dist\installer\intermediate\before-small-height-fix'
if (Test-Path $old) {New-Item -ItemType Directory -Force $archive | Out-Null; if (Test-Path (Join-Path $archive (Split-Path $old -Leaf))) {throw 'Intermediate installer already archived'}; Move-Item $old $archive}
Set-Location $root
$env:PYTHONPATH="$root\src;$root\validation_vendor"
$env:QT_QPA_PLATFORM='windows'; $env:QT_SCALE_FACTOR='0.5'; $env:PYTHONIOENCODING='utf-8'
foreach($name in @('FDM_TEST_MICROVIEW_DLL','FDM_MICROVIEW_DLL_DIR','FDM_TEST_MICROVIEW_FIXED_PERIOD','FDM_TEST_MICROVIEW_REUSE_PROCESSED')) {Remove-Item "Env:$name" -ErrorAction SilentlyContinue}
& 'C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe' scripts/build_windows_installer.py --exclude-area-models *> "$out\installer-build.log"
$code=$LASTEXITCODE
"installer_exit=$code" | Set-Content "$out\installer-status.txt"
Get-Content "$out\installer-build.log" -Tail 35
exit $code
