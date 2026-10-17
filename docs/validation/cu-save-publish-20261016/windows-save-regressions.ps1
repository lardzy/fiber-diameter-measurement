$ErrorActionPreference='Continue'
$root='C:\tmp\fdm-cu-live-save-peaks-20261016'
$out="$root\build\save-peaks\fixed"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$python='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
Set-Location $root
$env:PYTHONPATH="$root\src;$root\validation_vendor"
$env:QT_QPA_PLATFORM='windows'
$env:QT_SCALE_FACTOR='0.5'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONFAULTHANDLER='1'
& $python -m pytest -q tests/test_project_background_save.py *> "$out\background-save.log"
$saveExit=$LASTEXITCODE
"background_save_exit=$saveExit" | Tee-Object -FilePath "$out\status.txt"
Get-Content "$out\background-save.log" -Tail 5
for($i=1;$i -le 3;$i++){
 & $python -m pytest -q tests/test_build_support.py tests/test_build_windows_onedir.py tests/test_release_self_check.py *> "$out\combined-build-$i.log"
 $code=$LASTEXITCODE
 "combined_$i=$code" | Tee-Object -Append -FilePath "$out\status.txt"
 Get-Content "$out\combined-build-$i.log" -Tail 5
 if($code -ne 0){break}
}
exit 0
