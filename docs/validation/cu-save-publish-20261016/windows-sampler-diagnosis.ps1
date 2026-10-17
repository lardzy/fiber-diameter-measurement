$ErrorActionPreference='Continue'
$root='C:\tmp\fdm-cu-live-save-peaks-20261016'
$out="$root\build\save-peaks\final"
$python='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
Set-Location $root
$env:PYTHONPATH="$root\src;$root\validation_vendor"
$env:QT_QPA_PLATFORM='windows'
$env:QT_SCALE_FACTOR='0.5'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
foreach($mode in @('thread','process')) {
 & $python scripts/benchmark_cu_live_ui.py --sizes 768 --frame-height 576 --history-count 2000 --warmup 5 --samples 300 --evidence off --motion-input lsb-noise --frame-hz 25 --opencv-threads 1 --profile-publish --resource-sampling $mode --show-records --window-width 1280 --window-height 720 --output "$out\sampler-$mode.json" *> "$out\sampler-$mode.log"
 "sampler_$mode=$LASTEXITCODE" | Tee-Object -Append -FilePath "$out\status.txt"
 Get-Content "$out\sampler-$mode.log" -Tail 2
}
exit 0
