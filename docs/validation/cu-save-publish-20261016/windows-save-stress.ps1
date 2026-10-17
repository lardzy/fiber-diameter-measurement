$ErrorActionPreference='Continue'
$root='C:\tmp\fdm-cu-live-save-peaks-20261016'
$out="$root\build\save-peaks\fixed"
$python='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
Set-Location $root
$env:PYTHONPATH="$root\src;$root\validation_vendor"
$env:QT_QPA_PLATFORM='windows'
$env:QT_SCALE_FACTOR='0.5'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONFAULTHANDLER='1'
& $python scripts/validation/stress_project_save.py --rounds 200 --output "$out\save-stress.json" *> "$out\save-stress.log"
"stress_exit=$LASTEXITCODE" | Tee-Object -Append -FilePath "$out\status.txt"
Get-Content "$out\save-stress.log" -Tail 4
& $python scripts/benchmark_cu_live_ui.py --sizes 768 --frame-height 576 --history-count 2000 --warmup 5 --samples 200 --evidence off --motion-input lsb-noise --frame-hz 25 --opencv-threads 1 --profile-publish --window-width 1280 --window-height 720 --output "$out\publish-phases-before.json" *> "$out\publish-phases-before.log"
"profile_exit=$LASTEXITCODE" | Tee-Object -Append -FilePath "$out\status.txt"
Get-Content "$out\publish-phases-before.log" -Tail 4
exit 0
