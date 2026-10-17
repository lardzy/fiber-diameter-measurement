$ErrorActionPreference='Continue'
$root='C:\tmp\fdm-cu-live-save-peaks-20261016'
$out="$root\build\save-peaks\final"
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
$file="$root\src\fdm\ui\live_experiment_workspace.py"
$baseline='C:\tmp\fdm-cu-live-preview-fix-20261015\src\fdm\ui\live_experiment_workspace.py'
Copy-Item $file "$out\workspace-final.py" -Force
Copy-Item $baseline "$out\workspace-baseline.py" -Force
try {
 foreach($history in @(2000,0,200)) {
  foreach($variant in @('before','after')) {
   if($variant -eq 'before'){ Copy-Item "$out\workspace-baseline.py" $file -Force } else { Copy-Item "$out\workspace-final.py" $file -Force }
   $name="external-$variant-history-$history"
   & $python scripts/benchmark_cu_live_ui.py --sizes 768 --frame-height 576 --history-count $history --warmup 5 --samples 100 --evidence off --motion-input lsb-noise --frame-hz 25 --opencv-threads 1 --window-width 1280 --window-height 720 --resource-sampling process --output "$out\$name.json" *> "$out\$name.log"
   $code=$LASTEXITCODE
   "$name=$code" | Tee-Object -Append -FilePath "$out\status.txt"
   if($code -ne 0){break}
  }
 }
} finally { Copy-Item "$out\workspace-final.py" $file -Force }
& $python scripts/benchmark_cu_live_ui.py --sizes 768 --frame-height 576 --history-count 2000 --warmup 5 --samples 100 --evidence off --motion-input lsb-noise --frame-hz 25 --opencv-threads 1 --show-records --window-width 1280 --window-height 720 --resource-sampling process --screenshot "$out\external-records-expanded.png" --output "$out\external-after-expanded-history-2000.json" *> "$out\external-after-expanded.log"
"external_after_expanded=$LASTEXITCODE" | Tee-Object -Append -FilePath "$out\status.txt"
exit 0
