$ErrorActionPreference='Continue';$ProgressPreference='SilentlyContinue'
$root='C:\tmp\fdm-brightness-release-20261023';$share='\\Mac\Home\Downloads\fdm-brightness-release-transfer-20261023'
& 'C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe' (Join-Path $root 'verify_package.py') --root $root --installer (Join-Path $root 'dist\installer\fiber-diameter-measurement-setup-0.4.11-no-area-models.exe') --extractor 'C:\tmp\fdm-cu-live-final-r4-20261015\build\validation-tools\innounp.exe' > (Join-Path $root 'build\verify.log') 2>&1
[pscustomobject]@{exit_code=$LASTEXITCODE;finished=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content (Join-Path $root 'build\verify-status.json') -Encoding UTF8
Copy-Item (Join-Path $root 'build\verify-status.json') $share -Force
Copy-Item (Join-Path $root 'build\verify.log') $share -Force
