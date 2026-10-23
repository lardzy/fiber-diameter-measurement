$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$root='C:\tmp\fdm-brightness-release-20261023';$share='\\Mac\Home\Downloads\fdm-brightness-release-transfer-20261023'
$status=[ordered]@{started=(Get-Date).ToUniversalTime().ToString('o');finished=$null;exit_code=$null;root=$root}
try {
 & cmd.exe /c (Join-Path $root 'build.cmd') > (Join-Path $root 'build\release-build.log') 2>&1
 $status.exit_code=$LASTEXITCODE
} catch { $status.exit_code=-1;$status.error=$_.Exception.Message }
$status.finished=(Get-Date).ToUniversalTime().ToString('o')
$status | ConvertTo-Json | Set-Content (Join-Path $root 'build\release-status.json') -Encoding UTF8
Copy-Item (Join-Path $root 'build\release-status.json') $share -Force
Copy-Item (Join-Path $root 'build\release-build.log') $share -Force
