$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$root='C:\tmp\fdm-brightness-release-20261023';$share='\\Mac\Home\Downloads\fdm-brightness-release-transfer-20261023'
if(Test-Path $root){throw 'Fresh build root already exists'}
if((Get-FileHash (Join-Path $share 'source.zip') -Algorithm SHA256).Hash.ToLowerInvariant() -ne '3eaa3763d09992358d21dd2d558eb5e40258dbfa9843df0a6d6c521707b3b3be'){throw 'Source transfer hash mismatch'}
New-Item -ItemType Directory $root | Out-Null
Expand-Archive (Join-Path $share 'source.zip') -DestinationPath $root
& 'C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe' (Join-Path $root 'scripts\sync_cu_live_windows.py') --verify $root
if($LASTEXITCODE -ne 0){throw 'Source manifest mismatch'}
New-Item -ItemType Directory (Join-Path $root 'build\isolated-config'),(Join-Path $root 'build\temp') -Force | Out-Null
foreach($name in @('build.cmd','run-build.ps1','probe.ps1')){Copy-Item (Join-Path $share $name) $root}
$tokens=$null;$errors=$null
[void][Management.Automation.Language.Parser]::ParseFile((Join-Path $root 'probe.ps1'),[ref]$tokens,[ref]$errors)
if($errors.Count){$errors | Format-List | Out-String | Write-Output;throw 'Probe parse errors'}
$p=Start-Process powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $root 'run-build.ps1')) -WindowStyle Hidden -PassThru
[pscustomobject]@{started_pid=$p.Id;root=$root;probe_parse_errors=0;session=(Get-Process -Id $PID).SessionId} | ConvertTo-Json
