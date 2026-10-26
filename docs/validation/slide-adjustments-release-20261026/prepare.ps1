$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$root='C:\tmp\fdm-slide-adjustments-release-20261026'
$share='\\Mac\Home\Downloads\fdm-slide-adjustments-release-transfer-20261026'
if(Test-Path $root){throw 'Fresh release root already exists'}
if((Get-FileHash ($share+'\source.zip') -Algorithm SHA256).Hash.ToLowerInvariant() -ne '8b0b4ab0f0a2e5dca937aea2e7c9f83d290d38d05528bec5ea0328ec6742d881'){throw 'Source transfer hash mismatch'}
New-Item -ItemType Directory $root|Out-Null
Expand-Archive ($share+'\source.zip') -DestinationPath $root
& 'C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe' ($root+'\scripts\sync_cu_live_windows.py') --verify $root
if($LASTEXITCODE -ne 0){throw 'Source manifest mismatch'}
New-Item -ItemType Directory ($root+'\build\isolated-config'),($root+'\build\temp') -Force|Out-Null
foreach($name in @('build.cmd','run-build.ps1','probe.ps1')){Copy-Item ($share+'\'+$name) $root}
$tokens=$null;$errors=$null
[void][Management.Automation.Language.Parser]::ParseFile(($root+'\probe.ps1'),[ref]$tokens,[ref]$errors)
if($errors.Count){$errors|Format-List|Out-String|Write-Output;throw 'Probe parse errors'}
[pscustomobject]@{root=$root;probe_parse_errors=0;session=(Get-Process -Id $PID).SessionId;free_bytes=(Get-PSDrive C).Free}|ConvertTo-Json
