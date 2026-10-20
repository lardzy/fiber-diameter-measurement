# Install only after a narrow FDM registration backup; preserve unrelated VM work.
[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$root='C:\tmp\fdm-cu-live-fourth-round-20261019'
$target="$root\dist\windows\installed-candidate"
if(Test-Path $target){throw 'Candidate installation directory already exists'}
$backup="$root\build\fourth-round\installation-backup\baseline.json"
if(-not(Test-Path $backup)){throw 'FDM registration backup is required'}
$saved=[IO.File]::ReadAllText($backup)|ConvertFrom-Json
if($saved.target -ne $target -or $saved.whole_vm_rollback){throw 'Unexpected backup scope'}
foreach($item in $saved.original_files){if((Get-FileHash $item.path).Hash.ToLowerInvariant() -ne $item.sha256){throw 'Original installation changed'}}
$identity=[IO.File]::ReadAllText("$root\build\fourth-round\package\installer-identity.json")|ConvertFrom-Json
if(-not $identity.passed -or $identity.version -ne '0.4.10'){throw 'Verified 0.4.10 candidate required'}
if((Get-FileHash $identity.installer).Hash.ToLowerInvariant() -ne $identity.sha256){throw 'Installer identity mismatch'}
$log="$root\build\fourth-round\actual-installation.log"
$arguments='/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /NOICONS /TASKS="" /DIR="'+$target+'" /LOG="'+$log+'"'
$process=[Diagnostics.Process]::new()
$process.StartInfo.FileName=$identity.installer
$process.StartInfo.Arguments=$arguments
$process.StartInfo.UseShellExecute=$false
[void]$process.Start()
if(-not $process.WaitForExit(600000)){throw 'Installer timeout; inspect owned process'}
$result=@{exit_code=$process.ExitCode;installed=$target;caller=[Security.Principal.WindowsIdentity]::GetCurrent().Name;arguments=$arguments;build_id=$identity.build_id;installer_sha256=$identity.sha256;backup=$backup;whole_vm_rollback=$false;passed=$false}
if($process.ExitCode -ne 0){throw 'Installer failed'}
$installed=Join-Path $target 'FiberDiameterMeasurement.exe'
if((Get-FileHash $installed).Hash -ne (Get-FileHash (Join-Path $identity.payload 'FiberDiameterMeasurement.exe')).Hash){throw 'Installed executable differs from payload'}
& (Join-Path $PSScriptRoot 'manage_acquisition_fourth_round_installation.ps1') -Action Capture
if($LASTEXITCODE -ne 0){throw 'Installed registration snapshot failed'}
$result.passed=$true
$text=$result|ConvertTo-Json -Depth 5
[IO.File]::WriteAllText("$root\build\fourth-round\actual-installation.json",$text,[Text.UTF8Encoding]::new($false))
[Console]::WriteLine($text)
