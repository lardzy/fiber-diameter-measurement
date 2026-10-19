# SYSTEM installation inside the approved rollback checkpoint only.
$ErrorActionPreference='Stop'
$root='C:\tmp\fdm-cu-live-third-round-20261019'
$target="$root\dist\windows\installed-candidate"
if(Test-Path $target){throw 'Candidate installation directory already exists'}
$identity=[IO.File]::ReadAllText("$root\build\third-round\package\installer-identity.json")|ConvertFrom-Json
if((Get-FileHash $identity.installer).Hash.ToLowerInvariant() -ne $identity.sha256){throw 'Installer identity mismatch'}
$log="$root\build\third-round\actual-installation.log"
$args='/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /NOICONS /TASKS="" /DIR="'+$target+'" /LOG="'+$log+'"'
$p=[Diagnostics.Process]::new()
$p.StartInfo.FileName=$identity.installer
$p.StartInfo.Arguments=$args
$p.StartInfo.UseShellExecute=$false
[void]$p.Start()
if(-not $p.WaitForExit(600000)){throw 'Installer timeout; inspect owned process'}
$result=@{exit_code=$p.ExitCode;installed=$target;caller=[Security.Principal.WindowsIdentity]::GetCurrent().Name;arguments=$args;build_id=$identity.build_id;installer_sha256=$identity.sha256;checkpoint='{e780653e-feda-49b1-8191-dad59ca7f5a2}';passed=$false}
if($p.ExitCode -ne 0){throw 'Installer failed'}
$installed=Join-Path $target 'FiberDiameterMeasurement.exe'
if((Get-FileHash $installed).Hash -ne (Get-FileHash (Join-Path $identity.payload 'FiberDiameterMeasurement.exe')).Hash){throw 'Installed executable differs from payload'}
$result.passed=$true
$text=$result|ConvertTo-Json -Depth 5
[IO.File]::WriteAllText("$root\build\third-round\actual-installation.json",$text,[Text.UTF8Encoding]::new($false))
[Console]::WriteLine($text)
