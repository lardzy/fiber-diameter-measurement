$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
$out=Join-Path $root 'validation'
[IO.Directory]::CreateDirectory($out)|Out-Null
$exe=Join-Path $root 'dist\DsxEvidenceCollector.exe'
$run='run-'+[DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')
$caseRoot=Join-Path $out $run
[IO.Directory]::CreateDirectory($caseRoot)|Out-Null
$self=Start-Process -FilePath $exe -ArgumentList @('--self-test',(Join-Path $caseRoot 'tests')) -PassThru
if(-not $self.WaitForExit(60000)){Stop-Process -Id $self.Id;throw 'Owned self-test process exceeded 60 seconds.'}
$self.Refresh()
$selfExit=$self.ExitCode
$ui=Start-Process -FilePath $exe -ArgumentList @('--ui-smoke',(Join-Path $caseRoot 'ui')) -PassThru
if(-not $ui.WaitForExit(30000)){Stop-Process -Id $ui.Id;throw 'Owned UI smoke process exceeded 30 seconds.'}
$ui.Refresh()
$fallbackPassed=$false
try {
    $basic=Join-Path $caseRoot 'basic-fallback'
    & (Join-Path $root 'Collect-Basic.ps1') -DsxRoot (Join-Path $caseRoot 'missing-install') -OutputRoot $basic
    $summary=Get-ChildItem -LiteralPath $basic -Filter 'summary.json' -Recurse | Select-Object -First 1
    $data=Get-Content -LiteralPath $summary.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    if($data.identity_matched -or $data.network_connections_opened -or $data.process_command_lines_collected){throw 'Fallback boundary violation'}
    if($data.identities.Count -ne 34){throw 'Expected 34 missing identities'}
    $fallbackPassed=$true
} catch { $_.Exception.ToString() | Set-Content -LiteralPath (Join-Path $caseRoot 'fallback-error.txt') -Encoding UTF8 }
$admin=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$result=[ordered]@{run=$run;self_test_exit=$selfExit;ui_exit=$ui.ExitCode;fallback_passed=$fallbackPassed;validation_process_elevated=$admin;real_hardware_tested=$false;windows_1909_tested=$false;self_test_scope='Loopback random-port fake RDK only; temporary synthetic files/logs';ui_scope='Actual WinForms render in Windows 11 ARM64 VM, x64 process; no DSX installed/executed'}
$result|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $caseRoot 'validation-result.json') -Encoding UTF8
Copy-Item -LiteralPath (Join-Path $root 'dist') -Destination (Join-Path $caseRoot 'dist') -Recurse
$zip=Join-Path $root 'result.zip'
if(Test-Path -LiteralPath $zip){Remove-Item -LiteralPath $zip}
Compress-Archive -LiteralPath $caseRoot -DestinationPath $zip
$result|ConvertTo-Json
Get-Content -LiteralPath (Join-Path $caseRoot 'tests\self-tests.json') -Encoding UTF8
if($selfExit -ne 0 -or $ui.ExitCode -ne 0 -or -not $fallbackPassed){exit 1}
