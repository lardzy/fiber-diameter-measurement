# Explorer starts a scoped helper so no existing user FDM configuration is read or overwritten.
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root='C:\tmp\fdm-cu-live-fourth-round-20261019'
$exe="$root\dist\windows\installed-candidate\FiberDiameterMeasurement.exe"
$out="$root\build\fourth-round\explorer-start"
if(Test-Path "$out\result.json"){throw 'Preserve the previous Explorer result'}
[void](New-Item -ItemType Directory -Force -Path "$out\profile\Local","$out\profile\Roaming")
$before=@(Get-CimInstance Win32_Process|Where-Object{$_.ExecutablePath -eq $exe})
if($before.Count){throw 'Candidate already running; ambiguous launch'}
$identity=[IO.File]::ReadAllText("$root\build\fourth-round\package\installer-identity.json")|ConvertFrom-Json
if((Get-FileHash $exe).Hash -ne (Get-FileHash (Join-Path $identity.payload 'FiberDiameterMeasurement.exe')).Hash){throw 'Candidate differs from verified payload'}
$helper="$out\launch-isolated.cmd"
$commands=@('@echo off','setlocal',('set "LOCALAPPDATA='+$out+'\profile\Local"'),('set "APPDATA='+$out+'\profile\Roaming"'),'set "QT_SCALE_FACTOR=0.5"','set "FDM_TEST_MICROVIEW_DLL="','set "FDM_MICROVIEW_DLL_DIR="',('start "" /wait "'+$exe+'"'),'exit /b %errorlevel%')
[IO.File]::WriteAllLines($helper,$commands,[Text.Encoding]::ASCII)
$result=[ordered]@{schema=2;executable=$exe;build_id=$identity.build_id;launch='Explorer -> isolated cmd helper -> installed executable';helper=$helper;helper_sha256=(Get-FileHash $helper).Hash.ToLowerInvariant();isolated_profile="$out\profile";user=[Security.Principal.WindowsIdentity]::GetCurrent().Name;elevated=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator);policy=[string](Get-ExecutionPolicy);started_at=(Get-Date).ToString('o');ok=$false;hardware_tested=$false;pid=$null;parent_name=$null;grandparent_name=$null;title=$null;error=$null;graceful_exit=$false;forced_cleanup=$false}
$owned=$null
try{
 Start-Process -FilePath "$env:WINDIR\explorer.exe" -ArgumentList $helper
 $deadline=[DateTime]::UtcNow.AddSeconds(45)
 do{Start-Sleep -Milliseconds 200;$matches=@(Get-CimInstance Win32_Process|Where-Object{$_.ExecutablePath -eq $exe})}while($matches.Count -eq 0 -and [DateTime]::UtcNow -lt $deadline)
 if($matches.Count -ne 1){throw 'Expected one owned installed candidate'}
 $record=$matches[0];$owned=[Diagnostics.Process]::GetProcessById($record.ProcessId);$handle=$owned.Handle;$result.pid=$owned.Id
 $parent=Get-CimInstance Win32_Process -Filter "ProcessId=$($record.ParentProcessId)"
 $result.parent_name=$parent.Name
 $result.grandparent_name=(Get-CimInstance Win32_Process -Filter "ProcessId=$($parent.ParentProcessId)").Name
 if($parent.Name -ne 'cmd.exe' -or -not $parent.CommandLine.Contains($helper) -or $result.grandparent_name -ne 'explorer.exe'){throw 'Unexpected Explorer helper parent chain'}
 do{Start-Sleep -Milliseconds 200;$owned.Refresh()}while(-not $owned.HasExited -and $owned.MainWindowHandle -eq 0 -and [DateTime]::UtcNow -lt $deadline)
 if($owned.HasExited -or $owned.MainWindowHandle -eq 0){throw 'Main window absent'}
 if($result.elevated){throw 'Expected ordinary user context'}
 $result.title=$owned.MainWindowTitle;$result.main_window_handle=$owned.MainWindowHandle.ToInt64()
 if(-not $owned.CloseMainWindow()){throw 'Cannot close owned window'}
 $result.graceful_exit=$owned.WaitForExit(12000)
 if(-not $result.graceful_exit){throw 'Untouched candidate did not close normally'}
 $result.exit_code=$owned.ExitCode
 if($result.exit_code -ne 0){throw 'Application error exit'}
 $result.ok=$true
}catch{$result.error=$_.Exception.Message}finally{
 if($owned){$owned.Refresh();if(-not $owned.HasExited){$owned.Kill();$result.forced_cleanup=$true;[void]$owned.WaitForExit(5000)};$owned.Dispose()}
 $result.finished_at=(Get-Date).ToString('o');$text=$result|ConvertTo-Json -Depth 8;[IO.File]::WriteAllText("$out\result.json",$text,[Text.UTF8Encoding]::new($false));[Console]::WriteLine($text)
}
if(-not $result.ok){exit 1}
