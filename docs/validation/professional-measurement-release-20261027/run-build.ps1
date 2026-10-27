$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$root='C:\tmp\fdm-professional-measurement-release-20261027'
$status=[ordered]@{started=[DateTime]::UtcNow.ToString('o');finished=$null;exit_code=$null;root=$root}
try {
 $command='"'+$root+'\build.cmd" > "'+$root+'\build\release-build.log" 2>&1'
 & cmd.exe /d /s /c $command
 $status.exit_code=$LASTEXITCODE
} catch {$status.exit_code=-1;$status.error=$_.Exception.Message}
$status.finished=[DateTime]::UtcNow.ToString('o')
[IO.File]::WriteAllText(($root+'\build\release-status.json'),($status|ConvertTo-Json -Depth 5))
exit $status.exit_code
