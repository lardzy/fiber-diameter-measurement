$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
$out=Join-Path $root 'out'
[IO.Directory]::CreateDirectory($out)|Out-Null
$sid=(Get-CimInstance Win32_UserProfile | Where-Object {$_.LocalPath -eq 'C:\Users\lishuyang'} | Select-Object -First 1).SID
$apps=@(Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*','HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue|Where-Object {$_.DisplayName}|Select-Object DisplayName,DisplayVersion,Publisher,EstimatedSize,InstallLocation)
if($sid){$apps+=@(Get-ItemProperty ('Registry::HKEY_USERS\'+$sid+'\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*') -ErrorAction SilentlyContinue|Where-Object {$_.DisplayName}|Select-Object DisplayName,DisplayVersion,Publisher,EstimatedSize,InstallLocation)}
$os=Get-CimInstance Win32_OperatingSystem
$special=@('C:\pagefile.sys','C:\hiberfil.sys','C:\swapfile.sys','C:\Windows\MEMORY.DMP','C:\Windows.old'|ForEach-Object{Get-Item -LiteralPath $_ -Force -ErrorAction SilentlyContinue|Select-Object FullName,Length,Attributes,LastWriteTimeUtc})
$data=[ordered]@{utc=[DateTime]::UtcNow.ToString('o');identity=[Security.Principal.WindowsIdentity]::GetCurrent().Name;os=@{caption=$os.Caption;version=$os.Version;build=$os.BuildNumber};volumes=@(Get-CimInstance Win32_LogicalDisk|Select-Object DeviceID,DriveType,FileSystem,Size,FreeSpace,VolumeName);apps=$apps;special_files=$special;profiles=@(Get-CimInstance Win32_UserProfile|Select-Object LocalPath,Loaded,Special);user_shell_folders=(Get-ItemProperty ('Registry::HKEY_USERS\'+$sid+'\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders') -ErrorAction SilentlyContinue|Select-Object Desktop,Personal,'My Pictures','My Music','My Video','{374DE290-123F-4565-9164-39C4925E467B}');processes=@(Get-Process|Select-Object Id,ProcessName,Path);pagefiles=@(Get-CimInstance Win32_PageFileUsage|Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage)}
$data|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $out 'environment.json') -Encoding UTF8
& fsutil volume diskfree C: 2>&1|Out-File (Join-Path $out 'diskfree.txt') -Encoding UTF8
& fsutil behavior query DisableDeleteNotify 2>&1|Out-File (Join-Path $out 'trim-state.txt') -Encoding UTF8
& vssadmin list shadowstorage 2>&1|Out-File (Join-Path $out 'shadowstorage.txt') -Encoding UTF8
& vssadmin list shadows 2>&1|Out-File (Join-Path $out 'shadows.txt') -Encoding UTF8
& dism.exe /Online /Cleanup-Image /AnalyzeComponentStore /English /LogPath:$out\dism.log 2>&1|Out-File (Join-Path $out 'component-store.txt') -Encoding UTF8
$data.volumes|ConvertTo-Json
'Inventory complete; no cleanup command executed.'
