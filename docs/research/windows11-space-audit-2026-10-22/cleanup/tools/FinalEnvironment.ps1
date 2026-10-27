$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$plan=[IO.File]::ReadAllText((Join-Path $root 'plan.json'))|ConvertFrom-Json
$rows=@($plan.targets|ForEach-Object{@{path=$_.path;exists=([IO.Directory]::Exists($_.path) -or [IO.File]::Exists($_.path))}})
$disk=Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
$shadows=@(Get-CimInstance Win32_ShadowStorage|ForEach-Object{@{used_bytes=[long]$_.UsedSpace;allocated_bytes=[long]$_.AllocatedSpace;maximum_bytes=[long]$_.MaxSpace}})
$page=@(Get-CimInstance Win32_PageFileUsage|ForEach-Object{@{path=[string]$_.Name;allocated_mb=[int]$_.AllocatedBaseSize;current_mb=[int]$_.CurrentUsage}})
$python='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$smoke=@{python=$python;status='missing'}
if([IO.File]::Exists($python)){
  $output=& $python -B -c "import sys,numpy; from PySide6 import QtCore; print(sys.version.split()[0],numpy.__version__,QtCore.qVersion())" 2>&1
  $smoke=@{python=$python;exit_code=$LASTEXITCODE;status=if($LASTEXITCODE -eq 0){'passed'}else{'failed'};output=($output|ForEach-Object{[string]$_}) -join "`n";hardware_called=$false;bytecode_writes_disabled=$true}
}
$result=@{utc=[DateTime]::UtcNow.ToString('o');volume=@{size_bytes=[long]$disk.Size;free_bytes=[long]$disk.FreeSpace};target_presence=$rows;shadow_storage=$shadows;pagefile=$page;fdm_dependency_import=$smoke}
[IO.File]::WriteAllText((Join-Path $root 'final-environment.json'),(ConvertTo-Json -InputObject $result -Depth 8 -Compress),[Text.UTF8Encoding]::new($false))
