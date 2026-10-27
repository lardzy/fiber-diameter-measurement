param([string]$DsxRoot='', [string]$OutputRoot='')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
if(-not $OutputRoot){$OutputRoot=Join-Path $env:LOCALAPPDATA 'DSXEvidenceCollector\Basic'}
if(-not $DsxRoot){$DsxRoot=Read-Host 'DSX-BSW installation directory (empty is allowed)'}
$id=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')+'-'+[Guid]::NewGuid().ToString('N').Substring(0,8)
$out=Join-Path $OutputRoot ('basic-'+$id)
[IO.Directory]::CreateDirectory($out)|Out-Null
$gaps=New-Object Collections.Generic.List[object]
$files=New-Object Collections.Generic.List[object]
$procs=@()
$tcp=@()
try {
  $procs=@(Get-CimInstance Win32_Process -Property Name,ProcessId,ExecutablePath,CreationDate | Where-Object {$_.Name -like '*DSX*'} | Select-Object Name,ProcessId,ExecutablePath,CreationDate)
} catch {$gaps.Add([ordered]@{area='process';error=$_.Exception.Message})}
try {
  $ids=@($procs|ForEach-Object{$_.ProcessId})
  $tcp=@(Get-NetTCPConnection -ErrorAction Stop | Where-Object {$_.LocalPort -eq 35001 -or $_.RemotePort -eq 35001 -or $ids -contains $_.OwningProcess} | Select-Object LocalAddress,LocalPort,RemoteAddress,RemotePort,State,OwningProcess)
} catch {$gaps.Add([ordered]@{area='passive_tcp_table';error=$_.Exception.Message})}
$matched=$true
try {
  if(-not $DsxRoot){throw 'DSX root was not provided'}
  $baseline=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'baseline.json') -Raw -Encoding UTF8 | ConvertFrom-Json
  foreach($item in $baseline.files){
    $path=Join-Path $DsxRoot $item.path
    try {
      $hash=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
      $same=$hash -eq $item.sha256
      if(-not $same){$matched=$false}
      $files.Add([ordered]@{path=$item.path;sha256=$hash;expected=$item.sha256;matched=$same;version=(Get-Item -LiteralPath $path).VersionInfo.FileVersion})
    }catch{$matched=$false;$files.Add([ordered]@{path=$item.path;matched=$false;error=$_.Exception.Message})}
  }
} catch {$matched=$false;$gaps.Add([ordered]@{area='identity';error=$_.Exception.Message})}
$reg=Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
$net=Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full' -ErrorAction SilentlyContinue
$result=[ordered]@{schema=1;session_id=$id;utc=[DateTime]::UtcNow.ToString('o');scope='Basic passive inventory only; no GUI timeline or RDK experiments';os_build=$reg.CurrentBuildNumber;ubr=$reg.UBR;dotnet_release=$net.Release;powershell=$PSVersionTable.PSVersion.ToString();culture=[Globalization.CultureInfo]::CurrentCulture.Name;dsx_root=$DsxRoot;identity_matched=$matched;identities=$files;processes=$procs;tcp=$tcp;gaps=$gaps;network_connections_opened=$false;vendor_program_started=$false;process_command_lines_collected=$false}
$json=$result|ConvertTo-Json -Depth 8
[IO.File]::WriteAllText((Join-Path $out 'summary.json'),$json,[Text.UTF8Encoding]::new($false))
$hash=(Get-FileHash -LiteralPath (Join-Path $out 'summary.json')).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $out 'summary.sha256'),$hash+'  summary.json',[Text.UTF8Encoding]::new($false))
$zip=$out+'.zip'
Compress-Archive -LiteralPath @((Join-Path $out 'summary.json'),(Join-Path $out 'summary.sha256')) -DestinationPath $zip
Write-Host ('Saved: '+$out)
Write-Host 'This basic fallback does not connect to RDK or control a device.'
