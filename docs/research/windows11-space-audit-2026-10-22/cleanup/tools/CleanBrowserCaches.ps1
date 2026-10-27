# PowerShell 5.1. Exact disposable cache scopes only; never stops an app or service.
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
function Utc { [DateTime]::UtcNow.ToString('o') }
function Write-Json($path,$value,[switch]$CreateNew) {
  $bytes=[Text.UTF8Encoding]::new($false).GetBytes((ConvertTo-Json -InputObject $value -Depth 12 -Compress))
  $mode=if($CreateNew){[IO.FileMode]::CreateNew}else{[IO.FileMode]::Create}
  $stream=[IO.File]::Open($path,$mode,[IO.FileAccess]::Write,[IO.FileShare]::None)
  try{$stream.Write($bytes,0,$bytes.Length);$stream.Flush($true)}finally{$stream.Dispose()}
}
function Volume { @{utc=(Utc);free_bytes=(Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'" -ErrorAction Stop).FreeSpace} }
function Sum-HashtableBytes($items,[string]$key) {
  [long]$total=0
  foreach($item in $items){
    if($item -isnot [Collections.IDictionary]){throw 'Byte sum requires dictionary records'}
    if($item.Contains($key) -and $null -ne $item[$key]){$total += [long]$item[$key]}
  }
  return $total
}
function Test-HashtableByteSum {
  $cases=@(
    @{items=@();expected=[long]0},
    @{items=@(@{deleted_logical_bytes=[long]134439874});expected=[long]134439874},
    @{items=@(@{deleted_logical_bytes=[long]134439874},@{deleted_logical_bytes=[long]515327209});expected=[long]649767083},
    @{items=@(@{status='skipped'},@{deleted_logical_bytes=$null});expected=[long]0},
    @{items=@(@{deleted_logical_bytes=[long]3000000000},@{deleted_logical_bytes=[long]3000000000});expected=[long]6000000000}
  )
  foreach($case in $cases){if((Sum-HashtableBytes $case.items 'deleted_logical_bytes') -ne $case.expected){throw 'Hashtable byte-sum self-test failed'}}
  return @{passed=$true;cases=$cases.Count;kind='in_memory_hashtables'}
}
function Cache-Progress($phase,$message,$path=$null) {
  Write-Json (Join-Path $root 'cache-progress.json') @{utc=(Utc);phase=$phase;message=$message;path=$path}
}
function Records($objects) {
  $list=[Collections.Generic.List[VmCleanup.FileRecord]]::new()
  foreach($o in $objects){$list.Add([VmCleanup.FileRecord]@{
    Root=$o.Root;RelativePath=$o.RelativePath;FullPath=$o.FullPath;Length=[long]$o.Length
    LastWriteUtcTicks=[long]$o.LastWriteUtcTicks;Sha256=$o.Sha256;VolumeSerialNumber=[uint32]$o.VolumeSerialNumber
    FileIndexHigh=[uint32]$o.FileIndexHigh;FileIndexLow=[uint32]$o.FileIndexLow;ArchiveEntry=$o.ArchiveEntry
  })}
  return ,$list.ToArray()
}
function Activity($group) {
  $busy=[Collections.Generic.List[object]]::new()
  try {
    foreach($p in Get-CimInstance Win32_Process -ErrorAction Stop){
      $name=[string]$p.Name;$image=[string]$p.ExecutablePath
      $google=($name -in @('chrome.exe','GoogleUpdater.exe','GoogleUpdate.exe','GoogleCrashHandler.exe','GoogleCrashHandler64.exe')) -or ($image -match '\\Google\\(Chrome|GoogleUpdater|Update)\\')
      # An updater with an unreadable path cannot be ruled out safely.
      $google=$google -or ($name -eq 'updater.exe' -and (!$image -or $image -match '\\Google\\'))
      $edge=($name -in @('msedge.exe','msedgewebview2.exe','MicrosoftEdgeUpdate.exe')) -or ($image -match '\\Microsoft\\(Edge|EdgeUpdate|EdgeWebView)\\')
      if((($group -in @('chrome','google_updater')) -and $google) -or ($group -eq 'edge' -and $edge)){
        $busy.Add(@{kind='process';pid=$p.ProcessId;name=$name;image=$image})
      }
    }
    foreach($s in Get-CimInstance Win32_Service -ErrorAction Stop){
      $identity=([string]$s.Name)+' '+([string]$s.DisplayName)+' '+([string]$s.PathName)
      $relevant=if($group -eq 'edge'){$identity -match 'MicrosoftEdgeUpdate|edgeupdate|\\Microsoft\\EdgeUpdate\\'}else{$identity -match 'GoogleUpdater|GoogleUpdate|gupdate|Google Update|\\Google\\(GoogleUpdater|Update)\\'}
      if($relevant -and $s.State -ne 'Stopped'){$busy.Add(@{kind='service';name=$s.Name;state=$s.State})}
    }
    foreach($t in Get-ScheduledTask -ErrorAction Stop){
      $identity=([string]$t.TaskPath)+' '+([string]$t.TaskName)+' '+(($t.Actions|ForEach-Object{$_.Execute}) -join ' ')
      $relevant=if($group -eq 'edge'){$identity -match 'MicrosoftEdgeUpdate|EdgeUpdate'}else{$identity -match 'GoogleUpdater|GoogleUpdate|Google Update|GoogleSystem'}
      if($relevant -and [string]$t.State -eq 'Running'){$busy.Add(@{kind='scheduled_task';name=$t.TaskName;state=[string]$t.State})}
    }
    return @{clear=($busy.Count -eq 0);utc=(Utc);busy=$busy.ToArray()}
  }catch{return @{clear=$false;utc=(Utc);reason='activity_inspection_failed';message=$_.Exception.Message;busy=$busy.ToArray()}}
}

$targets=@(
  @{group='chrome';path='C:\Users\lishuyang\AppData\Local\Google\Chrome\User Data\Default\Cache'},
  @{group='chrome';path='C:\Users\lishuyang\AppData\Local\Google\Chrome\User Data\Default\Code Cache'},
  @{group='chrome';path='C:\Users\lishuyang\AppData\Local\Google\Chrome\User Data\Default\GPUCache'},
  @{group='edge';path='C:\Users\lishuyang\AppData\Local\Microsoft\Edge\User Data\Default\Cache'},
  @{group='edge';path='C:\Users\lishuyang\AppData\Local\Microsoft\Edge\User Data\Default\Code Cache'},
  @{group='edge';path='C:\Users\lishuyang\AppData\Local\Microsoft\Edge\User Data\Default\GPUCache'},
  @{group='edge';path='C:\Users\lishuyang\AppData\Local\Microsoft\Edge\User Data\Guest Profile\Cache'},
  @{group='edge';path='C:\Users\lishuyang\AppData\Local\Microsoft\Edge\User Data\Guest Profile\Code Cache'},
  @{group='google_updater';path='C:\Program Files (x86)\Google\GoogleUpdater\crx_cache'}
)
$rows=[Collections.Generic.List[object]]::new()
$before=$null;$planHash=$null;$fatalError=$null;$canWriteResult=$false
$resultPath=Join-Path $root 'cache-result.json'
function Save-Result($status) {
  $after=$null;try{$after=Volume}catch{}
  $deleted=@($rows|Where-Object{$_.status -eq 'deleted'})
  $deletedBytes=Sum-HashtableBytes $deleted 'deleted_logical_bytes'
  Write-Json $resultPath @{schema=1;status=$status;utc=(Utc);before=$before;after=$after;plan_sha256=$planHash;requested_paths=$targets.Count;deleted_paths=$deleted.Count;deleted_logical_bytes=$deletedBytes;paths=$rows.ToArray();error=$fatalError;sum_self_test=$sumTest;content_archived=$false;stopped_processes_or_services=$false}
}

try {
  # The durable receipt prevents automatic reruns after partial completion.
  Write-Json (Join-Path $root 'cache-started.json') @{pid=$PID;utc=(Utc);script='CleanBrowserCaches.ps1'} -CreateNew
  if(Test-Path -LiteralPath $resultPath){throw 'Existing cache result; preserve it and inspect before another run'}
  $canWriteResult=$true
  $sumTest=Test-HashtableByteSum
  Cache-Progress 'loading' 'Compiling guarded cleanup core'
  Add-Type -Path (Join-Path $root 'CleanupCore.cs') -ReferencedAssemblies System.dll,System.Core.dll,System.Runtime.Serialization.dll,System.Xml.dll,System.IO.Compression.dll,System.IO.Compression.FileSystem.dll
  $before=Volume
  $scriptHash=[VmCleanup.CleanupCore]::Hash($MyInvocation.MyCommand.Path)
  $coreHash=[VmCleanup.CleanupCore]::Hash((Join-Path $root 'CleanupCore.cs'))
  $index=0
  # Complete and persist every target's reviewed manifest before any deletion.
  foreach($target in $targets){
    $index++
    $row=@{path=$target.path;group=$target.group;status='preflight';files=0;logical_bytes=0}
    $rows.Add($row)
    $manifestPath=Join-Path $root ('cache-manifest-{0:d2}.json' -f $index)
    try {
      Cache-Progress 'preflight' ('Checking cache '+$index+' of '+$targets.Count) $target.path
      $gate=Activity $target.group;$row.preflight_activity=$gate
      if(!$gate.clear){$row.status='skipped_busy_or_unknown'}
      elseif(!(Test-Path -LiteralPath $target.path -PathType Container)){$row.status='skipped_missing_or_inaccessible'}
      else {
        $files=[VmCleanup.CleanupCore]::ListFiles($target.path)
        $hashed=0
        foreach($f in $files){
          if(($hashed % 250) -eq 0){Cache-Progress 'hashing' ('Hashing '+$hashed+' of '+$files.Count+' files') $target.path}
          $f.Sha256=[VmCleanup.CleanupCore]::Hash($f.FullPath);$hashed++
        }
        $row.files=$files.Count;[long]$fileBytes=0
        foreach($f in $files){$fileBytes += [long]$f.Length}
        $row.logical_bytes=$fileBytes
        $row.status=if($files.Count){'ready'}else{'skipped_empty'}
        Write-Json $manifestPath @{schema=1;path=$target.path;group=$target.group;utc=(Utc);files=$files;file_count=$files.Count;logical_bytes=$row.logical_bytes;script_sha256=$scriptHash;core_sha256=$coreHash} -CreateNew
        $row.manifest=[IO.Path]::GetFileName($manifestPath)
        $row.manifest_sha256=[VmCleanup.CleanupCore]::Hash($manifestPath)
      }
    }catch{$row.status='skipped_preflight_error';$row.message=$_.Exception.Message}
  }
  $planPath=Join-Path $root 'cache-plan.json'
  Write-Json $planPath @{schema=1;utc=(Utc);before=$before;targets=$rows.ToArray();script_sha256=$scriptHash;core_sha256=$coreHash} -CreateNew
  $planHash=[VmCleanup.CleanupCore]::Hash($planPath)
  [IO.File]::WriteAllText((Join-Path $root 'cache-plan.sha256'),$planHash,[Text.UTF8Encoding]::new($false))
  Save-Result 'manifests_persisted'
  Cache-Progress 'manifests_persisted' 'All cache manifests and plan hashes saved before deletion'

  foreach($row in $rows){
    if($row.status -ne 'ready'){continue}
    try {
      Cache-Progress 'delete_check' 'Rechecking manifest and current owners' $row.path
      if([VmCleanup.CleanupCore]::Hash($planPath) -ne $planHash){throw 'Persisted cache plan changed'}
      $manifestPath=Join-Path $root $row.manifest
      if([VmCleanup.CleanupCore]::Hash($manifestPath) -ne $row.manifest_sha256){throw 'Persisted cache manifest changed'}
      $manifest=[IO.File]::ReadAllText($manifestPath)|ConvertFrom-Json
      if($manifest.path -ne $row.path){throw 'Manifest target mismatch'}
      $records=Records $manifest.files
      # This check is deliberately adjacent to DeleteReviewed; never stop an owner.
      $gate=Activity $row.group;$row.delete_activity=$gate
      if(!$gate.clear){$row.status='skipped_busy_or_unknown';continue}
      Cache-Progress 'deleting' 'Deleting only unchanged reviewed cache files' $row.path
      $deleted=[VmCleanup.CleanupCore]::DeleteReviewed($row.path,$records)
      $row.status='deleted';$row.deleted_logical_bytes=$deleted.DeletedLogicalBytes
      $row.deleted_files=$deleted.DeletedFiles.Count;$row.root_still_exists=$deleted.RootStillExists
    }catch{$row.status='failed_or_partial';$row.message=$_.Exception.Message}
    finally{Save-Result 'in_progress'}
  }
  $failures=@($rows|Where-Object{$_.status -eq 'failed_or_partial'}).Count
  $skipped=@($rows|Where-Object{$_.status -like 'skipped*'}).Count
  $status=if($failures){'completed_with_failures'}elseif($skipped){'completed_with_skips'}else{'completed'}
  Save-Result $status
  Cache-Progress $status 'Cache cleanup finished; inspect cache-result.json'
}catch{
  # Never overwrite an older result when refusing a repeated invocation.
  $fatalError=$_.Exception.ToString()
  if($canWriteResult){
    Save-Result 'error'
    Cache-Progress 'error' 'Cache cleanup stopped; inspect cache-result.json'
  }
  throw
}
