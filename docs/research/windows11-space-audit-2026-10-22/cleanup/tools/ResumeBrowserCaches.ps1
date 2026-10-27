# PowerShell 5.1. One incident-specific resume; preserves all original receipts.
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
  Write-Json (Join-Path $root 'cache-resume-progress.json') @{utc=(Utc);phase=$phase;message=$message;path=$path}
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

$ExpectedPlanSha='1c1c606da24af1f525779415feac4c03e10fab364eb0020c92ca5317cfe00577'
$FirstPath='C:\Users\lishuyang\AppData\Local\Google\Chrome\User Data\Default\Cache'
$FirstManifestSha='3745a84bb43ce5c56db8242bf3ca4314ac4ed185480bece96c7b4c15dc8af5a8'
$rows=[Collections.Generic.List[object]]::new()
$before=$null;$fatalError=$null;$canWriteResult=$false;$sumTest=$null;$original=$null
$receiptHash=$null;$originalStartedHash=$null;$originalPid=0
$resultPath=Join-Path $root 'cache-resume-result.json'
$planPath=Join-Path $root 'cache-plan.json'
$receiptPath=Join-Path $root 'cache-result.json'
function Read-Json($path) { [IO.File]::ReadAllText($path)|ConvertFrom-Json }
function Assert-OriginalStopped {
  if($originalPid -le 0){throw 'Original PID missing'}
  if(@(Get-CimInstance Win32_Process -Filter ('ProcessId = '+$originalPid) -ErrorAction Stop).Count){throw 'Original PID is active; no resume permitted'}
}
function Assert-OriginalUnchanged {
  Assert-OriginalStopped
  if([VmCleanup.CleanupCore]::Hash($planPath) -ne $ExpectedPlanSha){throw 'Original plan changed'}
  if([VmCleanup.CleanupCore]::Hash($receiptPath) -ne $receiptHash){throw 'Original receipt changed'}
  if([VmCleanup.CleanupCore]::Hash((Join-Path $root 'cache-started.json')) -ne $originalStartedHash){throw 'Original start receipt changed'}
}
function Assert-Absent($path) {
  try{[void][IO.File]::GetAttributes($path)}
  catch{
    $cause=$_.Exception
    while($cause.InnerException){$cause=$cause.InnerException}
    if($cause -is [IO.FileNotFoundException] -or $cause -is [IO.DirectoryNotFoundException]){return}
    throw
  }
  throw 'First cache path exists or cannot be proven absent; this recovery does not delete it'
}
function Assert-UnchangedFiles($path,$expected) {
  $current=[VmCleanup.CleanupCore]::ListFiles($path)
  if($current.Count -ne $expected.Count){throw 'File count changed; skip target'}
  $byPath=@{}
  foreach($f in $current){$byPath[$f.FullPath]=$f}
  foreach($f in $expected){
    if(!$byPath.ContainsKey($f.FullPath)){throw 'File set changed; skip target'}
    $c=$byPath[$f.FullPath]
    foreach($key in @('Root','RelativePath','FullPath','Length','LastWriteUtcTicks','VolumeSerialNumber','FileIndexHigh','FileIndexLow')){
      if($c.$key -ne $f.$key){throw ('File identity changed; skip target: '+$f.FullPath)}
    }
    if([VmCleanup.CleanupCore]::Hash($f.FullPath) -ne $f.Sha256){throw ('File hash changed; skip target: '+$f.FullPath)}
  }
}
function Save-ResumeResult($status) {
  $after=$null;try{$after=Volume}catch{}
  $deleted=@($rows|Where-Object{$_.status -eq 'deleted'})
  $absent=@($rows|Where-Object{$_.status -eq 'verified_absent_after_receipt_failure'})
  $newBytes=Sum-HashtableBytes $deleted 'deleted_logical_bytes'
  $oldBytes=Sum-HashtableBytes $absent 'previously_absent_logical_bytes'
  Write-Json $resultPath @{schema=1;status=$status;utc=(Utc);before=$before;after=$after;original_before=$original.before;original_receipt_sha256=$receiptHash;plan_sha256=$ExpectedPlanSha;requested_paths=$targets.Count;deleted_paths=$deleted.Count;deleted_this_resume_logical_bytes=$newBytes;previously_absent_paths=$absent.Count;previously_absent_logical_bytes=$oldBytes;accounted_logical_bytes=($newBytes+$oldBytes);paths=$rows.ToArray();error=$fatalError;sum_self_test=$sumTest;original_receipts_preserved=$true;content_archived=$false;stopped_processes_or_services=$false}
}
try {
  Write-Json (Join-Path $root 'cache-resume-started.json') @{pid=$PID;utc=(Utc);script='ResumeBrowserCaches.ps1'} -CreateNew
  if(Test-Path -LiteralPath $resultPath){throw 'Existing resume result; preserve it and inspect'}
  $canWriteResult=$true
  $sumTest=Test-HashtableByteSum
  Cache-Progress 'loading' 'Compiling guarded cleanup core and checking original receipts'
  Add-Type -Path (Join-Path $root 'CleanupCore.cs') -ReferencedAssemblies System.dll,System.Core.dll,System.Runtime.Serialization.dll,System.Xml.dll,System.IO.Compression.dll,System.IO.Compression.FileSystem.dll
  $original=Read-Json $receiptPath
  if($original.status -ne 'manifests_persisted'){throw 'Original receipt is not the expected interrupted state'}
  if($original.plan_sha256 -ne $ExpectedPlanSha){throw 'Unexpected original plan identity'}
  if(([IO.File]::ReadAllText((Join-Path $root 'cache-plan.sha256'))).Trim() -ne $ExpectedPlanSha){throw 'Original plan hash sidecar mismatch'}
  $receiptHash=[VmCleanup.CleanupCore]::Hash($receiptPath)
  $originalStart=Read-Json (Join-Path $root 'cache-started.json')
  $originalPid=[int]$originalStart.pid
  $originalStartedHash=[VmCleanup.CleanupCore]::Hash((Join-Path $root 'cache-started.json'))
  Assert-OriginalUnchanged
  $plan=Read-Json $planPath
  if(@($plan.targets).Count -ne 9 -or @($original.paths).Count -ne 9){throw 'Unexpected target count'}
  $planByPath=@{};$originalByPath=@{}
  foreach($p in $plan.targets){if($planByPath.ContainsKey($p.path)){throw 'Duplicate original target'};$planByPath[$p.path]=$p}
  foreach($p in $original.paths){if($originalByPath.ContainsKey($p.path)){throw 'Duplicate original receipt target'};$originalByPath[$p.path]=$p}
  $readyCount=0
  foreach($allowed in $targets){
    if(!$planByPath.ContainsKey($allowed.path) -or !$originalByPath.ContainsKey($allowed.path)){throw 'Original target not in exact cache whitelist'}
    $p=$planByPath[$allowed.path];$r=$originalByPath[$allowed.path]
    foreach($key in @('path','group','status','manifest','manifest_sha256','files','logical_bytes')){if($p.$key -ne $r.$key){throw ('Original plan and receipt disagree: '+$allowed.path)}}
    if($p.group -ne $allowed.group){throw 'Cache ownership group mismatch'}
    if($p.status -eq 'ready'){
      $readyCount++
      if($p.manifest -notmatch '^cache-manifest-(01|02|03|09)\.json$'){throw 'Unexpected ready manifest'}
      $mp=Join-Path $root $p.manifest
      if([VmCleanup.CleanupCore]::Hash($mp) -ne $p.manifest_sha256){throw 'Original manifest hash mismatch'}
      $m=Read-Json $mp
      if($m.path -ne $p.path -or $m.group -ne $p.group -or $m.file_count -ne $p.files -or $m.logical_bytes -ne $p.logical_bytes){throw 'Original manifest metadata mismatch'}
    }elseif($p.status -ne 'skipped_busy_or_unknown'){throw 'Unexpected original nonready state'}
  }
  if($readyCount -ne 4){throw 'Unexpected ready target count'}
  $first=$planByPath[$FirstPath]
  if($first.status -ne 'ready' -or $first.logical_bytes -ne 134439874 -or $first.files -ne 448 -or $first.manifest_sha256 -ne $FirstManifestSha){throw 'First deleted cache identity mismatch'}
  $before=Volume
  Assert-Absent $FirstPath
  foreach($allowed in $targets){
    $p=$planByPath[$allowed.path]
    if($p.path -eq $FirstPath){$rows.Add(@{path=$p.path;group=$p.group;status='verified_absent_after_receipt_failure';previously_absent_logical_bytes=[long]134439874;files=448;manifest=$p.manifest;manifest_sha256=$p.manifest_sha256;observed_utc=(Utc);delete_attempted_in_resume=$false})}
    elseif($p.status -eq 'ready'){$rows.Add(@{path=$p.path;group=$p.group;status='ready';logical_bytes=[long]$p.logical_bytes;files=[int]$p.files;manifest=$p.manifest;manifest_sha256=$p.manifest_sha256})}
    else{$rows.Add(@{path=$p.path;group=$p.group;status='skipped_original_not_ready';original_status=$p.status;delete_attempted_in_resume=$false})}
  }
  Write-Json (Join-Path $root 'cache-resume-plan.json') @{schema=1;utc=(Utc);original_plan_sha256=$ExpectedPlanSha;original_receipt_sha256=$receiptHash;before=$before;targets=$rows.ToArray();sum_self_test=$sumTest} -CreateNew
  Save-ResumeResult 'original_receipts_verified'
  foreach($row in $rows){
    if($row.status -ne 'ready'){continue}
    $attempted=$false
    try {
      Cache-Progress 'rechecking' 'Checking original owner state, manifest and file identities' $row.path
      Assert-OriginalUnchanged
      $mp=Join-Path $root $row.manifest
      if([VmCleanup.CleanupCore]::Hash($mp) -ne $row.manifest_sha256){throw 'Original manifest changed'}
      $m=Read-Json $mp;$records=Records $m.files
      $gate=Activity $row.group;$row.precheck_activity=$gate
      if(!$gate.clear){$row.status='skipped_busy_or_unknown';continue}
      Assert-UnchangedFiles $row.path $records
      Cache-Progress 'delete_check' 'Rechecking owners immediately before guarded deletion' $row.path
      $gate=Activity $row.group;$row.delete_activity=$gate
      if(!$gate.clear){$row.status='skipped_busy_or_unknown';continue}
      $attempted=$true;$row.delete_attempted_in_resume=$true
      $deleted=[VmCleanup.CleanupCore]::DeleteReviewed($row.path,$records)
      $row.status='deleted';$row.deleted_logical_bytes=[long]$deleted.DeletedLogicalBytes
      $row.deleted_files=$deleted.DeletedFiles.Count;$row.root_still_exists=$deleted.RootStillExists
    }catch{
      $row.status=if($attempted){'failed_or_partial'}else{'skipped_changed_locked_or_unknown'}
      $row.message=$_.Exception.Message;$row.delete_attempted_in_resume=$attempted
    }finally{Save-ResumeResult 'in_progress'}
  }
  Assert-OriginalUnchanged
  $failures=@($rows|Where-Object{$_.status -eq 'failed_or_partial'}).Count
  $skipped=@($rows|Where-Object{$_.status -like 'skipped*'}).Count
  $status=if($failures){'completed_with_failures'}elseif($skipped){'completed_with_skips'}else{'completed'}
  Save-ResumeResult $status
  Cache-Progress $status 'Resume finished; original receipts were preserved'
}catch{
  $fatalError=$_.Exception.ToString()
  if($canWriteResult){Save-ResumeResult 'error';Cache-Progress 'error' 'Resume stopped; inspect cache-resume-result.json'}
  throw
}
