# PowerShell 5.1. Archive-only completion of a stopped read-only preflight.
# No deletion or device/application control is performed by this script.
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
function Utc { [DateTime]::UtcNow.ToString('o') }
function Write-Json($name,$value,[switch]$CreateNew) {
  $path=Join-Path $root $name
  $bytes=[Text.UTF8Encoding]::new($false).GetBytes((ConvertTo-Json -InputObject $value -Depth 14 -Compress))
  $mode=if($CreateNew){[IO.FileMode]::CreateNew}else{[IO.FileMode]::Create}
  $stream=[IO.File]::Open($path,$mode,[IO.FileAccess]::Write,[IO.FileShare]::None)
  try{$stream.Write($bytes,0,$bytes.Length);$stream.Flush($true)}finally{$stream.Dispose()}
}
function Progress($message) { Write-Json 'finish-preflight-progress.json' @{utc=(Utc);message=$message} }
function Records($objects) {
  $list=[Collections.Generic.List[VmCleanup.FileRecord]]::new()
  foreach($o in $objects){$list.Add([VmCleanup.FileRecord]@{
    Root=$o.Root;RelativePath=$o.RelativePath;FullPath=$o.FullPath;Length=[long]$o.Length
    LastWriteUtcTicks=[long]$o.LastWriteUtcTicks;Sha256=$o.Sha256;VolumeSerialNumber=[uint32]$o.VolumeSerialNumber
    FileIndexHigh=[uint32]$o.FileIndexHigh;FileIndexLow=[uint32]$o.FileIndexLow;ArchiveEntry=$o.ArchiveEntry
  })}
  return ,$list.ToArray()
}
function Content-Key($file) {
  if([string]$file.Sha256 -notmatch '^[0-9a-fA-F]{64}$' -or [long]$file.Length -lt 0){throw 'Invalid checkpoint content identity'}
  return ([string]$file.Sha256).ToLowerInvariant()+':'+([long]$file.Length).ToString([Globalization.CultureInfo]::InvariantCulture)
}
function Read-CompleteJson($name) {
  $path=Join-Path $root $name
  # Excluding write-sharing rejects an input which is still being written.
  $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
  try {
    $sha=[Security.Cryptography.SHA256]::Create()
    try{$hash=[BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-','').ToLowerInvariant()}finally{$sha.Dispose()}
    $stream.Position=0
    $reader=[IO.StreamReader]::new($stream,[Text.Encoding]::UTF8,$true,65536,$true)
    try{$data=$reader.ReadToEnd()|ConvertFrom-Json}finally{$reader.Dispose()}
  }finally{$stream.Dispose()}
  if([VmCleanup.CleanupCore]::Hash($path) -ne $hash){throw ('Input changed or resolved outside guarded path: '+$name)}
  return @{data=$data;sha256=$hash}
}

$owned=$false
try {
  Write-Json 'finish-preflight-started.json' @{pid=$PID;utc=(Utc)} -CreateNew
  $owned=$true
  foreach($name in @('preserved.zip','archive-mappings.json','plan.json','preflight-result.json','preflight-evidence.zip','finish-preflight-result.json')){
    if(Test-Path -LiteralPath (Join-Path $root $name)){throw ('Preserve existing output before finishing preflight: '+$name)}
  }
  Progress 'Loading guarded archive core and complete checkpoint inputs'
  Add-Type -Path (Join-Path $root 'CleanupCore.cs') -ReferencedAssemblies System.dll,System.Core.dll,System.Runtime.Serialization.dll,System.Xml.dll,System.IO.Compression.dll,System.IO.Compression.FileSystem.dll
  $checkpointRead=Read-CompleteJson 'candidate-checkpoint.json';$checkpoint=$checkpointRead.data
  $protectedRead=Read-CompleteJson 'protected-before.json';$protected=$protectedRead.data
  $guestRead=Read-CompleteJson 'guest-before.json'
  if(!$checkpoint.targets -or $null -eq $checkpoint.archive -or $null -eq $checkpoint.archive_reasons){throw 'Incomplete candidate checkpoint'}
  if(!$protected.files -or !$protected.roots -or !$protected.registry){throw 'Incomplete protected baseline'}
  if(!@($protected.registry|Where-Object{([string]$_.InstallLocation).TrimEnd('\') -eq 'C:\Users\lishuyang\Downloads\fdm-native-validation-installed-20261013'}).Count){throw 'Original installation registry baseline missing'}
  $archive=Records $checkpoint.archive
  if($archive.Count -ne 75073){throw ('Unexpected original archive path count: '+$archive.Count)}
  $targetFiles=@{};$targetRoots=@{};[long]$targetBytes=0
  foreach($target in $checkpoint.targets){
    if($targetRoots.ContainsKey($target.path)){throw 'Duplicate checkpoint target'}
    $targetRoots[$target.path]=$true;$targetBytes += [long]$target.bytes
    foreach($f in $target.files){
      if($targetFiles.ContainsKey($f.FullPath)){throw ('Duplicate checkpoint file: '+$f.FullPath)}
      $targetFiles[$f.FullPath]=$f
    }
  }
  foreach($p in $targetRoots.Keys){foreach($q in $targetRoots.Keys){if($p -ne $q -and $p.StartsWith($q.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Nested checkpoint targets'}}}
  $reasonByPath=@{}
  foreach($reason in $checkpoint.archive_reasons){
    if($reasonByPath.ContainsKey($reason.path)){throw 'Duplicate archive reason path'}
    $reasonByPath[$reason.path]=[string]$reason.reason
  }
  if($reasonByPath.Count -ne $archive.Count){throw 'Archive reasons do not cover every original resource'}
  $representatives=[Collections.Generic.List[VmCleanup.FileRecord]]::new()
  $representativeByContent=@{};$originalByPath=@{}
  [long]$originalBytes=0;[long]$uniqueBytes=0
  foreach($f in $archive){
    if($originalByPath.ContainsKey($f.FullPath)){throw ('Duplicate original archive path: '+$f.FullPath)}
    if(!$reasonByPath.ContainsKey($f.FullPath) -or !$targetFiles.ContainsKey($f.FullPath)){throw ('Archive resource is outside reviewed checkpoint: '+$f.FullPath)}
    $reviewed=$targetFiles[$f.FullPath]
    foreach($key in @('Root','RelativePath','FullPath','Length','LastWriteUtcTicks','Sha256','VolumeSerialNumber','FileIndexHigh','FileIndexLow')){
      if($f.$key -ne $reviewed.$key){throw ('Archive record differs from deletion checkpoint: '+$f.FullPath)}
    }
    $contentKey=Content-Key $f
    $originalByPath[$f.FullPath]=$f;$originalBytes += $f.Length
    if(!$representativeByContent.ContainsKey($contentKey)){
      $representativeByContent[$contentKey]=$f;$representatives.Add($f);$uniqueBytes += $f.Length
    }
  }
  # The first encountered original is the representative. Byte identity, not
  # names, determines deduplication; all source paths remain in the mapping.
  Progress ('Archiving '+$representatives.Count+' unique contents for '+$archive.Count+' original paths')
  $zipPath=Join-Path $root 'preserved.zip'
  [void][VmCleanup.CleanupCore]::ArchiveSelected($representatives.ToArray(),$zipPath)
  Progress 'Verifying unique ZIP entries and constructing complete restoration mappings'
  $verified=[VmCleanup.CleanupCore]::VerifyArchive($zipPath)
  if($verified.Files.Count -ne $representatives.Count -or $verified.TotalBytes -ne $uniqueBytes){throw 'Unique archive count or byte total mismatch'}
  $verifiedByContent=@{}
  foreach($f in $verified.Files){
    $key=Content-Key $f
    if($verifiedByContent.ContainsKey($key)){throw 'Duplicate content key in unique archive'}
    if(!$representativeByContent.ContainsKey($key) -or $representativeByContent[$key].FullPath -ne $f.FullPath){throw 'Archive representative identity mismatch'}
    $verifiedByContent[$key]=$f
  }
  $mappings=[Collections.Generic.List[object]]::new()
  foreach($f in $archive){
    $key=Content-Key $f
    if(!$verifiedByContent.ContainsKey($key)){throw ('Missing verified content for original path: '+$f.FullPath)}
    $representative=$verifiedByContent[$key]
    if(!$representative.ArchiveEntry){throw 'Verified ZIP entry mapping is missing'}
    $mappings.Add(@{original_full_path=$f.FullPath;original_root=$f.Root;original_relative_path=$f.RelativePath;original_last_write_utc_ticks=$f.LastWriteUtcTicks;representative_full_path=$representative.FullPath;archive_entry=$representative.ArchiveEntry;sha256=$representative.Sha256;length=$representative.Length;reason=$reasonByPath[$f.FullPath]})
  }
  if($mappings.Count -ne 75073){throw 'Restoration mapping lost original paths'}
  $archiveHash=[VmCleanup.CleanupCore]::Hash($zipPath)
  Write-Json 'archive-mappings.json' @{schema=1;created_utc=(Utc);method='sha256_and_length';source_checkpoint_sha256=$checkpointRead.sha256;archive_file='preserved.zip';archive_sha256=$archiveHash;original_path_count=$archive.Count;unique_content_count=$representatives.Count;original_logical_bytes=$originalBytes;unique_content_bytes=$uniqueBytes;entries=$mappings.ToArray()} -CreateNew
  $mappingsHash=[VmCleanup.CleanupCore]::Hash((Join-Path $root 'archive-mappings.json'))
  # No duplicate originals are reopened merely for deduplication. The checkpoint
  # supplies their content identities, and DeleteReviewed validates every source
  # again before deletion. Every mapped representative has just been byte-verified.
  foreach($inputRecord in @(@{name='candidate-checkpoint.json';hash=$checkpointRead.sha256},@{name='protected-before.json';hash=$protectedRead.sha256},@{name='guest-before.json';hash=$guestRead.sha256})){
    if([VmCleanup.CleanupCore]::Hash((Join-Path $root $inputRecord.name)) -ne $inputRecord.hash){throw ('Input changed while finishing preflight: '+$inputRecord.name)}
  }
  $protectedCount=@($protected.files).Count
  Write-Json 'plan.json' @{targets=$checkpoint.targets;skipped=$checkpoint.skipped;archive_reasons=$checkpoint.archive_reasons;archive_sha256=$archiveHash;archive_files=$archive.Count;archive_unique_files=$representatives.Count;archive_mappings_file='archive-mappings.json';archive_mappings_sha256=$mappingsHash;protected_files=$protectedCount;registry=$protected.registry;created_utc=(Utc);candidate_checkpoint_sha256=$checkpointRead.sha256} -CreateNew
  $planHash=[VmCleanup.CleanupCore]::Hash((Join-Path $root 'plan.json'))
  $result=@{status='ready_for_host_archive_verification';targets=@($checkpoint.targets).Count;skipped=$checkpoint.skipped;target_logical_bytes=$targetBytes;archive_files=$archive.Count;archive_original_paths=$archive.Count;archive_unique_files=$representatives.Count;archive_source_bytes=$originalBytes;archive_unique_source_bytes=$uniqueBytes;archive_bytes=([IO.FileInfo]::new($zipPath)).Length;archive_sha256=$archiveHash;archive_mappings_file='archive-mappings.json';archive_mappings_sha256=$mappingsHash;protected_files=$protectedCount;plan_sha256=$planHash;protected_sha256=$protectedRead.sha256;candidate_checkpoint_sha256=$checkpointRead.sha256;guest_before_sha256=$guestRead.sha256;deduplicated=$true}
  Write-Json 'preflight-result.json' $result -CreateNew
  Progress 'Packing plan, complete mappings, retained baseline and preflight receipts'
  $evidenceNames=@('archive-mappings.json','protected-before.json','guest-before.json','plan.json','preflight-result.json')
  $evidenceFiles=@($evidenceNames|ForEach-Object{Join-Path $root $_})
  $evidencePath=Join-Path $root 'preflight-evidence.zip'
  Compress-Archive -LiteralPath $evidenceFiles -DestinationPath $evidencePath -CompressionLevel Optimal
  # Verify all five metadata members against their exact saved files.
  $evidence=[IO.Compression.ZipFile]::OpenRead($evidencePath)
  try {
    if($evidence.Entries.Count -ne $evidenceNames.Count){throw 'Evidence ZIP member count mismatch'}
    foreach($name in $evidenceNames){
      $entry=$evidence.GetEntry($name)
      if(!$entry){throw ('Evidence member missing: '+$name)}
      $stream=$entry.Open();$sha=[Security.Cryptography.SHA256]::Create()
      try{$memberHash=[BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-','').ToLowerInvariant()}finally{$sha.Dispose();$stream.Dispose()}
      if($memberHash -ne [VmCleanup.CleanupCore]::Hash((Join-Path $root $name))){throw ('Evidence member differs: '+$name)}
    }
  }finally{$evidence.Dispose()}
  Write-Json 'finish-preflight-result.json' @{status='completed';utc=(Utc);original_paths=$archive.Count;unique_contents=$representatives.Count;mapping_paths=$mappings.Count;all_original_contents_reconstructable=$true;archive_sha256=$archiveHash;archive_mappings_sha256=$mappingsHash;plan_sha256=$planHash;protected_sha256=$protectedRead.sha256;evidence_zip_sha256=[VmCleanup.CleanupCore]::Hash($evidencePath);evidence_zip_bytes=([IO.FileInfo]::new($evidencePath)).Length;preflight_result_sha256=[VmCleanup.CleanupCore]::Hash((Join-Path $root 'preflight-result.json'));source_files_deleted=$false} -CreateNew
  Progress 'Completed; host must verify unique archive plus all original-path mappings before deletion'
}catch{
  if($owned -and !(Test-Path -LiteralPath (Join-Path $root 'finish-preflight-result.json'))){
    Write-Json 'finish-preflight-result.json' @{status='error';utc=(Utc);message=$_.Exception.ToString();source_files_deleted=$false} -CreateNew
  }
  throw
}
