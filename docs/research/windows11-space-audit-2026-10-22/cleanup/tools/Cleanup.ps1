param([ValidateSet('preflight','delete','maintenance','verify')][string]$Stage)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
function Save-Json($name,$value) {
  $text=ConvertTo-Json -InputObject $value -Depth 10 -Compress
  [IO.File]::WriteAllText((Join-Path $root $name),$text,[Text.UTF8Encoding]::new($false))
}
function Progress($message) { Save-Json 'progress.json' ([ordered]@{stage=$Stage;utc=[DateTime]::UtcNow.ToString('o');message=$message}) }
function Read-Json($name) { return ([IO.File]::ReadAllText((Join-Path $root $name))|ConvertFrom-Json) }
function Volume { return [ordered]@{utc=[DateTime]::UtcNow.ToString('o');free_bytes=(Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace} }
function Registry-Identity {
  $sid=(Get-CimInstance Win32_UserProfile|Where-Object{$_.LocalPath -eq 'C:\Users\lishuyang'}|Select-Object -First 1).SID
  return @(Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*','HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',('Registry::HKEY_USERS\'+$sid+'\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*') -ErrorAction SilentlyContinue |
    Where-Object {$_.DisplayName -like '*测量工具*' -or $_.InstallLocation -like '*fdm-native-validation*'} |
    Select-Object DisplayName,DisplayVersion,InstallLocation)
}
function Assert-Inactive($paths) {
  foreach($p in Get-Process){$image=$null;try{$image=$p.Path}catch{}
    if($image){foreach($path in $paths){if($image.Equals($path,[StringComparison]::OrdinalIgnoreCase) -or $image.StartsWith($path.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw "Target is in use: $path"}}}
  }
  foreach($p in Get-CimInstance Win32_Process){
    if($p.Name -match '^(FiberDiameterMeasurement|FiberAreaWorker|FiberScreenshotTool|cl|link|ninja|msbuild)\.exe$'){throw ('Application or build is active: '+$p.Name)}
    if($p.Name -match '^python(w)?\.exe$' -and $p.CommandLine -match 'fiber-diameter-measurement|fdm-cu-'){throw 'FDM Python task is active'}
    foreach($path in $paths){if($p.CommandLine -and $p.CommandLine.IndexOf($path,[StringComparison]::OrdinalIgnoreCase) -ge 0){throw ('Candidate referenced by active process '+$p.ProcessId)}}
  }
}
function Under-Any($path,$roots) {
  foreach($candidate in $roots){if($path.Equals($candidate,[StringComparison]::OrdinalIgnoreCase) -or $path.StartsWith($candidate.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){return $true}}
  return $false
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
try {
  Add-Type -Path (Join-Path $root 'CleanupCore.cs') -ReferencedAssemblies System.dll,System.Core.dll,System.Runtime.Serialization.dll,System.Xml.dll,System.IO.Compression.dll,System.IO.Compression.FileSystem.dll
  if($Stage -eq 'preflight'){
    Progress 'Running isolated cleanup self-tests'
    & (Join-Path $root 'SelfTest.ps1')
    $config=Read-Json 'targets.json'
    Save-Json 'guest-before.json' (Volume)
    $registry=Registry-Identity
    if(!@($registry|Where-Object{([string]$_.InstallLocation).TrimEnd('\') -eq 'C:\Users\lishuyang\Downloads\fdm-native-validation-installed-20261013'}).Count){throw 'Expected registered original FDM installation not found'}
    Assert-Inactive @($config.generated.path+$config.duplicate_runtime.path)
    $reference=[VmCleanup.CleanupCore]::ListFiles($config.retained_runtime)
    [VmCleanup.CleanupCore]::HashRecords($reference)
    $referenceByPath=@{};$referenceByHash=@{}
    foreach($f in $reference){$referenceByPath[$f.RelativePath]=$f;$referenceByHash[$f.Sha256]=$f.FullPath}
    $targets=[Collections.Generic.List[object]]::new()
    $skipped=[Collections.Generic.List[object]]::new()
    $archive=[Collections.Generic.List[VmCleanup.FileRecord]]::new()
    $archiveReasons=[Collections.Generic.List[object]]::new()
    $generatedBinary=@('.exe','.dll','.pyd','.lib','.obj','.o','.pdb','.pch','.pyc','.pkg','.pyz','.a','.exp')
    $checkpoint=Join-Path $root 'candidate-checkpoint.json'
    if([IO.File]::Exists($checkpoint)){
      $saved=Read-Json 'candidate-checkpoint.json'
      foreach($item in $saved.targets){$targets.Add($item)}
      foreach($item in $saved.skipped){$skipped.Add($item)}
      foreach($item in (Records $saved.archive)){$archive.Add($item)}
      foreach($item in $saved.archive_reasons){$archiveReasons.Add($item)}
    }else{
    foreach($target in $config.generated){
      Progress ('Hashing and classifying '+$target.path)
      $files=[VmCleanup.CleanupCore]::ListFiles($target.path)
      $bytes=($files|Measure-Object Length -Sum).Sum
      if($bytes -ne $target.logical_bytes -or $files.Count -ne $target.files){$skipped.Add(@{path=$target.path;reason='Size or file count changed since audit'});continue}
      [VmCleanup.CleanupCore]::HashRecords($files)
      $manifestHashes=@{}
      foreach($m in $files|Where-Object{[IO.Path]::GetFileName($_.FullPath) -eq 'release-manifest.json'}){
        $document=[IO.File]::ReadAllText($m.FullPath)|ConvertFrom-Json
        $folder=[IO.Path]::GetDirectoryName($m.FullPath)
        foreach($row in $document.files){if($row.sha256){$full=[IO.Path]::GetFullPath((Join-Path $folder ($row.path.Replace('/','\'))));if(!$full.StartsWith($folder+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Manifest path escaped package'};$manifestHashes[$full]=[string]$row.sha256}}
      }
      $modelDuplicates=[Collections.Generic.List[object]]::new()
      foreach($f in $files){
        $ext=[IO.Path]::GetExtension($f.FullPath).ToLowerInvariant()
        $reason='configuration_resource_source_or_diagnostic'
        $preserve=$true
        if($generatedBinary -contains $ext -or [IO.Path]::GetFileName($f.FullPath) -eq 'base_library.zip'){$preserve=$false}
        if($f.FullPath -match '\\(runtime|resources|camera|area-infer|caching|content-templates)\\'){$preserve=$true;$reason='runtime_or_editable_resource'}
        if($manifestHashes.ContainsKey($f.FullPath) -and $manifestHashes[$f.FullPath] -ne $f.Sha256){$preserve=$true;$reason='differs_from_release_manifest'}
        if($f.Length -gt 16MB -and $ext -in @('.pth','.pt','.onnx','.bin') -and $referenceByHash.ContainsKey($f.Sha256)){
          $preserve=$false;$modelDuplicates.Add(@{source=$f.FullPath;retained=$referenceByHash[$f.Sha256];sha256=$f.Sha256;bytes=$f.Length})
        }
        if($preserve){$archive.Add($f);$archiveReasons.Add(@{path=$f.FullPath;reason=$reason})}
      }
      $targets.Add(@{path=$target.path;category=$target.category;bytes=$bytes;files=$files;model_duplicates=$modelDuplicates.ToArray()})
    }
    foreach($target in $config.duplicate_runtime){
      Progress ('Comparing duplicate runtime '+$target.path)
      $files=[VmCleanup.CleanupCore]::ListFiles($target.path);$differences=[Collections.Generic.List[string]]::new()
      [VmCleanup.CleanupCore]::HashRecords($files)
      foreach($f in $files){
        if(!$referenceByPath.ContainsKey($f.RelativePath) -or $referenceByPath[$f.RelativePath].Sha256 -ne $f.Sha256){$differences.Add($f.RelativePath)}
      }
      if($differences.Count){$skipped.Add(@{path=$target.path;reason='Runtime contains unmatched files; retained whole tree';differences=$differences.ToArray()})}
      else{$targets.Add(@{path=$target.path;category='runtime_content_subset_verified';bytes=($files|Measure-Object Length -Sum).Sum;files=$files;retained=$config.retained_runtime})}
    }
    Save-Json 'candidate-checkpoint.json' @{targets=$targets.ToArray();skipped=$skipped.ToArray();archive=$archive.ToArray();archive_reasons=$archiveReasons.ToArray()}
    }
    $removalPaths=@($targets|ForEach-Object{$_.path})
    $protectedRoots=@('C:\Users\lishuyang\Downloads\fiber-diameter-measurement','C:\Users\lishuyang\Downloads\fdm-native-validation-installed-20261013','C:\Users\lishuyang\Downloads\flat-field-grating','C:\tmp\dsx-evidence-collector-20261022')
    $protectedRoots+=@([IO.Directory]::GetDirectories('C:\tmp','fdm-cu-*'))
    $protected=[Collections.Generic.List[object]]::new()
    $protectedLinks=[Collections.Generic.List[string]]::new()
    foreach($path in $protectedRoots){
      Progress ('Fingerprinting retained tree '+$path)
      $scan=[VmCleanup.CleanupCore]::ListRetainedFiles($path,[string[]]$removalPaths)
      [VmCleanup.CleanupCore]::HashRecords($scan.Files)
      foreach($f in $scan.Files){$protected.Add($f)}
      foreach($link in $scan.ReparsePoints){$protectedLinks.Add($link)}
    }
    Save-Json 'protected-before.json' @{roots=$protectedRoots;registry=$registry;files=$protected.ToArray();skipped_reparse_points=$protectedLinks.ToArray()}
    Progress 'Archiving configuration, templates, images and diagnostics'
    $archiveManifest=[VmCleanup.CleanupCore]::ArchiveSelected($archive.ToArray(),(Join-Path $root 'preserved.zip'))
    $verified=[VmCleanup.CleanupCore]::VerifyArchive((Join-Path $root 'preserved.zip'))
    $archiveHash=[VmCleanup.CleanupCore]::Hash((Join-Path $root 'preserved.zip'))
    Save-Json 'plan.json' @{targets=$targets.ToArray();skipped=$skipped.ToArray();archive_reasons=$archiveReasons.ToArray();archive_sha256=$archiveHash;protected_files=$protected.Count;registry=$registry;created_utc=[DateTime]::UtcNow.ToString('o')}
    $targetBytes=[long]0;foreach($item in $targets){$targetBytes += [long]$item.bytes}
    Save-Json 'preflight-result.json' @{status='ready_for_host_archive_verification';targets=$targets.Count;skipped=$skipped.ToArray();target_logical_bytes=$targetBytes;archive_files=$archive.Count;archive_bytes=([IO.FileInfo]::new((Join-Path $root 'preserved.zip'))).Length;archive_sha256=$archiveHash;protected_files=$protected.Count;plan_sha256=[VmCleanup.CleanupCore]::Hash((Join-Path $root 'plan.json'));protected_sha256=[VmCleanup.CleanupCore]::Hash((Join-Path $root 'protected-before.json'))}
    Compress-Archive -LiteralPath @((Join-Path $root 'plan.json'),(Join-Path $root 'protected-before.json'),(Join-Path $root 'guest-before.json'),(Join-Path $root 'preflight-result.json')) -DestinationPath (Join-Path $root 'preflight-evidence.zip')
  }
  elseif($Stage -eq 'delete'){
    $plan=Read-Json 'plan.json'
    $hostReceipt=Read-Json 'host-plan-verified.json'
    if([VmCleanup.CleanupCore]::Hash((Join-Path $root 'plan.json')) -ne $hostReceipt.plan_sha256 -or [VmCleanup.CleanupCore]::Hash((Join-Path $root 'protected-before.json')) -ne $hostReceipt.protected_sha256){throw 'Reviewed plan or protected baseline changed'}
    if($plan.archive_mappings_file){
      if($plan.archive_mappings_file -ne 'archive-mappings.json' -or [VmCleanup.CleanupCore]::Hash((Join-Path $root 'archive-mappings.json')) -ne $plan.archive_mappings_sha256 -or $plan.archive_mappings_sha256 -ne $hostReceipt.archive_mappings_sha256){throw 'Host-verified archive mappings changed'}
    }
    $ack=[IO.File]::ReadAllText((Join-Path $root 'host-archive-verified.sha256')).Trim()
    if($ack -ne $plan.archive_sha256 -or [VmCleanup.CleanupCore]::Hash((Join-Path $root 'preserved.zip')) -ne $ack){throw 'Host archive verification missing or mismatched'}
    [void][VmCleanup.CleanupCore]::VerifyArchive((Join-Path $root 'preserved.zip'))
    Assert-Inactive @($plan.targets.path)
    if((ConvertTo-Json -InputObject (Registry-Identity) -Compress) -ne (ConvertTo-Json -InputObject $plan.registry -Compress)){throw 'Registered FDM installation changed'}
    $baseline=Read-Json 'protected-before.json'
    Progress 'Checking all retained source and original installation identities and hashes before deletion'
    [VmCleanup.CleanupCore]::ValidateRecords((Records $baseline.files))
    $result=[Collections.Generic.List[object]]::new();$before=Volume
    foreach($target in $plan.targets){
      Progress ('Deleting reviewed exact target '+$target.path)
      try{$out=[VmCleanup.CleanupCore]::DeleteReviewed($target.path,(Records $target.files));$result.Add(@{path=$target.path;status='deleted';logical_bytes=$target.bytes;result=$out})}
      catch{$result.Add(@{path=$target.path;status='failed_or_partial';message=$_.Exception.Message})}
      Save-Json 'delete-progress.json' $result.ToArray()
    }
    Save-Json 'delete-result.json' @{status='completed';before=$before;after=(Volume);paths=$result.ToArray();archive_sha256=$ack}
  }
  elseif($Stage -eq 'maintenance'){
    $out=[Collections.Generic.List[object]]::new()
    $maintenanceBefore=Volume
    function Invoke-MaintenanceNative([string]$action,[string]$file,[string[]]$arguments,[string]$logName,[int[]]$successCodes=@(0)) {
      # Native stderr is diagnostic output, not a PowerShell terminating error.
      # Separate native redirection also avoids relying on pipeline LASTEXITCODE.
      $stdout=Join-Path $root ($logName+'.stdout')
      $stderr=Join-Path $root ($logName+'.stderr')
      $started=[DateTime]::UtcNow.ToString('o')
      $process=Start-Process -FilePath $file -ArgumentList $arguments -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
      $code=$process.ExitCode
      if($null -eq $code){throw ('Native process exit code unavailable: '+$action)}
      $code=[int]$code
      $text=[IO.File]::ReadAllText($stdout)
      $errorText=[IO.File]::ReadAllText($stderr)
      if($errorText){$text+="`r`n[stderr]`r`n"+$errorText}
      [IO.File]::WriteAllText((Join-Path $root $logName),$text,[Text.UTF8Encoding]::new($false))
      $row=@{action=$action;exit_code=$code;success=($successCodes -contains $code);restart_required=($code -eq 3010);started_utc=$started;finished_utc=[DateTime]::UtcNow.ToString('o');log=$logName}
      $out.Add($row)
      Save-Json 'maintenance-progress.json' @{actions=$out.ToArray();before=$maintenanceBefore}
      return $row
    }
    $uv='C:\Users\lishuyang\.local\bin\uv.exe'
    if(Test-Path -LiteralPath $uv){
      Progress 'Pruning unused uv cache, offline'
      [void](Invoke-MaintenanceNative 'uv cache prune --offline' $uv @('cache','prune','--cache-dir','C:\Users\lishuyang\AppData\Local\uv\cache','--offline') 'uv-prune.txt')
    }else{$out.Add(@{action='uv cache prune';status='executable_not_found'})}
    $npm='C:\Program Files\nodejs\npm.cmd'
    if(Test-Path -LiteralPath $npm){
      Progress 'Verifying and garbage collecting npm cache, offline'
      $npmCommand='/d /s /c ""'+$npm+'" cache verify --cache "C:\Users\lishuyang\AppData\Local\npm-cache" --offline"'
      [void](Invoke-MaintenanceNative 'npm cache verify --offline' (Join-Path $env:SystemRoot 'System32\cmd.exe') @($npmCommand) 'npm-verify.txt')
    }else{$out.Add(@{action='npm cache verify';status='executable_not_found'})}
    $dism=Join-Path $env:SystemRoot 'System32\dism.exe'
    Progress 'Analyzing Windows component store before cleanup'
    $analysis=Invoke-MaintenanceNative 'DISM AnalyzeComponentStore before' $dism @('/Online','/Cleanup-Image','/AnalyzeComponentStore','/English',('/LogPath:"'+(Join-Path $root 'dism-analyze-before.log')+'"')) 'system-before.txt' @(0,3010)
    if($analysis.exit_code -eq 0){
      Progress 'Windows component cleanup, no ResetBase and no forced reboot'
      $cleanup=Invoke-MaintenanceNative 'DISM StartComponentCleanup NoRestart' $dism @('/Online','/Cleanup-Image','/StartComponentCleanup','/NoRestart','/English',('/LogPath:"'+(Join-Path $root 'dism-maintenance.log')+'"')) 'dism-cleanup.txt' @(0,3010)
      if($cleanup.exit_code -eq 0){
        Progress 'Analyzing Windows component store after successful cleanup'
        [void](Invoke-MaintenanceNative 'DISM AnalyzeComponentStore after' $dism @('/Online','/Cleanup-Image','/AnalyzeComponentStore','/English',('/LogPath:"'+(Join-Path $root 'dism-analyze-after.log')+'"')) 'system-after.txt' @(0,3010))
      }else{
        $reason=if($cleanup.exit_code -eq 3010){'cleanup_requires_restart'}else{'cleanup_failed'}
        $out.Add(@{action='DISM AnalyzeComponentStore after';status='skipped';reason=$reason})
        Progress ('Post-cleanup analysis skipped: '+$reason)
      }
    }else{
      $reason=if($analysis.exit_code -eq 3010){'before_analysis_requires_restart'}else{'before_analysis_failed'}
      $out.Add(@{action='DISM StartComponentCleanup NoRestart';status='skipped';reason=$reason})
      $out.Add(@{action='DISM AnalyzeComponentStore after';status='skipped';reason=$reason})
    }
    $failed=@($out|Where-Object{$_.ContainsKey('success') -and !$_.success}).Count
    $restartRequired=@($out|Where-Object{$_.restart_required}).Count -gt 0
    $status=if($failed){'completed_with_failures'}elseif($restartRequired){'completed_restart_required'}else{'completed'}
    Save-Json 'maintenance-result.json' @{status=$status;actions=$out.ToArray();restart_required=$restartRequired;failed_actions=$failed;before=$maintenanceBefore;after=(Volume)}
  }
  elseif($Stage -eq 'verify'){
    $baseline=Read-Json 'protected-before.json';$differences=[Collections.Generic.List[object]]::new();$count=0
    Progress 'Verifying all retained source and original installation identities and hashes'
    try{[VmCleanup.CleanupCore]::ValidateRecords((Records $baseline.files));$count=$baseline.files.Count}
    catch{
      foreach($f in $baseline.files){
        try{if([VmCleanup.CleanupCore]::Hash($f.FullPath) -ne $f.Sha256){$differences.Add(@{path=$f.FullPath;reason='hash_changed'})}}
        catch{$differences.Add(@{path=$f.FullPath;reason=$_.Exception.Message})};$count++
      }
      if(!$differences.Count){$differences.Add(@{reason='identity_or_metadata_changed_despite_matching_content';message=$_.Exception.Message})}
    }
    $registryMatch=(ConvertTo-Json -InputObject (Registry-Identity) -Compress) -eq (ConvertTo-Json -InputObject $baseline.registry -Compress)
    Save-Json 'guest-after.json' (Volume)
    Save-Json 'verify-result.json' @{status=if($differences.Count -eq 0 -and $registryMatch){'passed'}else{'differences_found'};files_checked=$count;differences=$differences.ToArray();registered_installation_unchanged=$registryMatch;after=(Volume)}
  }
}
catch {
  Save-Json ($Stage+'-result.json') @{status='error';message=$_.Exception.ToString();position=$_.InvocationInfo.PositionMessage;utc=[DateTime]::UtcNow.ToString('o')}
  throw
}
