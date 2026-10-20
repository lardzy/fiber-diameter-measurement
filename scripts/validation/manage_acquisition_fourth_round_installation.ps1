# Narrow rollback for this candidate. Never snapshot/restore unrelated VM work.
[CmdletBinding()]
param([Parameter(Mandatory=$true)][ValidateSet('Backup','Capture','Restore')][string]$Action)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root='C:\tmp\fdm-cu-live-fourth-round-20261019'
$out="$root\build\fourth-round\installation-backup"
$target="$root\dist\windows\installed-candidate"
$old='C:\Users\lishuyang\Downloads\fdm-native-validation-installed-20261013'
$keys=@(
 'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{F0F6A8B5-4838-4DF0-B2C1-18F1D5AA4A66}_is1',
 'SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\{F0F6A8B5-4838-4DF0-B2C1-18F1D5AA4A66}_is1',
 'SOFTWARE\Classes\.fdmproj', 'SOFTWARE\Classes\.fdmslide',
 'SOFTWARE\Classes\LARD.FiberDiameterMeasurement.Project',
 'SOFTWARE\Classes\LARD.FiberDiameterMeasurement.DigitalSlide',
 'SOFTWARE\Classes\Applications\FiberDiameterMeasurement.exe'
)
function Read-Key([string]$path) {
 $key=[Microsoft.Win32.Registry]::LocalMachine.OpenSubKey($path)
 if(-not $key){return $null}
 try {
  $values=@(); foreach($name in @($key.GetValueNames()|Sort-Object)) {
   $values+=[ordered]@{name=$name;kind=[string]$key.GetValueKind($name);value=$key.GetValue($name,$null,[Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)}
  }
  $children=@(); foreach($name in @($key.GetSubKeyNames()|Sort-Object)) {$children+=[ordered]@{name=$name;value=(Read-Key ($path+'\'+$name))}}
  return [ordered]@{values=$values;children=$children}
 } finally {$key.Dispose()}
}
function Signature($value) {
 $text=if($null -eq $value){'null'}else{ConvertTo-Json -InputObject $value -Depth 40 -Compress}
 $hash=[Security.Cryptography.SHA256]::Create()
 try{return ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($text)))).Replace('-','').ToLowerInvariant()}finally{$hash.Dispose()}
}
function Inventory {
 $rows=@();foreach($key in $keys){$value=Read-Key $key;$rows+=@{key=$key;exists=($null -ne $value);signature=(Signature $value)}}
 return $rows
}
function Write-Json([string]$path,$value){[IO.File]::WriteAllText($path,($value|ConvertTo-Json -Depth 40),[Text.UTF8Encoding]::new($false))}
function Check-Old($hashes){foreach($item in $hashes){if((Get-FileHash -LiteralPath $item.path).Hash.ToLowerInvariant() -ne $item.sha256){throw ('Original install changed: '+$item.path)}}}
$manifest="$out\baseline.json"
if($Action -eq 'Backup'){
 if((Test-Path $out) -or (Test-Path $target)){throw 'Preserve an existing backup or installation; do not overwrite'}
 if(-not(Test-Path "$old\FiberDiameterMeasurement.exe")){throw 'Expected original installation absent'}
 [void](New-Item -ItemType Directory -Path $out)
 $rows=Inventory
 for($i=0;$i -lt $rows.Count;$i++){
  if($rows[$i].exists){$file="$out\key-$i.reg";& reg.exe export ('HKLM\'+$rows[$i].key) $file /y|Out-Null;if($LASTEXITCODE -ne 0){throw 'Registry export failed'};$rows[$i].file=$file;$rows[$i].file_sha256=(Get-FileHash $file).Hash.ToLowerInvariant()}
 }
 $hashes=@();foreach($name in @('FiberDiameterMeasurement.exe','release-manifest.json','unins000.exe','unins000.dat')){
  $file=Join-Path $old $name;if(Test-Path $file){$hashes+=@{path=$file;sha256=(Get-FileHash $file).Hash.ToLowerInvariant()}}
 }
 Write-Json $manifest @{schema=1;created_at=(Get-Date).ToString('o');target=$target;registry=$rows;original_files=$hashes;whole_vm_rollback=$false;scope='FDM app registration and owned candidate directory only'}
 [Console]::WriteLine($manifest);exit 0
}
if(-not(Test-Path $manifest)){throw 'Backup required'}
$baseline=[IO.File]::ReadAllText($manifest)|ConvertFrom-Json
if($baseline.target -ne $target -or $baseline.registry.Count -ne $keys.Count){throw 'Backup scope mismatch'}
for($i=0;$i -lt $keys.Count;$i++){if($baseline.registry[$i].key -ne $keys[$i]){throw 'Unexpected backup registry path'}}
Check-Old $baseline.original_files
$installed="$out\installed.json"
if($Action -eq 'Capture'){
 if(Test-Path $installed){throw 'Installation state already captured'}
 $registration=Get-ItemProperty ('Registry::HKEY_LOCAL_MACHINE\'+$keys[0])
 if($registration.InstallLocation.TrimEnd('\') -ne $target){throw 'Registration is not owned by this candidate'}
 Write-Json $installed @{registry=(Inventory);target=$target;executable_sha256=(Get-FileHash "$target\FiberDiameterMeasurement.exe").Hash.ToLowerInvariant()}
 exit 0
}
if(-not(Test-Path $installed)){throw 'Captured installed state required'}
$after=[IO.File]::ReadAllText($installed)|ConvertFrom-Json
if($after.target -ne $target -or $after.registry.Count -ne $keys.Count){throw 'Installed state scope mismatch'}
if((Get-FileHash "$target\FiberDiameterMeasurement.exe").Hash.ToLowerInvariant() -ne $after.executable_sha256){throw 'Candidate identity changed'}
$owned=@(Get-CimInstance Win32_Process|Where-Object{$_.ExecutablePath -and $_.ExecutablePath.StartsWith($target+'\',[StringComparison]::OrdinalIgnoreCase)})
if($owned.Count){throw 'Close only the candidate owned processes before restoring'}
$current=Inventory;$changed=@()
$preserved="$root\dist\windows\installed-candidate-validated"
if(Test-Path $preserved){throw 'Validated candidate preservation path exists'}
for($i=0;$i -lt $keys.Count;$i++){
 if($after.registry[$i].key -ne $keys[$i]){throw 'Unexpected captured registry path'}
 if($current[$i].signature -ne $after.registry[$i].signature){throw ('Registry changed after candidate install; inspect before rollback: '+$keys[$i])}
 if($baseline.registry[$i].signature -ne $current[$i].signature){
  if($baseline.registry[$i].exists){$item=$baseline.registry[$i];if((Get-FileHash $item.file).Hash.ToLowerInvariant() -ne $item.file_sha256){throw 'Registry backup changed'}}
  $changed+=$i
 }
}
# Restore only changed, identity-checked FDM keys, never any other registry path.
foreach($i in $changed){
 if($current[$i].exists){& reg.exe delete ('HKLM\'+$keys[$i]) /f|Out-Null;if($LASTEXITCODE -ne 0){throw 'Registry delete failed'}}
 if($baseline.registry[$i].exists){$item=$baseline.registry[$i];if((Get-FileHash $item.file).Hash.ToLowerInvariant() -ne $item.file_sha256){throw 'Registry backup changed'};& reg.exe import $item.file|Out-Null;if($LASTEXITCODE -ne 0){throw 'Registry restore failed'}}
}
$restored=Inventory
for($i=0;$i -lt $keys.Count;$i++){if($restored[$i].signature -ne $baseline.registry[$i].signature){throw 'Restored registration mismatch'}}
Check-Old $baseline.original_files
Move-Item -LiteralPath $target -Destination $preserved
Write-Json "$out\restored.json" @{passed=$true;changed_registry_indexes=$changed;restored_original_files=$baseline.original_files;candidate_preserved=$preserved;whole_vm_rollback=$false;uninstaller_run=$false;finished_at=(Get-Date).ToString('o')}
[Console]::WriteLine([IO.File]::ReadAllText("$out\restored.json"))
