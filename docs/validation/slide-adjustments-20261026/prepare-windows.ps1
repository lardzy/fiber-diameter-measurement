$ErrorActionPreference='Stop'
$root='C:\tmp\fdm-slide-adjustments-20261026\r2'
$transfer='\\Mac\Home\Downloads\fdm-slide-adjustments-transfer-20261026'
if(Test-Path $root){throw 'Test root already exists; use a new round.'}
New-Item -ItemType Directory -Path $root|Out-Null
Expand-Archive -LiteralPath ($transfer+'\source.zip') -DestinationPath $root
$manifest=Get-Content -Raw -Encoding UTF8 ($root+'\source-manifest.json')|ConvertFrom-Json
foreach($f in $manifest.files){if((Get-FileHash -Algorithm SHA256 ($root+'\'+$f.path)).Hash.ToLower() -ne $f.sha256){throw ('Source mismatch '+$f.path)}}
$nativeRoot='C:\tmp\fdm-cu-live-eighth-round-20261021'
$nativeSource=@()
foreach($f in $manifest.files|Where-Object{$_.path -like 'native/*'}){
 $oldHash=(Get-FileHash -Algorithm SHA256 ($nativeRoot+'\'+$f.path)).Hash.ToLower()
 if($oldHash -ne $f.sha256){throw ('Native source mismatch '+$f.path)}
 $nativeSource+=@{path=$f.path;sha256=$oldHash}
}
Copy-Item -LiteralPath ($nativeRoot+'\src\fdm\_native\_kernels.cp312-win_amd64.pyd') -Destination ($root+'\src\fdm\_native')
Copy-Item -LiteralPath ($nativeRoot+'\src\fdm\_native\_microview.cp312-win_amd64.pyd') -Destination ($root+'\src\fdm\_native')
$native=@(Get-ChildItem ($root+'\src\fdm\_native\*.pyd')|ForEach-Object{@{name=$_.Name;sha256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()}})
[IO.File]::WriteAllText(($root+'\native-reuse.json'),(@{source=$nativeRoot;source_matches=$nativeSource;extensions=$native}|ConvertTo-Json -Depth 8))
Copy-Item -LiteralPath ($transfer+'\run-windows.ps1') -Destination $root
Write-Output $root
