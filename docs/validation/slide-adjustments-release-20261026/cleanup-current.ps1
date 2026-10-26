$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$r='C:\tmp\fdm-slide-adjustments-release-20261026'
$s='\\Mac\Home\Downloads\fdm-slide-adjustments-release-transfer-20261026'
$retained=Get-Content ($s+'\host-retained-installer.json') -Raw -Encoding UTF8|ConvertFrom-Json
$transfer=Get-Content ($s+'\transfer.json') -Raw -Encoding UTF8|ConvertFrom-Json
$source=Get-Content ($r+'\.fdm_snapshot.json') -Raw -Encoding UTF8|ConvertFrom-Json
if($source.head -ne '4417873881085c289e9f1e329b36904d85c77861'){throw 'Unexpected source root identity'}
if(-not $retained.verified -or $retained.sha256 -ne $transfer.sha256){throw 'Host retention receipt mismatch'}
if((Get-FileHash ($s+'\'+$transfer.installer) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $retained.sha256){throw 'Installer hash mismatch'}
if((Get-FileHash ($s+'\evidence.zip') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $transfer.evidence_sha256){throw 'Evidence archive hash mismatch'}
$gui=Get-ChildItem ($r+'\build\frozen-gui') -Filter result.json -Recurse|Sort-Object LastWriteTime -Descending|Select-Object -First 1
$g=Get-Content $gui.FullName -Raw -Encoding UTF8|ConvertFrom-Json
if(-not $g.passed -or -not $g.graceful_exit){throw 'Frozen GUI validation did not pass or process did not exit'}
if(@(Get-Process|Where-Object{$_.Path -like ($r+'\*')}).Count){throw 'An owned executable is still running'}
if((Get-Item $r -Force).Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Root is a reparse point'}
$items=@(Get-ChildItem $r -Recurse -Force)
if(@($items|Where-Object{$_.Attributes -band [IO.FileAttributes]::ReparsePoint}).Count){throw 'Reparse point present; inspect first'}
$files=@($items|Where-Object{-not $_.PSIsContainer})
$report=[ordered]@{root=$r;files=$files.Count;bytes=($files|Measure-Object Length -Sum).Sum;deleted=$false;retained_installer=$retained.path;sha256=$retained.sha256;evidence_sha256=$transfer.evidence_sha256;reason='Own temporary source, native build work, onedir, extracted payload and duplicate installer after verified archival';utc=[DateTime]::UtcNow.ToString('o')}
[IO.File]::WriteAllText(($s+'\windows-cleanup.json'),($report|ConvertTo-Json))
Remove-Item -LiteralPath $r -Recurse -Force
if(Test-Path $r){throw 'Temporary release directory remains'}
$report.deleted=$true
[IO.File]::WriteAllText(($s+'\windows-cleanup.json'),($report|ConvertTo-Json))
$report|ConvertTo-Json
