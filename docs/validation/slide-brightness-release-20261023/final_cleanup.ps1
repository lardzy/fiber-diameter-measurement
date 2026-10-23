$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$r='C:\tmp\fdm-brightness-release-20261023'
$s='\\Mac\Home\Downloads\fdm-brightness-release-transfer-20261023'
$retained=Get-Content ($s+'\host-retained-installer.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$final=$s+'\FDM-0.4.11-slide-brightness-487f083d0ba1.exe'
$expected='2c6f550749f7a21a3f0399086c1c7b1e695dc6ceac2bcd7797a297d1c0544822'
if(-not $retained.verified -or $retained.sha256 -ne $expected){throw 'Host retention receipt mismatch'}
if((Get-FileHash -LiteralPath $final -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected){throw 'Final host installer hash mismatch'}
$transfer=Get-Content ($s+'\transfer.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if((Get-FileHash ($s+'\evidence.zip') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $transfer.evidence_sha256){throw 'Archived evidence hash mismatch'}
$gui=Get-ChildItem ($r+'\build\frozen-gui') -Filter result.json -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$g=Get-Content $gui.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
if(-not $g.passed -or -not $g.graceful_exit){throw 'GUI not passed or still running'}
$running=@(Get-Process | Where-Object {$_.Path -like ($r+'\*')})
if($running.Count){throw 'Owned executable still running'}
$items=@(Get-ChildItem $r -Recurse -Force -File)
$reparse=@(Get-ChildItem $r -Recurse -Force | Where-Object {($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0})
if($reparse.Count){throw 'Reparse points present; inspect instead of recursive deletion'}
$report=[ordered]@{root=$r;files=$items.Count;bytes=($items | Measure-Object Length -Sum).Sum;deleted=$false;final_installer=$retained.path;sha256=$expected;evidence_sha256=$transfer.evidence_sha256;reason='Own temporary source copy, compiler work files, onedir, extracted validation payload, and duplicate installer after archiving verified evidence';utc=(Get-Date).ToUniversalTime().ToString('o')}
$report | ConvertTo-Json | Set-Content ($s+'\windows-current-cleanup.json') -Encoding UTF8
Remove-Item -LiteralPath $r -Recurse -Force
if(Test-Path $r){throw 'Temporary directory remains'}
$report.deleted=$true
$report | ConvertTo-Json | Set-Content ($s+'\windows-current-cleanup.json') -Encoding UTF8
$report | ConvertTo-Json
