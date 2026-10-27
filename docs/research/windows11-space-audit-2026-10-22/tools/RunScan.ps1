$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
Add-Type -Path (Join-Path $root 'SpaceScan.cs')
$scan=Join-Path $root 'out\scan'
[SpaceScan]::Run('C:\',$scan)
$transfer=Join-Path $root 'transfer'
[IO.Directory]::CreateDirectory($transfer)|Out-Null
[IO.Directory]::CreateDirectory((Join-Path $transfer 'scan'))|Out-Null
Get-ChildItem -LiteralPath (Join-Path $root 'out') -File|Where-Object {$_.Name -ne 'dism.log'}|ForEach-Object{Copy-Item -LiteralPath $_.FullName -Destination $transfer}
Get-ChildItem -LiteralPath $scan -File|Where-Object {$_.Name -ne 'all-files-index-local-only.csv'}|ForEach-Object{Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $transfer 'scan')}
Compress-Archive -LiteralPath $transfer -DestinationPath (Join-Path $root 'results.zip')
Get-Content -LiteralPath (Join-Path $scan 'summary.json') -Raw -Encoding UTF8
