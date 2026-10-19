# Reuse the ordinary-user Explorer launch/owned-process cleanup contract. This
# must run inside the approved VM checkpoint after installing the candidate.
$ErrorActionPreference='Stop'
$source=[IO.File]::ReadAllText((Join-Path $PSScriptRoot 'probe_second_round_explorer.ps1'))
$source=$source.Replace('C:\tmp\fdm-cu-live-second-round-20261017','C:\tmp\fdm-cu-live-third-round-20261019').Replace('build\second-round\','build\third-round\')
if($source.Contains('fdm-cu-live-second-round-20261017')) {throw 'Unexpected legacy candidate path'}
& ([scriptblock]::Create($source))
