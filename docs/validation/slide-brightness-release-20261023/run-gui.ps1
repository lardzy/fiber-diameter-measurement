$ProgressPreference='SilentlyContinue';$ErrorActionPreference='Stop'
$r='C:\tmp\fdm-brightness-release-20261023'
$id=Get-Content ($r+'\build\package\installer-identity.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if(-not $id.passed){throw 'Package verification must pass first'}
& ($r+'\probe.ps1') -Executable (Join-Path $id.payload 'FiberDiameterMeasurement.exe') -Slide ($r+'\synthetic.fdmslide') -OutputRoot ($r+'\build\frozen-gui')
