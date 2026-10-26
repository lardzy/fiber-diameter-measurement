$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$r='C:\tmp\fdm-slide-adjustments-release-20261026'
$id=Get-Content ($r+'\build\package\installer-identity.json') -Raw -Encoding UTF8|ConvertFrom-Json
if(-not $id.passed){throw 'Package verification must pass first'}
$fixture='C:\tmp\fdm-slide-adjustments-20261026\r2\out\combined-1366x768-1-light\synthetic.fdmslide'
Copy-Item -LiteralPath $fixture -Destination ($r+'\synthetic.fdmslide')
& ($r+'\probe.ps1') -Executable (Join-Path $id.payload 'FiberDiameterMeasurement.exe') -Slide ($r+'\synthetic.fdmslide') -OutputRoot ($r+'\build\frozen-gui')
exit $LASTEXITCODE
