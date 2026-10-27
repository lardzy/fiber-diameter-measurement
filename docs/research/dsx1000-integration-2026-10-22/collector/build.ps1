$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
$out=Join-Path $root 'dist'
[IO.Directory]::CreateDirectory($out)|Out-Null
$framework=Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319'
$csc=Join-Path $framework 'csc.exe'
if(-not (Test-Path -LiteralPath $csc)){throw 'Existing Framework64 csc.exe is unavailable; no installation attempted.'}
$referenceRoot='C:\Program Files (x86)\Reference Assemblies\Microsoft\Framework\.NETFramework\v4.8'
$hasReferencePack=Test-Path -LiteralPath $referenceRoot
$libraries=@('System.dll','System.Core.dll','System.Drawing.dll','System.Windows.Forms.dll','System.Management.dll','System.Web.Extensions.dll','System.IO.Compression.dll','System.IO.Compression.FileSystem.dll')
$argsList=@('/nologo','/target:winexe','/platform:x64','/optimize+','/langversion:5','/codepage:65001',('/out:'+(Join-Path $out 'DsxEvidenceCollector.exe')),('/win32manifest:'+(Join-Path $root 'src\app.manifest')))
if($hasReferencePack){$argsList+=('/lib:'+$referenceRoot);$argsList+='/nostdlib+';$argsList+=('/reference:'+(Join-Path $referenceRoot 'mscorlib.dll'))}
foreach($lib in $libraries){$argsList+=('/reference:'+$lib)}
$sources=@(Get-ChildItem -LiteralPath (Join-Path $root 'src') -Filter '*.cs' | Sort-Object Name | ForEach-Object {$_.FullName})
$argsList+=$sources
$buildOutput=& $csc @argsList 2>&1
$buildExit=$LASTEXITCODE
$buildOutput | Out-String | Set-Content -LiteralPath (Join-Path $out 'build.log') -Encoding UTF8
$buildOutput | Write-Output
if($buildExit -ne 0){throw ('C# compiler failed: '+$buildExit)}
Copy-Item -LiteralPath (Join-Path $root 'src\DsxEvidenceCollector.exe.config') -Destination $out -Force
Copy-Item -LiteralPath (Join-Path $root 'baseline.json') -Destination $out -Force
$identity=[ordered]@{schema=1;timestamp_utc=[DateTime]::UtcNow.ToString('o');os=[Environment]::OSVersion.VersionString;runtime=[Environment]::Version.ToString();compiler=$csc;compiler_version=(Get-Item $csc).VersionInfo.FileVersion;target='x64 / .NET Framework 4.8 / C#5';reference_pack_48=$hasReferencePack;reference_policy='Uses existing framework assemblies when v4.8 targeting pack is absent; source restricted to .NET 4.8 APIs. Win10 1909 not tested.';exe_sha256=(Get-FileHash (Join-Path $out 'DsxEvidenceCollector.exe')).Hash.ToLowerInvariant();sources=@($sources|ForEach-Object{[ordered]@{name=[IO.Path]::GetFileName($_);sha256=(Get-FileHash $_).Hash.ToLowerInvariant()}});vendor_code_executed=$false;fdm_installation_modified=$false}
$identity | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $out 'build-identity.json') -Encoding UTF8
$identity | ConvertTo-Json -Depth 6
