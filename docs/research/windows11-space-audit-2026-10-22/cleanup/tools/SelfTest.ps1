# Windows PowerShell 5.1. Creates/deletes only synthetic fixtures beneath ../selftest/.
# Existing test runs are retained. It never enumerates cleanup candidates or invokes the VM.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2

if (-not ('VmCleanup.CleanupCore' -as [type])) {
    Add-Type -TypeDefinition ([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'CleanupCore.cs'))) -ReferencedAssemblies @(
        'System.dll', 'System.Core.dll', 'System.Runtime.Serialization.dll', 'System.Xml.dll',
        'System.IO.Compression.dll', 'System.IO.Compression.FileSystem.dll'
    )
}
$cleanupRoot = $PSScriptRoot
$testRoot = Join-Path $cleanupRoot ('selftest\' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '-' + [Guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($testRoot)
$results = New-Object 'System.Collections.Generic.List[object]'
$utf8 = New-Object Text.UTF8Encoding($false)

function Write-Fixture([string]$Path, [string]$Text) {
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path))
    [IO.File]::WriteAllText($Path, $Text, $script:utf8)
}
function Get-Reviewed([string]$Root) {
    foreach ($record in [VmCleanup.CleanupCore]::ListFiles($Root)) {
        $record.Sha256 = [VmCleanup.CleanupCore]::Hash($record.FullPath)
        $record
    }
}
function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Assert-Rejected([scriptblock]$Operation) {
    $failure = $null
    try { & $Operation | Out-Null } catch { $failure = $_.Exception.ToString() }
    if (-not $failure) { throw 'Expected rejection, but operation succeeded.' }
    return $failure
}
function Record-Pass([string]$Name, [object]$Detail) {
    $script:results.Add([pscustomobject]@{ name=$Name; passed=$true; detail=$Detail })
    Write-Output ('PASS ' + $Name)
}

$success = $false
try {
    $scope = Join-Path $testRoot 'archive-roundtrip'
    Write-Fixture (Join-Path $scope '说明.txt') '中文路径与归档核对'
    Write-Fixture (Join-Path $scope 'nested\second.bin') 'second synthetic file'
    [void][IO.Directory]::CreateDirectory((Join-Path $scope 'empty\deeper'))
    [VmCleanup.FileRecord[]]$files = @(Get-Reviewed $scope)
    $json = ConvertTo-Json -InputObject @($files) -Depth 8
    [IO.File]::WriteAllText((Join-Path $testRoot 'reviewed-roundtrip.json'), $json, $utf8)
    # Explicit mapping avoids PowerShell provider metadata and works in Windows PS 5.1.
    [VmCleanup.FileRecord[]]$roundtrip = @((ConvertFrom-Json $json) | ForEach-Object {
        [VmCleanup.FileRecord]@{
            Root=$_.Root; RelativePath=$_.RelativePath; FullPath=$_.FullPath
            Length=[long]$_.Length; LastWriteUtcTicks=[long]$_.LastWriteUtcTicks; Sha256=$_.Sha256
            VolumeSerialNumber=[uint32]$_.VolumeSerialNumber; FileIndexHigh=[uint32]$_.FileIndexHigh
            FileIndexLow=[uint32]$_.FileIndexLow; ArchiveEntry=$_.ArchiveEntry
        }
    })
    $zip = Join-Path $testRoot 'selected.zip'
    $archived = [VmCleanup.CleanupCore]::ArchiveSelected($roundtrip, $zip)
    $verified = [VmCleanup.CleanupCore]::VerifyArchive($zip)
    Assert-True ($verified.Files.Count -eq 2) 'Archive did not contain exactly two reviewed files.'
    $deleted = [VmCleanup.CleanupCore]::DeleteReviewed($scope, $roundtrip)
    Assert-True ($deleted.DeletedFiles.Count -eq 2 -and -not [IO.Directory]::Exists($scope)) 'Reviewed directory deletion failed.'
    Record-Pass 'archive-verify-json-roundtrip-delete-empty-directories' @{
        archive=$zip; archiveSha256=[VmCleanup.CleanupCore]::Hash($zip)
        files=$verified.Files.Count; bytes=$verified.TotalBytes; removedDirectories=$deleted.DeletedDirectories.Count
    }

    $scope = Join-Path $testRoot 'changed-file'
    $file = Join-Path $scope 'payload.txt'
    Write-Fixture $file 'abc'
    [VmCleanup.FileRecord[]]$files = @(Get-Reviewed $scope)
    Write-Fixture $file 'xyz'
    # Same size and restored timestamp: the content hash must still reject the file.
    [IO.File]::SetLastWriteTimeUtc($file, [DateTime]::new($files[0].LastWriteUtcTicks, [DateTimeKind]::Utc))
    $reason = Assert-Rejected { [VmCleanup.CleanupCore]::DeleteReviewed($scope, $files) }
    Assert-True ([IO.File]::Exists($file)) 'Modified file was deleted.'
    Record-Pass 'modified-content-rejected' $reason

    $scope = Join-Path $testRoot 'unexpected-file'
    $file = Join-Path $scope 'reviewed.txt'
    Write-Fixture $file 'reviewed original'
    [VmCleanup.FileRecord[]]$files = @(Get-Reviewed $scope)
    $extra = Join-Path $scope 'extra.txt'
    Write-Fixture $extra 'new unreviewed file'
    $reason = Assert-Rejected { [VmCleanup.CleanupCore]::DeleteReviewed($scope, $files) }
    Assert-True ([IO.File]::Exists($file) -and [IO.File]::Exists($extra)) 'New-file refusal did not preserve all files.'
    Record-Pass 'additional-file-refuses-entire-root' $reason

    $scope = Join-Path $testRoot 'replaced-file'
    $file = Join-Path $scope 'same-content.txt'
    Write-Fixture $file 'same content'
    [VmCleanup.FileRecord[]]$files = @(Get-Reviewed $scope)
    [IO.File]::Move($file, (Join-Path $testRoot 'original-file-retained.txt'))
    Write-Fixture $file 'same content'
    [IO.File]::SetLastWriteTimeUtc($file, [DateTime]::new($files[0].LastWriteUtcTicks, [DateTimeKind]::Utc))
    $reason = Assert-Rejected { [VmCleanup.CleanupCore]::DeleteReviewed($scope, $files) }
    Assert-True ([IO.File]::Exists($file)) 'Replacement with identical contents was deleted.'
    Record-Pass 'replaced-file-identity-rejected' $reason

    $file = Join-Path $testRoot 'standalone-synthetic.exe'
    Write-Fixture $file 'synthetic text; never executed'
    [VmCleanup.FileRecord[]]$files = @(Get-Reviewed $file)
    $zip = Join-Path $testRoot 'single-file.zip'
    $singleArchive = [VmCleanup.CleanupCore]::ArchiveSelected($files, $zip)
    $deleted = [VmCleanup.CleanupCore]::DeleteReviewed($file, $files)
    Assert-True ($deleted.DeletedFiles.Count -eq 1 -and -not [IO.File]::Exists($file)) 'Single-file root failed.'
    Record-Pass 'single-file-root-archive-and-delete' $singleArchive.Files[0].ArchiveEntry

    $reason = Assert-Rejected { [VmCleanup.CleanupCore]::ListFiles([IO.Path]::GetPathRoot($testRoot)) }
    Record-Pass 'drive-root-refused' $reason
    $reason = Assert-Rejected { [VmCleanup.CleanupCore]::ListFiles('\\Mac\Home') }
    Record-Pass 'network-shared-path-refused' $reason
    $success = $true
} catch {
    $results.Add([pscustomobject]@{ name='selftest-failure'; passed=$false; detail=$_.Exception.ToString() })
    throw
} finally {
    $report = [pscustomobject]@{
        schemaVersion=1; completedUtc=[DateTime]::UtcNow.ToString('o'); passed=$success
        testRoot=$testRoot; checks=$results.ToArray()
        note='Synthetic fixtures only. All writes and intended deletions are inside this unique selftest run; evidence is retained.'
    }
    [IO.File]::WriteAllText((Join-Path $testRoot 'results.json'), (ConvertTo-Json -InputObject $report -Depth 12), $utf8)
    Write-Output ('Self-test evidence: ' + $testRoot)
}
