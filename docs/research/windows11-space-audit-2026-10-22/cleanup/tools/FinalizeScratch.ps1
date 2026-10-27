# Windows PowerShell 5.1. Execute a ReadAllText -> ScriptBlock copy in memory.
# Only these two fixed scratch roots may be deleted; cleanup evidence stays on the host.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$cleanupRoot = 'C:\tmp\fdm-space-cleanup-20261022'
$scratchRoots = @('C:\tmp\windows11-space-audit-20261022', $cleanupRoot)
$rows = [Collections.Generic.List[object]]::new()
$before = $null; $after = $null; $errorText = $null
$gateEvidence = $null; $mutex = $null; $ownsMutex = $false
$deletionStarted = $false; $startedUtc = [DateTime]::UtcNow.ToString('o')

function Volume {
    $disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'" -ErrorAction Stop
    if ($null -eq $disk -or $null -eq $disk.FreeSpace) { throw 'Cannot read C: free space.' }
    return @{ utc=[DateTime]::UtcNow.ToString('o'); free_bytes=[long]$disk.FreeSpace }
}
function Decode-Command([string]$Command) {
    $decoded = $Command
    if ($Command -notmatch '(?i)(?:powershell|pwsh)(?:\.exe)?(?:["\s]|$)') { return $decoded }
    $match = [regex]::Match($Command, '(?i)(?:^|\s)-(?:EncodedCommand|Enc|EC|E)\s+["'']?([A-Za-z0-9+/=]+)')
    if ($match.Success) {
        try { $decoded += ' ' + [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($match.Groups[1].Value)) }
        catch { throw 'Cannot inspect encoded PowerShell command.' }
    }
    return $decoded
}
function Assert-NoCleanupJobs {
    foreach ($process in Get-CimInstance Win32_Process -ErrorAction Stop) {
        if ([int]$process.ProcessId -eq $PID) { continue }
        $name = [string]$process.Name; $command = [string]$process.CommandLine
        if ($name -match '^(powershell|pwsh|powershell_ise)\.exe$' -and !$command) {
            throw ('PowerShell activity is unreadable; pid=' + $process.ProcessId)
        }
        if ($command -and (Decode-Command $command) -match '(?i)(?:Cleanup|CleanBrowserCaches)\.ps1') {
            throw ('Another cleanup process is active; pid=' + $process.ProcessId + '; name=' + $name)
        }
    }
    foreach ($job in Get-Job -ErrorAction Stop) {
        if ([string]$job.State -in @('Running','NotStarted','Blocked') -and
            (Decode-Command ([string]$job.Command)) -match '(?i)(?:Cleanup|CleanBrowserCaches)\.ps1') {
            throw ('Another cleanup job is active; id=' + $job.Id)
        }
    }
    foreach ($task in Get-ScheduledTask -ErrorAction Stop) {
        if ([string]$task.State -ne 'Running') { continue }
        foreach ($action in $task.Actions) {
            $execute = $action.PSObject.Properties['Execute']; $arguments = $action.PSObject.Properties['Arguments']
            if ($null -eq $execute -and $null -eq $arguments) { continue }
            $command = ''
            if ($null -ne $execute) { $command += [string]$execute.Value }
            if ($null -ne $arguments) { $command += ' ' + [string]$arguments.Value }
            if ((Decode-Command $command) -match '(?i)(?:Cleanup|CleanBrowserCaches)\.ps1') {
                throw ('Another cleanup scheduled task is active: ' + $task.TaskName)
            }
        }
    }
}
function Assert-PlainDirectory([string]$Path) {
    $current=[IO.Path]::GetFullPath($Path)
    while ($current) {
        $attributes=[IO.File]::GetAttributes($current)
        if (($attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
            ($attributes -band [IO.FileAttributes]::Directory) -eq 0) {
            throw ('Non-directory or reparse path rejected before compiler initialization: ' + $current)
        }
        $current=[IO.Path]::GetDirectoryName($current)
    }
}
function Check-Receipt([string]$ReceiptName, [string]$ArchiveName) {
    $receiptPath = Join-Path $cleanupRoot $ReceiptName
    # Guard every input path through CleanupCore before reading its small receipt.
    [void][VmCleanup.CleanupCore]::Hash($receiptPath)
    $expected = [IO.File]::ReadAllText($receiptPath).Trim()
    if ($expected -notmatch '^[0-9a-fA-F]{64}$') { throw ('Invalid host receipt: ' + $ReceiptName) }
    $actual = [VmCleanup.CleanupCore]::Hash((Join-Path $cleanupRoot $ArchiveName))
    if ($actual -ne $expected) { throw ('Host receipt does not match archive: ' + $ArchiveName) }
    return $actual
}

try {
    $before = Volume
    $mutex = [Threading.Mutex]::new($false, 'Global\FdmSpaceFinalizeScratch20261022')
    $ownsMutex = $mutex.WaitOne(0)
    if (!$ownsMutex) { throw 'Another scratch finalization is active.' }
    Assert-NoCleanupJobs
    if (![IO.Directory]::Exists($cleanupRoot)) { throw 'Fixed cleanup evidence root is missing.' }
    Assert-PlainDirectory $cleanupRoot

    if (-not ('VmCleanup.CleanupCore' -as [type])) {
        # CodeDOM may use TEMP/TMP while compiling. Keep its temporary files in scope.
        $compileTemp = Join-Path $cleanupRoot 'finalize-compile-tmp'
        if ([IO.Directory]::Exists($compileTemp)) { Assert-PlainDirectory $compileTemp }
        [void][IO.Directory]::CreateDirectory($compileTemp)
        Assert-PlainDirectory $compileTemp
        $previousTemp = $env:TEMP; $previousTmp = $env:TMP
        try {
            $env:TEMP = $compileTemp; $env:TMP = $compileTemp
            Add-Type -TypeDefinition ([IO.File]::ReadAllText((Join-Path $cleanupRoot 'CleanupCore.cs'))) -ReferencedAssemblies @(
                'System.dll','System.Core.dll','System.Runtime.Serialization.dll','System.Xml.dll',
                'System.IO.Compression.dll','System.IO.Compression.FileSystem.dll'
            )
        } finally { $env:TEMP = $previousTemp; $env:TMP = $previousTmp }
    }

    $verifyPath = Join-Path $cleanupRoot 'verify-result.json'
    $verifyHash = [VmCleanup.CleanupCore]::Hash($verifyPath)
    $verify = [IO.File]::ReadAllText($verifyPath) | ConvertFrom-Json
    if ([string]$verify.status -ne 'passed' -or $verify.registered_installation_unchanged -ne $true -or
        @($verify.differences).Count -ne 0) { throw 'Protected-file verification has not passed.' }
    $finalHash = Check-Receipt 'host-final-evidence-verified.sha256' 'final-evidence.zip'
    $preservedHash = Check-Receipt 'host-archive-verified.sha256' 'preserved.zip'

    # ReferencedAssemblies for CodeDOM does not load these types into PowerShell 5.1.
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    # Confirm that the successful verification is the exact evidence already copied to Mac.
    $zipStream = [IO.File]::Open((Join-Path $cleanupRoot 'final-evidence.zip'), [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try {
        $zip = [IO.Compression.ZipArchive]::new($zipStream, [IO.Compression.ZipArchiveMode]::Read, $true)
        try {
            $entries = @($zip.Entries | Where-Object { $_.FullName -ceq 'verify-result.json' })
            if ($entries.Count -ne 1) { throw 'Final evidence must contain one root-level verify-result.json.' }
            $entryStream = $entries[0].Open(); $sha = [Security.Cryptography.SHA256]::Create()
            try { $archivedVerifyHash = [BitConverter]::ToString($sha.ComputeHash($entryStream)).Replace('-','').ToLowerInvariant() }
            finally { $sha.Dispose(); $entryStream.Dispose() }
            if ($archivedVerifyHash -ne $verifyHash -or [VmCleanup.CleanupCore]::Hash($verifyPath) -ne $verifyHash) {
                throw 'Current verification result differs from the host-verified final evidence.'
            }
            if ([VmCleanup.CleanupCore]::Hash((Join-Path $cleanupRoot 'final-evidence.zip')) -ne $finalHash) {
                throw 'Final evidence archive changed during validation.'
            }
        } finally { $zip.Dispose() }
    } finally { $zipStream.Dispose() }
    $gateEvidence = @{ verify_result_sha256=$verifyHash; final_evidence_sha256=$finalHash; preserved_sha256=$preservedHash }

    # A surviving marker requires manual inspection after a failure; there is no retry path.
    $marker = [IO.File]::Open((Join-Path $cleanupRoot 'finalize-started.json'), [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes((ConvertTo-Json -Compress -InputObject @{ pid=$PID; utc=$startedUtc; gates=$gateEvidence }))
        $marker.Write($bytes,0,$bytes.Length); $marker.Flush($true)
    } finally { $marker.Dispose() }

    $reviewed = @{}
    foreach ($path in $scratchRoots) {
        $row = @{ path=$path; files=0; logical_bytes=[long]0; status='preflight' }; $rows.Add($row)
        if (![IO.Directory]::Exists($path)) {
            if ([IO.File]::Exists($path)) { throw ('Scratch root is a file: ' + $path) }
            $row.status='already_absent'; continue
        }
        [VmCleanup.FileRecord[]]$files = @([VmCleanup.CleanupCore]::ListFiles($path))
        foreach ($file in $files) { $file.Sha256=[VmCleanup.CleanupCore]::Hash($file.FullPath) }
        $row.files=$files.Count; $row.logical_bytes=[long](($files | Measure-Object Length -Sum).Sum)
        if (!$files.Count) { $row.status='skipped_empty'; continue }
        $reviewed[$path]=$files; $row.status='ready'
    }
    Assert-NoCleanupJobs
    foreach ($row in $rows) {
        if ($row.status -ne 'ready') { continue }
        Assert-NoCleanupJobs
        $deletionStarted=$true; $row.status='deleting'
        $deleted=[VmCleanup.CleanupCore]::DeleteReviewed($row.path, $reviewed[$row.path])
        $row.deleted_files=$deleted.DeletedFiles.Count
        $row.deleted_logical_bytes=$deleted.DeletedLogicalBytes
        $row.root_still_exists=$deleted.RootStillExists
        if ($deleted.RootStillExists) { $row.status='root_remaining'; throw ('Scratch root remains after exact deletion: ' + $row.path) }
        $row.status='deleted'
    }
} catch {
    $errorText=$_.Exception.Message
    foreach ($row in $rows) { if ($row.status -eq 'deleting') { $row.status='failed_or_partial' } }
} finally {
    try { $after=Volume } catch { if (!$errorText) { $errorText=$_.Exception.Message } }
    if ($ownsMutex -and $null -ne $mutex) { $mutex.ReleaseMutex() }
    if ($null -ne $mutex) { $mutex.Dispose() }
}
$status = if ($errorText) { if ($deletionStarted) { 'failed_or_partial' } else { 'refused_before_deletion' } } else { 'completed' }
$result = [ordered]@{ schema=1; status=$status; started_utc=$startedUtc; finished_utc=[DateTime]::UtcNow.ToString('o'); before=$before; after=$after; gates=$gateEvidence; paths=$rows.ToArray(); error=$errorText }
ConvertTo-Json -InputObject $result -Depth 8 -Compress
if ($errorText) { exit 1 }
