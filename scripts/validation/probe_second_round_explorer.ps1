# Start the installed candidate through Explorer's shell, not a PowerShell child.
# Run after the owned GUI matrix has closed and before restoring the VM checkpoint.
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$root = 'C:\tmp\fdm-cu-live-second-round-20261017'
$exe = "$root\dist\windows\installed-candidate\FiberDiameterMeasurement.exe"
$out = "$root\build\second-round\explorer-start"
[void](New-Item -ItemType Directory -Force -Path $out)
$before = @(Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -eq $exe })
if ($before.Count) { throw 'The installed candidate is already running; refusing an ambiguous launch.' }
$identity = Get-Content "$root\build\second-round\package\installer-identity.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$payloadExe = Join-Path $identity.payload 'FiberDiameterMeasurement.exe'
if ((Get-FileHash $exe).Hash -ne (Get-FileHash $payloadExe).Hash) { throw 'Installed executable differs from verified installer.' }
$result = [ordered]@{
    schema=1; executable=$exe; build_id=$identity.build_id; launch='explorer.exe executable-path'
    user=[Security.Principal.WindowsIdentity]::GetCurrent().Name
    elevated=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    policy=[string](Get-ExecutionPolicy); started_at=(Get-Date).ToString('o'); ok=$false
    hardware_tested=$false; pid=$null; parent_name=$null; title=$null; error=$null
    graceful_exit=$false; forced_cleanup=$false
}
$owned = $null
try {
    Start-Process -FilePath "$env:WINDIR\explorer.exe" -ArgumentList $exe
    $deadline = [DateTime]::UtcNow.AddSeconds(45)
    do {
        Start-Sleep -Milliseconds 200
        $matches = @(Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -eq $exe })
    } while ($matches.Count -eq 0 -and [DateTime]::UtcNow -lt $deadline)
    if ($matches.Count -ne 1) { throw 'Expected exactly one new installed candidate process.' }
    $record = $matches[0]
    $owned = [Diagnostics.Process]::GetProcessById($record.ProcessId)
    # Retain a process handle before closing an Explorer-created process; a
    # later lookup after exit cannot reliably recover its exit code.
    $processHandle = $owned.Handle
    $result.pid = $owned.Id
    $result.parent_name = (Get-CimInstance Win32_Process -Filter "ProcessId=$($record.ParentProcessId)").Name
    do {
        Start-Sleep -Milliseconds 200
        $owned.Refresh()
    } while (-not $owned.HasExited -and $owned.MainWindowHandle -eq 0 -and [DateTime]::UtcNow -lt $deadline)
    if ($owned.HasExited -or $owned.MainWindowHandle -eq 0) { throw 'Main window did not appear.' }
    $result.title = $owned.MainWindowTitle
    if ($result.parent_name -ne 'explorer.exe') { throw 'Launch was not owned by Explorer.' }
    if ($result.elevated) { throw 'Ordinary-user launch unexpectedly ran in an elevated caller.' }
    $result.main_window_handle = $owned.MainWindowHandle.ToInt64()
    if (-not $owned.CloseMainWindow()) { throw 'Could not close the owned main window.' }
    $result.graceful_exit = $owned.WaitForExit(12000)
    if (-not $result.graceful_exit) { throw 'Untouched application did not close normally.' }
    $result.exit_code = $owned.ExitCode
    if ($owned.ExitCode -ne 0) { throw 'Application exited with an error.' }
    $result.ok = $true
} catch {
    $result.error = $_.Exception.Message
} finally {
    if ($owned) {
        $owned.Refresh()
        if (-not $owned.HasExited) { $owned.Kill(); $result.forced_cleanup=$true; [void]$owned.WaitForExit(5000) }
        $owned.Dispose()
    }
    $result.finished_at = (Get-Date).ToString('o')
    $text = $result | ConvertTo-Json -Depth 8
    [IO.File]::WriteAllText((Join-Path $out 'result.json'), $text, [Text.UTF8Encoding]::new($false))
    [Console]::WriteLine($text)
}
if (-not $result.ok) { exit 1 }
