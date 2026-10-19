# Read-only guest inventory. Does not open a camera, Excel, or the installed app.
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$resources = [ordered]@{
    python = 'C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
    native_tools = 'C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    compiler = 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
    extractor = 'C:\tmp\fdm-cu-live-final-r4-20261015\build\validation-tools\innounp.exe'
}
$paths = @($resources.GetEnumerator() | ForEach-Object {
    [ordered]@{name=$_.Key; path=$_.Value; exists=(Test-Path -LiteralPath $_.Value -PathType Leaf)}
})
$installed = @(
    foreach ($registryRoot in @('HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall',
        'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall',
        'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall')) {
        Get-ItemProperty "$registryRoot\*" -ErrorAction SilentlyContinue |
            Where-Object { $_.PSChildName -eq '{F0F6A8B5-4838-4DF0-B2C1-18F1D5AA4A66}_is1' } |
            Select-Object DisplayName, DisplayVersion, InstallLocation, PSPath
    }
)
$processes = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('FiberDiameterMeasurement.exe', 'EXCEL.EXE')
} | Select-Object Name, ProcessId, ExecutablePath)
$result = [ordered]@{
    schema=1; timestamp=(Get-Date).ToString('o')
    user=[Security.Principal.WindowsIdentity]::GetCurrent().Name
    elevated=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    powershell=$PSVersionTable.PSVersion.ToString()
    policies=@(Get-ExecutionPolicy -List | ForEach-Object { [ordered]@{scope=[string]$_.Scope; policy=[string]$_.ExecutionPolicy} })
    installed=$installed; resources=$paths; existing_processes=$processes
    excel_registered=($null -ne [type]::GetTypeFromProgID('Excel.Application'))
    candidate_directory_exists=(Test-Path -LiteralPath 'C:\tmp\fdm-cu-live-third-round-20261019')
    system_state_modified=$false
}
$result | ConvertTo-Json -Depth 8
