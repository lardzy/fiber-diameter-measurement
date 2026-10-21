$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root='C:\tmp\fdm-cu-live-explicit-gestures-20261021'
Set-Location $root
$env:PYTHONPATH="$root\src;$root\validation_vendor"
$env:QT_QPA_PLATFORM='windows'
$env:QT_SCALE_FACTOR='0.5'
$env:PYTHONIOENCODING='utf-8'
$python='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$ErrorActionPreference='Continue'
& $python scripts/validation/verify_seventh_round_installed_selfcheck.py --root $root *> "$root\build\seventh-round\installed-selfcheck-launch.log"
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
$ErrorActionPreference='Stop'
$probe=[IO.File]::ReadAllText("$root\scripts\validation\probe_acquisition_seventh_round_explorer.ps1")
$needle=' if(-not $owned.CloseMainWindow()){throw ''Cannot close owned window''}'
$capture=@'
 Add-Type -AssemblyName System.Drawing
 Add-Type -AssemblyName System.Windows.Forms
 Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class FdmVisibleCapture { [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left,Top,Right,Bottom; } [DllImport("user32.dll")] public static extern bool SetProcessDPIAware(); [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd,out RECT rect); [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd); [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr hWnd); [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd,int mode); [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow(); [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hWnd,IntPtr dc,uint flags); }'
 [void][FdmVisibleCapture]::SetProcessDPIAware()
 [void][FdmVisibleCapture]::ShowWindow($owned.MainWindowHandle,9)
 $shell=New-Object -ComObject WScript.Shell
 [void]$shell.AppActivate($owned.Id)
 [void][Runtime.InteropServices.Marshal]::ReleaseComObject($shell)
 [void][FdmVisibleCapture]::BringWindowToTop($owned.MainWindowHandle)
 [void][FdmVisibleCapture]::SetForegroundWindow($owned.MainWindowHandle)
 Start-Sleep -Milliseconds 1000
 $owned.Refresh()
 if($owned.MainWindowTitle -ne '显微测量工作台'){throw 'Unexpected owned application window title'}
 $foreground=[FdmVisibleCapture]::GetForegroundWindow()
 $result.foreground_verified=($foreground -eq $owned.MainWindowHandle)
 $result.foreground_handle=$foreground.ToInt64()
 if(-not $result.foreground_verified){throw 'Owned FDM window was not brought to the foreground'}
 $windowRect=New-Object FdmVisibleCapture+RECT
 if(-not [FdmVisibleCapture]::GetWindowRect($owned.MainWindowHandle,[ref]$windowRect)){throw 'Cannot capture installed window bounds'}
 $rect=[Drawing.Rectangle]::FromLTRB($windowRect.Left,$windowRect.Top,$windowRect.Right,$windowRect.Bottom)
 if($rect.Width -le 0 -or $rect.Height -le 0){throw 'Invalid owned window size'}
 $bitmap=[Drawing.Bitmap]::new($rect.Width,$rect.Height)
 $graphics=[Drawing.Graphics]::FromImage($bitmap)
 $dc=$graphics.GetHdc()
 try{$printed=[FdmVisibleCapture]::PrintWindow($owned.MainWindowHandle,$dc,2)}finally{$graphics.ReleaseHdc($dc)}
 try{if(-not $printed){throw 'PrintWindow failed'};$bitmap.Save("$out\installed-window-printwindow.png",[Drawing.Imaging.ImageFormat]::Png)}finally{$graphics.Dispose();$bitmap.Dispose()}
 $result.printwindow_returned=$printed
 $screenRect=[Drawing.Rectangle]::Intersect($rect,[Windows.Forms.SystemInformation]::VirtualScreen)
 if($result.foreground_verified -and $screenRect.Width -gt 0 -and $screenRect.Height -gt 0){
  $bitmap=[Drawing.Bitmap]::new($screenRect.Width,$screenRect.Height);$graphics=[Drawing.Graphics]::FromImage($bitmap)
  try{$graphics.CopyFromScreen($screenRect.Location,[Drawing.Point]::Empty,$screenRect.Size);$bitmap.Save("$out\installed-window-screen.png",[Drawing.Imaging.ImageFormat]::Png)}finally{$graphics.Dispose();$bitmap.Dispose()}
  $result.screen_capture="$out\installed-window-screen.png"
 }
 $result.screenshot="$out\installed-window-printwindow.png"
 $result.screenshot_sha256=(Get-FileHash $result.screenshot).Hash.ToLowerInvariant()
 $result.screenshot_rectangle=@{x=$rect.X;y=$rect.Y;width=$rect.Width;height=$rect.Height}
 if(-not $owned.CloseMainWindow()){throw 'Cannot close owned window'}
'@
if($probe.Split([string[]]@($needle),[StringSplitOptions]::None).Count -ne 2){throw 'Unexpected Explorer close gate'}
$probe=$probe.Replace($needle,$capture)
$path="$root\build\seventh-round\probe-installed-visible.ps1"
[IO.File]::WriteAllText($path,$probe,[Text.UTF8Encoding]::new($true))
& $path *> "$root\build\seventh-round\explorer-launch.log"
exit $LASTEXITCODE
