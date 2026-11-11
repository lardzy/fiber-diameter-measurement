# Real desktop Excel modal-dialog and process-exit probes; private test PID only.
param([Parameter(Mandatory=$true)][string]$Root,[uint32]$ModalPid=0)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
Add-Type -TypeDefinition @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class FdmBusyProbe {
 public delegate bool Visit(IntPtr h,IntPtr p);
 [DllImport("user32.dll")]static extern bool EnumWindows(Visit v,IntPtr p);
 [DllImport("user32.dll")]static extern bool EnumChildWindows(IntPtr h,Visit v,IntPtr p);
 [DllImport("user32.dll")]static extern uint GetWindowThreadProcessId(IntPtr h,out uint p);
 [DllImport("user32.dll",CharSet=CharSet.Unicode)]static extern int GetClassName(IntPtr h,StringBuilder s,int n);
 [DllImport("user32.dll",CharSet=CharSet.Unicode)]static extern int GetWindowText(IntPtr h,StringBuilder s,int n);
 [DllImport("user32.dll")]public static extern bool PostMessage(IntPtr h,uint m,IntPtr w,IntPtr l);
 [DllImport("user32.dll")]public static extern bool SetForegroundWindow(IntPtr h);
 [StructLayout(LayoutKind.Sequential)]public struct Rect{public int Left,Top,Right,Bottom;}
 [StructLayout(LayoutKind.Sequential)]public struct GuiInfo{public int Size;public uint Flags;public IntPtr Active,Focus,Capture,Menu,Move,Caret;public Rect CaretRect;}
 [DllImport("user32.dll")]static extern bool GetGUIThreadInfo(uint thread,ref GuiInfo info);
 public static GuiInfo State(IntPtr window){uint pid;uint thread=GetWindowThreadProcessId(window,out pid);var info=new GuiInfo();info.Size=Marshal.SizeOf(info);if(!GetGUIThreadInfo(thread,ref info))throw new InvalidOperationException("Cannot read Excel caret state");return info;}
 public static string Class(IntPtr h){var value=new StringBuilder(128);GetClassName(h,value,128);return value.ToString();}
 [DllImport("oleacc.dll")]static extern int AccessibleObjectFromWindow(IntPtr h,uint id,ref Guid g,[MarshalAs(UnmanagedType.Interface)]out object o);
 public static object Window(uint pid){object found=null;Visit child=(h,p)=>{var s=new StringBuilder(128);GetClassName(h,s,128);
  if(s.ToString()!="EXCEL7")return true;var id=new Guid("00020400-0000-0000-C000-000000000046");object o;
  if(AccessibleObjectFromWindow(h,0xfffffff0,ref id,out o)==0&&o!=null){found=o;return false;}return true;};
  EnumWindows((h,p)=>{uint id;GetWindowThreadProcessId(h,out id);if(id==pid)EnumChildWindows(h,child,IntPtr.Zero);return found==null;},IntPtr.Zero);return found;}
 public static IntPtr Dialog(uint pid){IntPtr found=IntPtr.Zero;EnumWindows((h,p)=>{uint id;GetWindowThreadProcessId(h,out id);var s=new StringBuilder(128);GetWindowText(h,s,128);
  if(id==pid&&s.ToString()=="FDM worker busy probe"){found=h;return false;}return true;},IntPtr.Zero);return found;}
}
'@
if($ModalPid){
 $w=[FdmBusyProbe]::Window($ModalPid);$a=$w.Application
 try{[void]$a.InputBox('Controlled FDM modal test','FDM worker busy probe','',0,0,[Type]::Missing,[Type]::Missing,2)}
 finally{[void][Runtime.InteropServices.Marshal]::ReleaseComObject($a);[void][Runtime.InteropServices.Marshal]::ReleaseComObject($w)}
 exit 0
}
$result=[ordered]@{passed=$false;events=@();checks=@();errors=@()};$worker=$null;$excel=$null;$window=$null;$modal=$null
$old=@(Get-Process EXCEL -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
function Check($condition,$label){if(!$condition){throw $label};$script:result.checks+=@($label)}
function Send($operation,$generation=1,$revision=1,$payload=@{}){
 $r=@{operation=$operation;request_id=[Guid]::NewGuid().ToString('N');session_id='interruption';generation=$generation;input_revision=$revision;payload=$payload}
 $script:requestId=$r.request_id
 $bytes=[Text.UTF8Encoding]::new($false).GetBytes(($r|ConvertTo-Json -Depth 30 -Compress)+"`n")
 $script:worker.StandardInput.BaseStream.Write($bytes,0,$bytes.Length);$script:worker.StandardInput.BaseStream.Flush()
}
function Read-Reply($pending=$null){
 do{
  if(!$pending){$pending=$script:worker.StandardOutput.ReadLineAsync()}
  if(!$pending.Wait(60000)){throw 'worker response timeout'}
  if(!$pending.Result){throw 'worker exited'}
  $reply=$pending.Result|ConvertFrom-Json;$pending=$null
  Check ($reply.request_id -eq $script:requestId) 'reply identity matched'
  $script:result.events+=@($reply)
 }while($reply.event)
 return $reply
}
try{
 $start=[Diagnostics.ProcessStartInfo]::new('powershell.exe',('-NoProfile -NonInteractive -STA -ExecutionPolicy Bypass -File "'+(Join-Path $Root 'excel_live.ps1')+'"'))
 $start.UseShellExecute=$false;$start.CreateNoWindow=$true;$start.RedirectStandardInput=$true;$start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
 $start.StandardOutputEncoding=[Text.UTF8Encoding]::new($false);$start.StandardErrorEncoding=[Text.UTF8Encoding]::new($false)
 $worker=[Diagnostics.Process]::Start($start)
 $payload=(Get-Content (Join-Path $Root 'requests.json') -Raw -Encoding UTF8|ConvertFrom-Json).diameter
 $payload.template=Join-Path $Root 'sheet_dia.xlt'
 Send 'open' 1 1 $payload;$reply=Read-Reply
 Check ($reply.ok -and $old -notcontains [int]$reply.excel_pid) 'opened independent test instance'
 $ownedId=[int]$reply.excel_pid;$ownedStarted=(Get-Process -Id $ownedId).StartTime
 $window=[FdmBusyProbe]::Window([uint32]$ownedId);$excel=$window.Application
 $book=$excel.Workbooks.Item([string]$reply.name);$sheet=$book.Worksheets.Item([string]$payload.profile.data_sheet)
 $cell=$sheet.Range('B2');Check (!$cell.Locked) 'editable non-input cell selected for F2 probe'
 [void]$book.Activate();[void]$sheet.Activate();[void]$cell.Select()
 $excelHwnd=[IntPtr]$excel.Hwnd
 [void][FdmBusyProbe]::SetForegroundWindow($excelHwnd)
 $beforeEdit=[FdmBusyProbe]::State($excelHwnd);$beforeReady=[bool]$excel.Ready
 [void]$excel.SendKeys('{F2}',$true)
 $editWatch=[Diagnostics.Stopwatch]::StartNew()
 do{$edit=[FdmBusyProbe]::State($excelHwnd);if([FdmBusyProbe]::Class($edit.Focus) -ne 'EXCEL6'){Start-Sleep -Milliseconds 50}}while([FdmBusyProbe]::Class($edit.Focus) -ne 'EXCEL6' -and $editWatch.ElapsedMilliseconds -lt 3000)
 $result.f2_state=@{before_ready=$beforeReady;before_caret=$beforeEdit.Caret.ToInt64();before_focus_class=[FdmBusyProbe]::Class($beforeEdit.Focus);before_flags=$beforeEdit.Flags;caret=$edit.Caret.ToInt64();focus_class=[FdmBusyProbe]::Class($edit.Focus);caret_class=[FdmBusyProbe]::Class($edit.Caret);flags=$edit.Flags;ready_during=$excel.Ready;caret_rectangle=$edit.CaretRect}
 Check ($beforeReady -and [FdmBusyProbe]::Class($beforeEdit.Focus) -eq 'EXCEL7' -and [FdmBusyProbe]::Class($edit.Focus) -eq 'EXCEL6' -and $edit.Caret -ne $beforeEdit.Caret -and $edit.CaretRect.Right-$edit.CaretRect.Left -eq 1) 'F2 switches the native grid to an Excel text editor with a one-pixel caret'
 Send 'sync' 1 2 $payload
 $pending=$worker.StandardOutput.ReadLineAsync();$answered=$pending.Wait(1000)
 $result.f2_response_within_1000ms=$answered
 [void][FdmBusyProbe]::PostMessage($edit.Focus,0x100,[IntPtr]27,[IntPtr]::Zero)
 [void][FdmBusyProbe]::PostMessage($edit.Focus,0x101,[IntPtr]27,[IntPtr]::Zero)
 $reply=Read-Reply $pending
 $result.f2_outcome=@{ok=$reply.ok;code=$reply.code;workbook_connected=$reply.workbook_connected}
 Check (!$reply.ok -and $reply.code -eq 'excel_busy' -and $reply.workbook_connected) 'real F2 edit returns busy and keeps workbook association'
 Send 'sync' 1 3 $payload;$reply=Read-Reply
 Check ($reply.ok -and $reply.name -eq $book.Name) 'Esc allows complete sync to the same workbook'
 Check ([double]$sheet.Range('C8').Value2 -eq 3) 'post-edit raw count is complete'
 $result.f2_after=@{ready=[bool]$excel.Ready;caret=([FdmBusyProbe]::State($excelHwnd)).Caret.ToInt64()}
 [void][Runtime.InteropServices.Marshal]::ReleaseComObject($cell);[void][Runtime.InteropServices.Marshal]::ReleaseComObject($sheet)
 $modal=Start-Process powershell.exe -ArgumentList @('-NoProfile','-STA','-ExecutionPolicy','Bypass','-File',('"'+$PSCommandPath+'"'),'-Root',('"'+$Root+'"'),'-ModalPid',$ownedId) -PassThru -WindowStyle Hidden
 $watch=[Diagnostics.Stopwatch]::StartNew()
 do{$dialog=[FdmBusyProbe]::Dialog([uint32]$ownedId);if($dialog -eq [IntPtr]::Zero){Start-Sleep -Milliseconds 50}}while($dialog -eq [IntPtr]::Zero -and $watch.ElapsedMilliseconds -lt 10000)
 Check ($dialog -ne [IntPtr]::Zero) 'real Excel InputBox modal observed'
 Send 'sync' 1 4 $payload
 $pending=$worker.StandardOutput.ReadLineAsync();$answered=$pending.Wait(500)
 $result.modal_response_within_500ms=$answered
 [void][FdmBusyProbe]::PostMessage($dialog,0x10,[IntPtr]::Zero,[IntPtr]::Zero)
 Check ($modal.WaitForExit(10000)) 'owned modal closed and helper exited'
 $reply=Read-Reply $pending
 $result.modal_outcome=@{ok=$reply.ok;code=$reply.code;needs_full_sync=$reply.needs_full_sync}
 Check ($reply.ok -or $reply.code -in @('excel_busy','excel_failed')) 'modal reply classified without closing Excel'
 Send 'sync' 1 5 $payload;$reply=Read-Reply
 Check ($reply.ok) 'full sync recovered after real modal'
 $book=$excel.Workbooks.Item([string]$reply.name)
 Check ([double]$book.Worksheets.Item([string]$payload.profile.data_sheet).Range('C8').Value2 -eq 3) 'post-modal count is complete'
 # Only this probe-created instance is deliberately terminated, by identity.
 $current=Get-Process -Id $ownedId
 Check ($current.StartTime -eq $ownedStarted) 'owned Excel process identity verified before exit probe'
 Stop-Process -Id $ownedId -Force;$current.WaitForExit(5000)|Out-Null
 [void][Runtime.InteropServices.Marshal]::ReleaseComObject($excel);$excel=$null
 [void][Runtime.InteropServices.Marshal]::ReleaseComObject($window);$window=$null
 Send 'status' 1 5;$reply=Read-Reply
 Check (!$reply.ok -and $reply.code -eq 'workbook_closed') 'Excel process exit reports closed without success'
 Send 'open' 2 6 $payload;$reply=Read-Reply
 Check ($reply.ok -and [int]$reply.excel_pid -ne $ownedId) 'explicit reconnect creates new instance and rebuilds input'
 $window=[FdmBusyProbe]::Window([uint32]$reply.excel_pid);$excel=$window.Application
 Send 'detach' 2 6;$reply=Read-Reply;Check ($reply.ok) 'reconnected workbook detaches'
 Send 'shutdown' 0 0;$reply=Read-Reply;Check ($reply.ok) 'worker shuts down'
 $worker.StandardInput.Close();Check ($worker.WaitForExit(5000)) 'worker exits'
 $result.passed=$true
}catch{$result.errors+=@{message=$_.Exception.Message;stack=$_.ScriptStackTrace}}finally{
 if($excelHwnd){try{$edit=[FdmBusyProbe]::State($excelHwnd);if($edit.Caret -ne [IntPtr]::Zero){[void][FdmBusyProbe]::PostMessage($edit.Focus,0x100,[IntPtr]27,[IntPtr]::Zero);[void][FdmBusyProbe]::PostMessage($edit.Focus,0x101,[IntPtr]27,[IntPtr]::Zero)}}catch{}}
 if($modal -and !$modal.HasExited){$dialog=[FdmBusyProbe]::Dialog([uint32]$ownedId);if($dialog -ne [IntPtr]::Zero){[void][FdmBusyProbe]::PostMessage($dialog,0x10,[IntPtr]::Zero,[IntPtr]::Zero)}}
 if($worker -and !$worker.HasExited){$worker.StandardInput.Close();if(!$worker.WaitForExit(5000)){$worker.Kill()}}
 if($excel){$excel.DisplayAlerts=$false;while($excel.Workbooks.Count){$excel.Workbooks.Item(1).Close($false)};$excel.Quit();[void][Runtime.InteropServices.Marshal]::ReleaseComObject($excel)}
 if($window){[void][Runtime.InteropServices.Marshal]::ReleaseComObject($window)}
 [GC]::Collect();[GC]::WaitForPendingFinalizers()
 [IO.File]::WriteAllText((Join-Path $Root 'worker-interruptions.json'),($result|ConvertTo-Json -Depth 30),[Text.UTF8Encoding]::new($false))
 $result|ConvertTo-Json -Depth 30
}
if(!$result.passed){exit 1}
