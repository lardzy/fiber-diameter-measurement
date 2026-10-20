# Windows PowerShell 5.1. Reuse the audited, PID-scoped UIA tools of the first
# redesign probe, but execute the fifth-round navigation contract below.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Executable,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [ValidateSet('dark','light','navy','warm')][string]$Theme='light',
    [int]$ClientWidth=1600,
    [int]$ClientHeight=900,
    [ValidateSet(1.0,1.25,1.5)][double]$UiScale=1.0,
    [switch]$Populated,
    [switch]$ExportReport
)
$ErrorActionPreference='Stop'
$helperPath=Join-Path $PSScriptRoot 'probe_workflow_redesign_frozen.ps1'
$helper=[IO.File]::ReadAllText($helperPath)
$marker="try {`n    [void]`$p.Start()"
$normalized=$helper.Replace("`r`n","`n")
$offset=$normalized.IndexOf($marker,[StringComparison]::Ordinal)
if($offset -lt 0 -or $normalized.LastIndexOf($marker,[StringComparison]::Ordinal) -ne $offset) {
    throw 'UIA helper boundary changed; review the shared probe before proceeding.'
}
# The prefix defines only parameters, isolated process/profile preparation and
# safety helpers. It does not execute the legacy navigation or launch a process.
$prefix=$normalized.Substring(0,$offset).Replace("(Join-Path `$PSScriptRoot '..\..')","'C:\tmp\fdm-cu-live-fifth-round-20261020'")
. ([scriptblock]::Create($prefix)) -Executable $Executable -OutputRoot $OutputRoot -Theme $Theme -ClientWidth $ClientWidth -ClientHeight $ClientHeight -ExerciseBackgroundSaving:$false
if($validationRoot -ne 'C:\tmp\fdm-cu-live-fifth-round-20261020') {throw 'Use the fifth-round isolated checkout'}
$result['schema']=5
$p.StartInfo.EnvironmentVariables['QT_SCALE_FACTOR']=($UiScale/2.0).ToString([Globalization.CultureInfo]::InvariantCulture)
$result['ui_scale']=$UiScale
$result['native_device_scale']=2.0
$result['qt_scale_factor']=$UiScale/2.0
$result['populated']=[bool]$Populated
$result['capture_note']='VM host-camera sharing is temporarily disabled and restored after scoped installation validation. No Microview hardware acceptance.'
$result['steps']=@()
$result['fixture_sha256']=$null
$result['excel_export_tested']=$false
if($ExportReport -and -not $Populated){throw 'Export requires the populated fixture'}
$main=$null
$mainHandle=0
if($Populated) {
    # Evidence asset hashes plus atomic suffixes exceed MAX_PATH under the
    # nested run/case/timestamp log tree. Keep only the fixture in a short,
    # uniquely owned path; preserve the full run directory for audit output.
    $fixtureRoot=Join-Path $validationRoot ('build\fifth-round\fixtures\'+[guid]::NewGuid().ToString('N').Substring(0,12))
    [void](New-Item -ItemType Directory -Path $fixtureRoot)
    $probeProject=Join-Path $fixtureRoot 'review.fdmproj'
    $result['fixture_path']=$probeProject
    $python='C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
    $env:PYTHONPATH=Join-Path $validationRoot 'src'
    $env:FDM_FIFTH_FIXTURE=$probeProject
    $env:FDM_FIFTH_EXPORT=[string][int][bool]$ExportReport
    $fixture=@'
import os,json
from pathlib import Path
from fdm.models import ProjectState
from fdm.project_io import ProjectIO
from fdm.live_experiment import ExperimentSession, MaterialSnapshot, default_materials
from fdm.live_experiment.excel_template import ExportSnapshot,template_request
from fdm.live_experiment.evidence import EvidenceStore
from fdm.live_experiment.frames import FrameHandle
import numpy as np
p=ProjectState.empty()
s=ExperimentSession(sample_name="第五轮视觉验证",sample_number="0000123",workspace_options={"mode":"content"},evidence_options={"version":1,"mode":"crop_jpeg"})
materials=default_materials() if os.environ["FDM_FIFTH_EXPORT"] == "1" else [MaterialSnapshot(f"fiber-{n}",f"验证纤维 {n}",1.3) for n in range(1,9)]
for name in ("验证甲","验证乙","验证丙"):
    op=s.add_operator(name)
    for direction in ("warp","weft"):
        s.set_materials(materials,operator_id=op,direction=direction)
        for i in range(24):
            s.add_measurement(18+i/10,materials[i%len(materials)].id,operator_id=op,direction=direction,field_id=f"{name}-{direction}")
        s.adjust_count(materials[0].id,3,operator_id=op,direction=direction,field_id=f"{name}-{direction}")
if os.environ["FDM_FIFTH_EXPORT"] != "1":
    # Owned synthetic pixels validate persisted-crop review, never algorithm accuracy.
    store=EvidenceStore(Path(os.environ["FDM_FIFTH_FIXTURE"]).parent/"evidence-source")
    references=[]
    for i in range(12):
        pixels=np.full((576,768,3),205-i,dtype=np.uint8)
        width=20+i
        pixels[:,380:380+width]=70+i*2
        frame=FrameHandle.snapshot(pixels,source_id="installed-review-fixture",sequence=i,live=False)
        saved=store.reserve(frame,options=s.effective_evidence_options,points=[(380,288),(380+width,288)])
        references.append((saved["evidence_id"],width))
    written={item["evidence_id"]:item for item in store.close()}
    for i,(identity,width) in enumerate(references):
        s.add_measurement(width,materials[i%len(materials)].id,operator_id=op,direction="weft",field_id="fixture-crop",
            endpoints=((380,288),(380+width,288)),evidence=written[identity],
            provenance={"measurement_number":145+i,"original_diameter_um":width})
s.establish_history_baseline()
p.live_experiments.append(s)
ProjectIO.save(p,os.environ["FDM_FIFTH_FIXTURE"])
Path(os.environ["FDM_FIFTH_FIXTURE"]).with_suffix(".expected.json").write_text(json.dumps(template_request(ExportSnapshot.capture(s),"","",show=False),ensure_ascii=False),encoding="utf-8")
'@
    $fixturePath=Join-Path $runDirectory 'fixture.py'
    [IO.File]::WriteAllText($fixturePath,$fixture,$utf8)
    & $python $fixturePath
    if($LASTEXITCODE -ne 0) {throw 'Fixture creation failed'}
    $result.fixture_sha256=(Get-FileHash -LiteralPath $probeProject).Hash.ToLowerInvariant()
    $p.StartInfo.Arguments='"'+$probeProject+'"'
}
function Named($Name,$Type=$null) {
    return Visible-Unique @(Find-OwnNamed $Name $Type) $Name
}
function Click-Named($Name,$Type=$null) {
    [void](Invoke-Exact (Named $Name $Type))
    Start-Sleep -Milliseconds 150
    $result.steps+=@{action='invoke';name=$Name}
}
function Select-Trial($Name) {
    $tab=Visible-Unique @(Find-Within $main $Name ([System.Windows.Automation.ControlType]::TabItem)) $Name
    $tab.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
    Start-Sleep -Milliseconds 180
    $result.steps+=@{action='select';name=$Name}
}
function Confirm-ExcelEntry {
    $entry=Named '导出 Excel 报告' ([System.Windows.Automation.ControlType]::Button)
    $bounds=$entry.Current.BoundingRectangle
    $mainBounds=$main.Current.BoundingRectangle
    if($bounds.Top -lt $mainBounds.Top -or $bounds.Bottom -gt $mainBounds.Bottom) {throw 'Excel entry is clipped'}
    $result['excel_entry']=Get-ElementRecord $entry
}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class FifthRoundDialogControls {
    [StructLayout(LayoutKind.Sequential)] struct Rect { public int Left; public int Top; public int Right; public int Bottom; }
    [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr hwnd,out Rect rect);
    public static int[] NativeBounds(uint expected,long window) {
        var hwnd=new IntPtr(window); uint actual; GetWindowThreadProcessId(hwnd,out actual);
        if(actual!=expected) throw new InvalidOperationException("Refusing foreign bounds target");
        Rect rect; if(!GetWindowRect(hwnd,out rect)) throw new InvalidOperationException("Window rectangle unavailable");
        return new[]{rect.Left,rect.Top,rect.Right-rect.Left,rect.Bottom-rect.Top};
    }
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr hwnd,out uint pid);
    [DllImport("user32.dll")] static extern bool IsChild(IntPtr parent,IntPtr child);
    [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern int GetClassName(IntPtr hwnd,StringBuilder name,int count);
    [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern IntPtr SendMessage(IntPtr hwnd,uint msg,IntPtr wParam,string text);
    [DllImport("user32.dll")] static extern bool PostMessage(IntPtr hwnd,uint msg,IntPtr wParam,IntPtr lParam);
    [StructLayout(LayoutKind.Sequential)] struct Point { public int X; public int Y; }
    [DllImport("user32.dll")] static extern bool ScreenToClient(IntPtr hwnd,ref Point point);
    public static void Click(uint expected,long window,int screenX,int screenY) {
        var hwnd=new IntPtr(window); uint actual; GetWindowThreadProcessId(hwnd,out actual);
        if(actual!=expected) throw new InvalidOperationException("Refusing foreign click target");
        var point=new Point{X=screenX,Y=screenY};
        if(!ScreenToClient(hwnd,ref point)) throw new InvalidOperationException("Coordinate conversion failed");
        var pos=new IntPtr((point.Y<<16)|(point.X&0xffff));
        if(!PostMessage(hwnd,0x0200,IntPtr.Zero,pos) || !PostMessage(hwnd,0x0201,new IntPtr(1),pos)
           || !PostMessage(hwnd,0x0202,IntPtr.Zero,pos)) throw new InvalidOperationException("Native mouse event failed");
    }
    [DllImport("user32.dll")] static extern bool SetCursorPos(int x,int y);
    [DllImport("user32.dll")] static extern IntPtr WindowFromPoint(Point point);
    [DllImport("user32.dll")] static extern void mouse_event(uint flags,uint dx,uint dy,uint data,UIntPtr extra);
    public static void DoubleClick(uint expected,long window,int x,int y) {
        uint owner; GetWindowThreadProcessId(new IntPtr(window),out owner);
        if(owner!=expected) throw new InvalidOperationException("Refusing foreign album");
        var point=new Point{X=x,Y=y}; uint actual; GetWindowThreadProcessId(WindowFromPoint(point),out actual);
        if(actual!=expected) throw new InvalidOperationException("Album card covered by foreign window");
        if(!SetCursorPos(x,y)) throw new InvalidOperationException("Pointer placement failed");
        for(int i=0;i<2;i++) {
            GetWindowThreadProcessId(WindowFromPoint(point),out actual);
            if(actual!=expected) throw new InvalidOperationException("Click target changed");
            mouse_event(0x0002,0,0,0,UIntPtr.Zero); mouse_event(0x0004,0,0,0,UIntPtr.Zero);
            System.Threading.Thread.Sleep(75);
        }
    }
    static void Check(uint expected,long dialog,long handle,string wanted) {
        uint actual;GetWindowThreadProcessId(new IntPtr(handle),out actual);
        var name=new StringBuilder(128);GetClassName(new IntPtr(handle),name,name.Capacity);
        if(actual!=expected || !IsChild(new IntPtr(dialog),new IntPtr(handle)) || name.ToString()!=wanted)
            throw new InvalidOperationException("Refusing foreign or unexpected dialog control");
    }
    public static void SetFilename(uint pid,long dialog,long edit,string text) {
        Check(pid,dialog,edit,"Edit");SendMessage(new IntPtr(edit),0x000C,IntPtr.Zero,text);
    }
    public static void Save(uint pid,long dialog,long button) {
        Check(pid,dialog,button,"Button");
        if(!PostMessage(new IntPtr(button),0x00F5,IntPtr.Zero,IntPtr.Zero)) throw new InvalidOperationException("Save click failed");
    }
}
'@
function Exercise-ExcelExport {
    if(@(Get-Process EXCEL -ErrorAction SilentlyContinue).Count){throw 'Excel must be closed before this isolated export case'}
    $destination=Join-Path $runDirectory 'fixed-entry-report.xls'
    Click-Named '导出 Excel 报告' ([System.Windows.Automation.ControlType]::Button)
    $deadline=[DateTime]::UtcNow.AddSeconds(15)
    do {
        Start-Sleep -Milliseconds 100
        $dialogs=@(Get-OwnedRoots|Where-Object {$_.window.Visible -and $_.window.Title -eq '生成模板报告'})
    } while($dialogs.Count -ne 1 -and [DateTime]::UtcNow -lt $deadline)
    if($dialogs.Count -ne 1){throw 'Direct Excel entry did not open the save dialog'}
    $dialog=$dialogs[0].element
    Save-Snapshot '09-excel-save-dialog'
    $edits=@($dialog.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.Condition]::TrueCondition)|Where-Object {
        $_.Current.ProcessId -eq $p.Id -and $_.Current.ClassName -eq 'Edit' -and $_.Current.AutomationId -eq '1001'
    })
    if($edits.Count -ne 1){throw 'Expected one owned filename edit'}
    [FifthRoundDialogControls]::SetFilename([uint32]$p.Id,$dialogs[0].window.Handle,$edits[0].Current.NativeWindowHandle,$destination)
    $save=@($dialog.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.Condition]::TrueCondition)|Where-Object {
        $_.Current.ProcessId -eq $p.Id -and $_.Current.ClassName -eq 'Button' -and $_.Current.AutomationId -eq '1' -and $_.Current.Name -match '^(保存|Save)'
    })
    if($save.Count -ne 1){throw 'Expected one owned Save button'}
    [FifthRoundDialogControls]::Save([uint32]$p.Id,$dialogs[0].window.Handle,$save[0].Current.NativeWindowHandle)
    $deadline=[DateTime]::UtcNow.AddSeconds(180)
    do {Start-Sleep -Milliseconds 250} while(-not (Test-Path -LiteralPath $destination) -and [DateTime]::UtcNow -lt $deadline)
    if(-not (Test-Path -LiteralPath $destination)){throw 'Direct Excel report did not appear before deadline'}
    Start-Sleep -Milliseconds 1200
    $template=Join-Path $appConfig 'templates\sheet.xlt'
    $expected=Join-Path $runDirectory 'expected.json'
    Copy-Item -LiteralPath ([IO.Path]::ChangeExtension($probeProject,'expected.json')) -Destination $expected
    $audit=Join-Path $runDirectory 'excel-verification.json'
    & (Join-Path $validationRoot 'scripts\validation\verify_acquisition_fifth_round_excel.ps1') -Report $destination -Template $template -Expected $expected -Output $audit
    $verified=[IO.File]::ReadAllText($audit)|ConvertFrom-Json
    if(-not $verified.passed){throw 'Excel workbook audit failed'}
    $result['excel_export_tested']=$true
    $result['excel_verification']=$verified
    Save-Snapshot '10-excel-success'
}
function Confirm-Navigation {
    [void](Named '采集与试验' ([System.Windows.Automation.ControlType]::CheckBox))
    foreach($name in @('实时采集','含量试验','直径试验')) {
        [void](Visible-Unique @(Find-Within $main $name ([System.Windows.Automation.ControlType]::TabItem)) $name)
    }
}
function Confirm-FifthEntries {
    foreach($name in @('全图分析或刷新候选','候选显示与不透明度','证据相册及存档设置')) {
        $entry=Named $name
        if($entry.Current.IsOffscreen){throw "Fifth-round entry is offscreen: $name"}
        $result.steps+=@{action='fifth-entry-visible';name=$name;bounds=(Get-ElementRecord $entry)}
    }
    $auto=Named '稳定后自动分析' ([System.Windows.Automation.ControlType]::CheckBox)
    $toggle=$auto.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern)
    if($toggle.Current.ToggleState -ne [System.Windows.Automation.ToggleState]::Off){throw 'Automatic analysis must default off'}
    $toggle.Toggle();Start-Sleep -Milliseconds 120
    if($toggle.Current.ToggleState -ne [System.Windows.Automation.ToggleState]::On){throw 'Automatic control did not enable'}
    $toggle.Toggle();Start-Sleep -Milliseconds 120
    $result['automatic_toggle_verified']=$true
}
function Save-OwnedWindow($Element,[string]$Name) {
    if($Element.Current.ProcessId -ne $p.Id){throw 'Refusing foreign window capture'}
    $r=$Element.Current.BoundingRectangle
    if($r.Width -lt 1 -or $r.Height -lt 1){throw 'Owned window has no capture area'}
    $bitmap=[Drawing.Bitmap]::new([int]$r.Width,[int]$r.Height)
    $graphics=[Drawing.Graphics]::FromImage($bitmap)
    try{$graphics.CopyFromScreen([int]$r.X,[int]$r.Y,0,0,$bitmap.Size);$bitmap.Save((Join-Path $runDirectory ($Name+'.png')),[Drawing.Imaging.ImageFormat]::Png)}
    finally{$graphics.Dispose();$bitmap.Dispose()}
    $bounds=[FifthRoundDialogControls]::NativeBounds([uint32]$p.Id,[long]$Element.Current.NativeWindowHandle)
    $native=[Drawing.Rectangle]::new($bounds[0],$bounds[1],$bounds[2],$bounds[3])
    $screen=[Windows.Forms.Screen]::FromHandle([IntPtr]$Element.Current.NativeWindowHandle)
    $intersection=[Drawing.Rectangle]::Intersect($native,$screen.WorkingArea)
    $result[($Name+'-native-geometry')]=@{bounds=$bounds;work_area=@($screen.WorkingArea.X,$screen.WorkingArea.Y,$screen.WorkingArea.Width,$screen.WorkingArea.Height);intersection=@($intersection.X,$intersection.Y,$intersection.Width,$intersection.Height);fully_inside_work_area=$screen.WorkingArea.Contains($native);source='GetWindowRect';moved_for_capture=$false}
    $nativeBitmap=[Drawing.Bitmap]::new($native.Width,$native.Height)
    $nativeGraphics=[Drawing.Graphics]::FromImage($nativeBitmap)
    try{$nativeGraphics.CopyFromScreen($native.X,$native.Y,0,0,$nativeBitmap.Size);$nativeBitmap.Save((Join-Path $runDirectory ($Name+'-native.png')),[Drawing.Imaging.ImageFormat]::Png)}
    finally{$nativeGraphics.Dispose();$nativeBitmap.Dispose()}
}
function Exercise-EvidenceReview {
    Click-Named '证据相册及存档设置'
    $deadline=[DateTime]::UtcNow.AddSeconds(10)
    do {Start-Sleep -Milliseconds 100;$albums=@(Get-OwnedRoots|Where-Object {$_.window.Visible -and $_.window.Title -like '证据相册 · *'})}
    while($albums.Count -ne 1 -and [DateTime]::UtcNow -lt $deadline)
    if($albums.Count -ne 1){throw 'Evidence album did not appear'}
    $album=$albums[0]
    $view=Visible-Unique @(Find-Within $album.element '测量证据相册') 'Evidence album list'
    foreach($label in @('操作者','方向','纤维类别','证据状态')){
        [void](Visible-Unique @(Find-Within $album.element $label ([System.Windows.Automation.ControlType]::ComboBox)) $label)
    }
    [void](Visible-Unique @(Find-Within $album.element '回到最新图像证据') 'Album latest')
    [void](Visible-Unique @(Find-Within $album.element '缩略图大小') 'Album thumbnail size')
    Start-Sleep -Milliseconds 700
    $items=@($view.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::ControlTypeProperty,[System.Windows.Automation.ControlType]::ListItem)))
    if($items.Count -lt 12){throw 'Expected all twelve persisted evidence cards'}
    $result['evidence_album_item_count']=$items.Count
    Save-Snapshot '04a-evidence-album'
    Save-OwnedWindow $album.element '04a-evidence-album-window'
    if(-not $result['04a-evidence-album-window-native-geometry'].fully_inside_work_area){
        throw 'Normal evidence album native frame extends beyond the monitor work area'
    }
    $windowPattern=$null
    if($album.element.TryGetCurrentPattern([System.Windows.Automation.WindowPattern]::Pattern,[ref]$windowPattern) -and $windowPattern.Current.CanMaximize){
        $windowPattern.SetWindowVisualState([System.Windows.Automation.WindowVisualState]::Maximized)
        Start-Sleep -Milliseconds 200
        Save-Snapshot '04b-evidence-album-maximized'
        Save-OwnedWindow $album.element '04b-evidence-album-window-maximized'
        $windowPattern.SetWindowVisualState([System.Windows.Automation.WindowVisualState]::Normal)
        Start-Sleep -Milliseconds 200
        Save-OwnedWindow $album.element '04b-evidence-album-window-restored'
        $before=$result['04a-evidence-album-window-native-geometry'].bounds
        $restored=$result['04b-evidence-album-window-restored-native-geometry']
        if(-not $restored.fully_inside_work_area -or $restored.bounds[2] -ne $before[2] -or $restored.bounds[3] -ne $before[3]){
            throw 'Evidence album did not restore its normal native size and visible frame'
        }
    }
    Start-Sleep -Milliseconds 250
    # Qt's QListView UIA SetFocus is unsupported in this installed runtime.
    # Select a visible historical card with actual native double-click events.
    $target=@($items|Where-Object{-not $_.Current.IsOffscreen})[1]
    if(-not $target -or $target.Current.ProcessId -ne $p.Id){throw 'Historical album card not visible'}
    $bounds=$target.Current.BoundingRectangle
    $result['historical_evidence_card']=$target.Current.Name
    [FifthRoundDialogControls]::DoubleClick([uint32]$p.Id,$album.window.Handle,[int]($bounds.Left+$bounds.Width/2),[int]($bounds.Top+$bounds.Height/2))
    $deadline=[DateTime]::UtcNow.AddSeconds(10)
    do {Start-Sleep -Milliseconds 100;$pins=@(Find-OwnNamed '固定为小窗' ([System.Windows.Automation.ControlType]::CheckBox)|Where-Object {-not $_.Current.IsOffscreen})}
    while($pins.Count -ne 1 -and [DateTime]::UtcNow -lt $deadline)
    if($pins.Count -ne 1){throw 'Evidence review did not appear'}
    Save-Snapshot '04c-evidence-review'
    $pins[0].GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern).Toggle()
    Start-Sleep -Milliseconds 200
    if(@(Find-OwnNamed '显示测量线' ([System.Windows.Automation.ControlType]::CheckBox)|Where-Object {-not $_.Current.IsOffscreen}).Count -lt 1){throw 'Evidence annotation toggle missing'}
    Save-Snapshot '04d-evidence-pinned'
    Click-Named '收起' ([System.Windows.Automation.ControlType]::Button)
    [void][CuProbeNative]::Close([uint32]$p.Id,$album.window.Handle)
    Start-Sleep -Milliseconds 150
    $result['cropped_evidence_review_verified']=$true
    $result['evidence_album_verified']=$true
}
function Confirm-MixedAndStorage {
    # Exercise a real native mouse press on the UIA-located InstantPopup
    # button: Qt's InvokePattern calls click() without the popup press path.
    $tools=Named '试验工具与设置'
    Ensure-OwnedVisible $tools
    $bounds=$tools.Current.BoundingRectangle
    [FifthRoundDialogControls]::Click([uint32]$p.Id,$mainHandle,[int]($bounds.Left+$bounds.Width/2),[int]($bounds.Top+$bounds.Height/2))
    $deadline=[DateTime]::UtcNow.AddSeconds(4)
    do{Start-Sleep -Milliseconds 100;$advanced=@(Find-OwnNamed '高级选项' ([System.Windows.Automation.ControlType]::MenuItem)|Where-Object{-not $_.Current.IsOffscreen})}while($advanced.Count -ne 1 -and [DateTime]::UtcNow -lt $deadline)
    $result.steps+=@{action='native-mouse-menu';name='试验工具与设置'}
    Click-Named '高级选项' ([System.Windows.Automation.ControlType]::MenuItem)
    $mode=Named '试验交互模式' ([System.Windows.Automation.ControlType]::ComboBox)
    $value=$mode.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern).Current.Value
    if($value -notlike '混合*'){throw 'Mixed input must be the default'}
    $result['mixed_input_value']=$value
    $storage=Named '测量图片存档' ([System.Windows.Automation.ControlType]::ComboBox)
    $expand=$storage.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern)
    $expand.Expand();Start-Sleep -Milliseconds 150
    foreach($title in @('不保存','局部 JPEG · 质量 95','完整无损 PNG')){
        [void](Visible-Unique @(Find-OwnNamed $title ([System.Windows.Automation.ControlType]::ListItem)) $title)
    }
    Save-Snapshot '03a-advanced-storage-options'
    $expand.Collapse()
    $result['mixed_input_default_verified']=$true
    $result['advanced_storage_options_verified']=$true
    Click-Named '关闭' ([System.Windows.Automation.ControlType]::Button)
}
try {
    [void]$p.Start()
    $result.pid=$p.Id
    $deadline=[DateTime]::UtcNow.AddSeconds(35)
    do {
        Start-Sleep -Milliseconds 200
        $p.Refresh()
        $windows=@([CuProbeNative]::Windows([uint32]$p.Id) | Where-Object {$_.Visible -and $_.Title -eq '显微测量工作台'})
    } while(-not $p.HasExited -and $windows.Count -ne 1 -and [DateTime]::UtcNow -lt $deadline)
    if($p.HasExited -or $windows.Count -ne 1) {throw 'Expected one main window'}
    $mainHandle=$windows[0].Handle
    $main=[System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$mainHandle)
    $result.main_window=$true
    [void][CuProbeNative]::ResizeClient($mainHandle,$ClientWidth,$ClientHeight)
    Start-Sleep -Milliseconds 250
    $result['os_dpi']=[CuProbeNative]::GetDpiForWindow([IntPtr]$mainHandle)
    Save-Snapshot '01-main'
    Click-Named '采集与试验' ([System.Windows.Automation.ControlType]::CheckBox)
    Save-Snapshot '02-realtime'
    Select-Trial '含量试验'
    Confirm-Navigation
    Confirm-ExcelEntry
    Confirm-FifthEntries
    Confirm-MixedAndStorage
    Save-Snapshot '03-content'
    if(-not $Populated) {
        [void](Named '选择纤维类别' ([System.Windows.Automation.ControlType]::Button))
        $result['empty_category_entry_verified']=$true
        Click-Named '返回采集' ([System.Windows.Automation.ControlType]::Button)
        Select-Trial '含量试验'
        [void](Named '选择纤维类别' ([System.Windows.Automation.ControlType]::Button))
        $result['empty_return_verified']=$true
    } elseif(-not $ExportReport) {
        Exercise-EvidenceReview
        Click-Named '记录' ([System.Windows.Automation.ControlType]::CheckBox)
        $operator=Named '记录操作者筛选' ([System.Windows.Automation.ControlType]::ComboBox)
        $direction=Named '记录方向筛选' ([System.Windows.Automation.ControlType]::ComboBox)
        $result['record_operator_filter']=Get-ElementRecord $operator
        $result['record_direction_filter']=Get-ElementRecord $direction
        $tables=@(Find-OwnNamed '' ([System.Windows.Automation.ControlType]::Table) | Where-Object {-not $_.Current.IsOffscreen})
        $result['visible_record_tables']=@($tables | ForEach-Object {Get-ElementRecord $_})
        foreach($filter in @($operator,$direction)) {
            $selection=$null
            if($filter.TryGetCurrentPattern([System.Windows.Automation.SelectionPattern]::Pattern,[ref]$selection)) {
                $selected=@($selection.GetCurrentSelection() | ForEach-Object {$_.Current.Name})
                $result.steps+=@{action='filter-selection';name=$filter.Current.Name;selected=$selected}
                if($selected.Count -ne 1 -or $selected[0] -notmatch '^全部') {throw 'Records did not default to all scopes'}
            }
        }
        Save-Snapshot '04-records-default'
        foreach($placement in @('停靠底部','停靠左侧','浮动窗口')) {
            Click-Named '位置' ([System.Windows.Automation.ControlType]::Button)
            Click-Named $placement ([System.Windows.Automation.ControlType]::MenuItem)
            Confirm-Navigation
            Confirm-ExcelEntry
            Save-Snapshot ('05-'+$placement)
        }
        # Returning to realtime must hide the owned floating records window.
        Click-Named '返回采集' ([System.Windows.Automation.ControlType]::Button)
        if(@(Find-OwnNamed '记录操作者筛选' | Where-Object {-not $_.Current.IsOffscreen}).Count) {throw 'Records window persisted over realtime'}
        Save-Snapshot '06-realtime-return'
        Select-Trial '含量试验'
        [void](Named '记录操作者筛选' ([System.Windows.Automation.ControlType]::ComboBox))
        Click-Named '记录' ([System.Windows.Automation.ControlType]::CheckBox)
        Click-Named '展开或收起样品信息' ([System.Windows.Automation.ControlType]::CheckBox)
        Confirm-ExcelEntry
        Save-Snapshot '07-sample-expanded'
    }
    if($ExportReport){Exercise-ExcelExport}
    for($i=0;$i -lt 5 -and -not $ExportReport;$i++) {
        foreach($name in @('直径试验','实时采集','含量试验')) {Select-Trial $name; Confirm-Navigation}
    }
    Save-Snapshot '08-navigation-return'
    Click-Named '测量分析' ([System.Windows.Automation.ControlType]::CheckBox)
    [void][CuProbeNative]::Close([uint32]$p.Id,$mainHandle)
    $result.graceful_exit=$p.WaitForExit(12000)
    if(-not $result.graceful_exit) {Save-Snapshot '99-close-failure';throw 'Untouched/loaded trial did not close cleanly'}
    $result['exit_code']=$p.ExitCode
    if($p.ExitCode -ne 0) {throw 'Candidate exited with an error'}
    if($Populated -and (Get-FileHash -LiteralPath $probeProject).Hash.ToLowerInvariant() -ne $result.fixture_sha256) {throw 'Read-only navigation rewrote the owned fixture'}
    $result.ok=$true
} catch {
    $result.errors+=@($_.Exception.Message)
    $result['failure_stack']=$_.ScriptStackTrace
    try {Save-Snapshot '99-failure'} catch {$result.errors+=@($_.Exception.Message)}
} finally {
    if($result.pid) {
        $p.Refresh()
        if(-not $p.HasExited) {$p.Kill();$result.forced_cleanup=$true;[void]$p.WaitForExit(5000)}
        $result['process_exited_after_cleanup']=$p.HasExited
        $p.Dispose()
    }
    $result['finished_at']=(Get-Date).ToString('o')
    Write-JsonFile (Join-Path $runDirectory 'result.json') $result
    [Console]::WriteLine('FDM_FIFTH_GUI_RESULT='+($result|ConvertTo-Json -Depth 14 -Compress))
}
if(-not $result.ok) {exit 1}
