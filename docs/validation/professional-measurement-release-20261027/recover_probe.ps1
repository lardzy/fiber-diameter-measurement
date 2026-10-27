$ErrorActionPreference='Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$p=Get-Process -Id 6520
if(-not $p.Path.StartsWith('C:\tmp\fdm-professional-measurement-release-20261027\dist\windows\installer-payload-330c059e82e6\{app}\')){throw 'Wrong test process'}
$pidCondition=[System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::ProcessIdProperty,6520)
$dialogs=@([System.Windows.Automation.AutomationElement]::RootElement.FindAll([System.Windows.Automation.TreeScope]::Descendants,$pidCondition)|Where-Object{$_.Current.Name -eq '未保存的改动' -and $_.Current.ControlType -eq [System.Windows.Automation.ControlType]::Window})
if($dialogs.Count -ne 1){throw 'Expected one owned save prompt'}
$buttons=@($dialogs[0].FindAll([System.Windows.Automation.TreeScope]::Descendants,$pidCondition)|Where-Object{$_.Current.Name -eq '保存' -and $_.Current.ControlType -eq [System.Windows.Automation.ControlType]::Button})
if($buttons.Count -ne 1){throw 'Ambiguous dialog save button'}
$buttons[0].GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
[void]$p.WaitForExit(30000)
if(-not $p.HasExited){throw 'Owned process did not close normally'}
@{pid=6520;normal_exit=$true;forced_termination=$false;action='Invoked Save in owned unsaved-changes prompt';at=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json
