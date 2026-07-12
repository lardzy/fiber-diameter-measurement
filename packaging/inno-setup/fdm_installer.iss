#define MyAppName "Fiber Diameter Measurement"
#include "version.auto.iss"
#define MyAppPublisher "LARD"
#define MyAppExeName "FiberDiameterMeasurement.exe"
#define MyAppShortcutName "特纤通用测量工具"
#define ProjectProgId "LARD.FiberDiameterMeasurement.Project"
#define DigitalSlideProgId "LARD.FiberDiameterMeasurement.DigitalSlide"
#define ProjectRoot AddBackslash(SourcePath) + "..\.."
#define MyAppSourceDir ProjectRoot + "\dist\windows\FiberDiameterMeasurement"
#define MyAppOutputDir ProjectRoot + "\dist\installer"
#define MyAppOutputBaseFilename "fiber-diameter-measurement-setup-" + MyAppVersion
#define MyAppIconFile ProjectRoot + "\packaging\assets\icons\app-icon.ico"

#ifnexist MyAppSourceDir + "\" + MyAppExeName
  #error "PyInstaller output not found. Build dist/windows/FiberDiameterMeasurement first."
#endif

#ifnexist MyAppSourceDir + "\release-manifest.json"
  #error "release-manifest.json not found. Build the onedir package through scripts/build_windows_onedir.py."
#endif

#ifnexist MyAppSourceDir + "\build-id.txt"
  #error "build-id.txt not found. Build the onedir package through scripts/build_windows_onedir.py."
#endif

#ifnexist MyAppIconFile
  #error "Application icon not found. Expected packaging/assets/icons/app-icon.ico."
#endif

[Setup]
AppId={{F0F6A8B5-4838-4DF0-B2C1-18F1D5AA4A66}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
SetupIconFile={#MyAppIconFile}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir={#MyAppOutputDir}
OutputBaseFilename={#MyAppOutputBaseFilename}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}
ChangesEnvironment=no
ChangesAssociations=yes
DisableProgramGroupPage=no
CloseApplications=yes
CloseApplicationsFilter=*.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}";
Name: "fileassoc"; Description: "关联 .fdmproj 项目和 .fdmslide 数字化切片"; GroupDescription: "文件关联:"; Flags: checkedonce

[Files]
Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppShortcutName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 {#MyAppShortcutName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppShortcutName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
Root: HKLM; Subkey: "Software\Classes\.fdmproj"; ValueType: string; ValueName: ""; ValueData: "{#ProjectProgId}"; Flags: uninsdeletekeyifempty; Tasks: fileassoc
Root: HKLM; Subkey: "Software\Classes\.fdmproj\OpenWithProgids"; ValueType: string; ValueName: "{#ProjectProgId}"; ValueData: ""; Flags: uninsdeletevalue uninsdeletekeyifempty; Tasks: fileassoc
Root: HKLM; Subkey: "Software\Classes\{#ProjectProgId}"; ValueType: string; ValueName: ""; ValueData: "Fiber 测量项目"; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKLM; Subkey: "Software\Classes\{#ProjectProgId}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"; Tasks: fileassoc
Root: HKLM; Subkey: "Software\Classes\{#ProjectProgId}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: fileassoc

Root: HKLM; Subkey: "Software\Classes\.fdmslide"; ValueType: string; ValueName: ""; ValueData: "{#DigitalSlideProgId}"; Flags: uninsdeletekeyifempty; Tasks: fileassoc
Root: HKLM; Subkey: "Software\Classes\.fdmslide\OpenWithProgids"; ValueType: string; ValueName: "{#DigitalSlideProgId}"; ValueData: ""; Flags: uninsdeletevalue uninsdeletekeyifempty; Tasks: fileassoc
Root: HKLM; Subkey: "Software\Classes\{#DigitalSlideProgId}"; ValueType: string; ValueName: ""; ValueData: "Fiber 数字化切片"; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKLM; Subkey: "Software\Classes\{#DigitalSlideProgId}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"; Tasks: fileassoc
Root: HKLM; Subkey: "Software\Classes\{#DigitalSlideProgId}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: fileassoc

Root: HKLM; Subkey: "Software\Classes\Applications\{#MyAppExeName}"; ValueType: string; ValueName: "FriendlyAppName"; ValueData: "{#MyAppName}"; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKLM; Subkey: "Software\Classes\Applications\{#MyAppExeName}\SupportedTypes"; ValueType: string; ValueName: ".fdmproj"; ValueData: ""; Flags: uninsdeletevalue uninsdeletekeyifempty; Tasks: fileassoc
Root: HKLM; Subkey: "Software\Classes\Applications\{#MyAppExeName}\SupportedTypes"; ValueType: string; ValueName: ".fdmslide"; ValueData: ""; Flags: uninsdeletevalue uninsdeletekeyifempty; Tasks: fileassoc
Root: HKLM; Subkey: "Software\Classes\Applications\{#MyAppExeName}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Flags: uninsdeletekey; Tasks: fileassoc

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
procedure RemoveOwnedExtensionDefault(const ExtensionName, ProgId: String);
var
  CurrentProgId: String;
  ExtensionKey: String;
begin
  ExtensionKey := 'Software\Classes\' + ExtensionName;
  if RegQueryStringValue(HKEY_LOCAL_MACHINE, ExtensionKey, '', CurrentProgId) and
     (CompareText(CurrentProgId, ProgId) = 0) then
  begin
    RegDeleteValue(HKEY_LOCAL_MACHINE, ExtensionKey, '');
    RegDeleteKeyIfEmpty(HKEY_LOCAL_MACHINE, ExtensionKey);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
  begin
    RemoveOwnedExtensionDefault('.fdmproj', '{#ProjectProgId}');
    RemoveOwnedExtensionDefault('.fdmslide', '{#DigitalSlideProgId}');
  end;
end;
