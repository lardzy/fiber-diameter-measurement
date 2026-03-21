#define MyAppName "Fiber Diameter Measurement"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Codex"
#define MyAppExeName "FiberDiameterMeasurement.exe"
#define ProjectRoot AddBackslash(SourcePath) + "..\.."
#define MyAppSourceDir ProjectRoot + "\dist\windows\FiberDiameterMeasurement"
#define MyAppOutputDir ProjectRoot + "\dist\installer"
#define MyAppOutputBaseFilename "fiber-diameter-measurement-setup-" + MyAppVersion

#ifnexist "{#MyAppSourceDir}\{#MyAppExeName}"
  #error "PyInstaller output not found. Build dist/windows/FiberDiameterMeasurement first."
#endif

[Setup]
AppId={{F0F6A8B5-4838-4DF0-B2C1-18F1D5AA4A66}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
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
DisableProgramGroupPage=no
CloseApplications=yes
CloseApplicationsFilter=*.exe

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent
