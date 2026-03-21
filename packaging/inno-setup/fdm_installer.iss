#define MyAppName "Fiber Diameter Measurement"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "LARD"
#define MyAppExeName "特纤通用直径测量工具v0_1.exe"
#define MyAppShortcutName "特纤通用直径测量工具v0_1"
#define ProjectRoot AddBackslash(SourcePath) + "..\.."
#define MyAppSourceDir ProjectRoot + "\dist\windows\FiberDiameterMeasurement"
#define MyAppOutputDir ProjectRoot + "\dist\installer"
#define MyAppOutputBaseFilename "fiber-diameter-measurement-setup-" + MyAppVersion

#ifnexist MyAppSourceDir + "\" + MyAppExeName
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
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}";

[Files]
Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppShortcutName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 {#MyAppShortcutName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppShortcutName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent
