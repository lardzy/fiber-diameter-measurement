; Inno Setup Simplified Chinese messages for Fiber Diameter Measurement.
; UTF-8 with BOM. Message IDs follow Inno Setup 6.5.0 Default.isl.
; Also checked against 6.7.3 and 7.1.0 (same message IDs and placeholders).
; Japanese.isl supplied with this task was used as a structural reference.
; Keep %1, %2, %n and [name]/[name/ver]/[gb]/[mb] placeholders intact.

[LangOptions]
LanguageName=简体中文
LanguageID=$0804
LanguageCodePage=936
DialogFontName=Microsoft YaHei UI
DialogFontSize=9
WelcomeFontName=Microsoft YaHei UI
WelcomeFontSize=12

[Messages]

; Application titles
SetupAppTitle=安装程序
SetupWindowTitle=%1 安装程序
UninstallAppTitle=卸载程序
UninstallAppFullTitle=%1 卸载程序

; Common messages
InformationTitle=信息
ConfirmTitle=确认
ErrorTitle=错误

; Setup loader
SetupLdrStartupMessage=即将安装 %1。是否继续？
LdrCannotCreateTemp=无法创建临时文件。安装已中止。
LdrCannotExecTemp=无法执行临时目录中的文件。安装已中止。

; Startup errors
LastErrorMessage=%1。%n%n错误 %2：%3
SetupFileMissing=安装目录中缺少文件 %1。请修复此问题或重新获取安装程序。
SetupFileCorrupt=安装文件已损坏。请重新获取安装程序。
SetupFileCorruptOrWrongVer=安装文件已损坏或与当前安装程序版本不兼容。请修复此问题或重新获取安装程序。
InvalidParameter=命令行中包含无效参数：%n%n%1
SetupAlreadyRunning=安装程序已在运行。
WindowsVersionNotSupported=此程序不支持当前的 Windows 版本。
WindowsServicePackRequired=此程序需要 %1 Service Pack %2 或更高版本。
NotOnThisPlatform=此程序无法在 %1 上运行。
OnlyOnThisPlatform=此程序必须在 %1 上运行。
OnlyOnTheseArchitectures=此程序只能安装在支持以下处理器架构的 Windows 版本上：%n%n%1
WinVersionTooLowError=此程序需要 %1 %2 或更高版本。
WinVersionTooHighError=此程序无法安装在 %1 %2 或更高版本上。
AdminPrivilegesRequired=安装此程序需要以管理员身份登录。
PowerUserPrivilegesRequired=安装此程序需要以管理员或高级用户身份登录。
SetupAppRunningError=检测到 %1 正在运行。%n%n请关闭该程序的所有窗口，然后单击“确定”继续；或单击“取消”退出安装程序。
UninstallAppRunningError=检测到 %1 正在运行。%n%n请关闭该程序的所有窗口，然后单击“确定”继续；或单击“取消”退出卸载程序。

; Installation mode
PrivilegesRequiredOverrideTitle=选择安装模式
PrivilegesRequiredOverrideInstruction=请选择安装模式
PrivilegesRequiredOverrideText1=可以为所有用户安装 %1（需要管理员权限），也可以仅为当前用户安装。
PrivilegesRequiredOverrideText2=可以仅为当前用户安装 %1，也可以为所有用户安装（需要管理员权限）。
PrivilegesRequiredOverrideAllUsers=为所有用户安装(&A)
PrivilegesRequiredOverrideAllUsersRecommended=为所有用户安装(&A)（推荐）
PrivilegesRequiredOverrideCurrentUser=仅为当前用户安装(&M)
PrivilegesRequiredOverrideCurrentUserRecommended=仅为当前用户安装(&M)（推荐）

; Directory errors
ErrorCreatingDir=创建目录“%1”时出错。
ErrorTooManyFilesInDir=无法在目录“%1”中创建文件，因为该目录中的文件过多。

; Setup common messages
ExitSetupTitle=退出安装
ExitSetupMessage=安装尚未完成。现在退出将无法完成程序安装。%n%n以后可以再次运行安装程序来完成安装。%n%n是否退出安装？
AboutSetupMenuItem=关于安装程序(&A)...
AboutSetupTitle=关于安装程序
AboutSetupMessage=%1 %2%n%3%n%n%1 网站：%n%4
AboutSetupNote=
TranslatorNote=
HelpTextNote=

; Buttons
ButtonBack=< 上一步(&B)
ButtonNext=下一步(&N) >
ButtonInstall=安装(&I)
ButtonOK=确定
ButtonCancel=取消
ButtonYes=是(&Y)
ButtonYesToAll=全部选是(&A)
ButtonNo=否(&N)
ButtonNoToAll=全部选否(&O)
ButtonFinish=完成(&F)
ButtonBrowse=浏览(&B)...
ButtonWizardBrowse=浏览(&R)...
ButtonNewFolder=新建文件夹(&M)

; Language selection
SelectLanguageTitle=选择安装语言
SelectLanguageLabel=请选择安装过程中使用的语言。

; Common wizard text
ClickNext=单击“下一步”继续，或单击“取消”退出安装程序。
BeveledLabel=
BrowseDialogTitle=浏览文件夹
BrowseDialogLabel=请在下方列表中选择文件夹，然后单击“确定”。
NewFolderName=新建文件夹

; Welcome
WelcomeLabel1=欢迎使用 [name] 安装向导
WelcomeLabel2=此向导将在您的计算机上安装 [name/ver]。%n%n建议在继续之前关闭其他所有应用程序。

; Password
WizardPassword=密码
PasswordLabel1=此安装程序受密码保护。
PasswordLabel3=请输入密码，然后单击“下一步”继续。密码区分大小写。
PasswordEditLabel=密码(&P)：
IncorrectPassword=输入的密码不正确，请重试。

; License agreement
WizardLicense=许可协议
LicenseLabel=请在继续之前阅读以下重要信息。
LicenseLabel3=请阅读以下许可协议。必须接受此协议才能继续安装。
LicenseAccepted=我接受此协议(&A)
LicenseNotAccepted=我不接受此协议(&D)

; Information
WizardInfoBefore=信息
InfoBeforeLabel=请在继续之前阅读以下重要信息。
InfoBeforeClickLabel=准备好继续安装时，请单击“下一步”。
WizardInfoAfter=信息
InfoAfterLabel=请在继续之前阅读以下重要信息。
InfoAfterClickLabel=准备好继续安装时，请单击“下一步”。

; User information
WizardUserInfo=用户信息
UserInfoDesc=请输入您的信息。
UserInfoName=用户名(&U)：
UserInfoOrg=组织(&O)：
UserInfoSerial=序列号(&S)：
UserInfoNameRequired=请输入用户名。

; Destination directory
WizardSelectDir=选择安装位置
SelectDirDesc=您希望将 [name] 安装到哪里？
SelectDirLabel3=安装程序将把 [name] 安装到以下文件夹。
SelectDirBrowseLabel=单击“下一步”继续。如需选择其他文件夹，请单击“浏览”。
DiskSpaceGBLabel=至少需要 [gb] GB 可用磁盘空间。
DiskSpaceMBLabel=至少需要 [mb] MB 可用磁盘空间。
CannotInstallToNetworkDrive=无法安装到网络驱动器。
CannotInstallToUNCPath=无法安装到 UNC 路径。
InvalidPath=请输入包含驱动器盘符的完整路径，例如：%n%nC:\APP%n%n或输入以下格式的 UNC 路径：%n%n\\server\share
InvalidDrive=所选驱动器或 UNC 共享不存在或无法访问。请选择其他位置。
DiskSpaceWarningTitle=磁盘空间不足
DiskSpaceWarning=安装至少需要 %1 KB 可用磁盘空间，但所选驱动器只有 %2 KB 可用空间。%n%n是否仍要继续？
DirNameTooLong=文件夹名称或路径过长。
InvalidDirName=文件夹名称无效。
BadDirName32=文件夹名称不能包含以下字符：%n%n%1
DirExistsTitle=文件夹已存在
DirExists=文件夹%n%n%1%n%n已存在。是否仍要安装到此文件夹？
DirDoesntExistTitle=文件夹不存在
DirDoesntExist=文件夹%n%n%1%n%n不存在。是否创建此文件夹？

; Components
WizardSelectComponents=选择组件
SelectComponentsDesc=您希望安装哪些组件？
SelectComponentsLabel2=请勾选要安装的组件，并取消勾选不需要的组件。单击“下一步”继续。
FullInstallation=完整安装
CompactInstallation=精简安装
CustomInstallation=自定义安装
NoUninstallWarningTitle=已有组件
NoUninstallWarning=检测到以下组件已安装：%n%n%1%n%n取消勾选这些组件不会将其卸载。%n%n是否仍要继续？
ComponentSize1=%1 KB
ComponentSize2=%1 MB
ComponentsDiskSpaceGBLabel=当前所选组件至少需要 [gb] GB 可用磁盘空间。
ComponentsDiskSpaceMBLabel=当前所选组件至少需要 [mb] MB 可用磁盘空间。

; Additional tasks
WizardSelectTasks=选择附加任务
SelectTasksDesc=您希望执行哪些附加任务？
SelectTasksLabel2=请选择安装 [name] 时需要执行的附加任务，然后单击“下一步”。

; Start Menu folder
WizardSelectProgramGroup=选择开始菜单文件夹
SelectStartMenuFolderDesc=您希望在哪里创建程序快捷方式？
SelectStartMenuFolderLabel3=安装程序将在以下开始菜单文件夹中创建程序快捷方式。
SelectStartMenuFolderBrowseLabel=单击“下一步”继续。如需选择其他文件夹，请单击“浏览”。
MustEnterGroupName=请输入文件夹名称。
GroupNameTooLong=文件夹名称或路径过长。
InvalidGroupName=文件夹名称无效。
BadGroupName=文件夹名称不能包含以下字符：%n%n%1
NoProgramGroupCheck2=不创建开始菜单文件夹(&D)

; Ready to install
WizardReady=准备安装
ReadyLabel1=现在可以开始在您的计算机上安装 [name]。
ReadyLabel2a=单击“安装”开始安装；如需检查或更改设置，请单击“上一步”。
ReadyLabel2b=单击“安装”开始安装。
ReadyMemoUserInfo=用户信息：
ReadyMemoDir=安装位置：
ReadyMemoType=安装类型：
ReadyMemoComponents=所选组件：
ReadyMemoGroup=开始菜单文件夹：
ReadyMemoTasks=附加任务：

; Downloads
DownloadingLabel2=正在下载文件...
ButtonStopDownload=停止下载(&S)
StopDownload=是否停止下载？
ErrorDownloadAborted=下载已中止
ErrorDownloadFailed=下载失败：%1 %2
ErrorDownloadSizeFailed=无法获取文件大小：%1 %2
ErrorProgress=进度无效：%1 / %2
ErrorFileSize=文件大小不符：预期 %1，实际 %2

; Archive extraction
ExtractingLabel=正在解压文件...
ButtonStopExtraction=停止解压(&S)
StopExtraction=是否停止解压？
ErrorExtractionAborted=解压已中止
ErrorExtractionFailed=解压失败：%1
ArchiveIncorrectPassword=密码不正确
ArchiveIsCorrupted=压缩文件已损坏
ArchiveUnsupportedFormat=不支持此压缩格式

; Preparing to install
WizardPreparing=正在准备安装
PreparingDesc=正在准备将 [name] 安装到您的计算机上。
PreviousInstallNotCompleted=之前的程序安装或卸载尚未完成，需要重新启动计算机才能完成。%n%n请重启计算机后再次运行安装程序，以完成 [name] 的安装。
CannotContinue=安装无法继续。请单击“取消”退出安装程序。
ApplicationsFound=以下应用程序正在使用需要更新的文件。建议允许安装程序自动关闭这些应用程序。
ApplicationsFound2=以下应用程序正在使用需要更新的文件。建议允许安装程序自动关闭这些应用程序。安装完成后，安装程序将尝试重新启动这些应用程序。
CloseApplications=自动关闭应用程序(&A)
DontCloseApplications=不关闭应用程序(&D)
ErrorCloseApplications=安装程序无法自动关闭所有应用程序。建议在继续之前关闭所有正在使用待更新文件的应用程序。
PrepareToInstallNeedsRestart=安装程序需要重新启动计算机。请在重启后再次运行安装程序，以完成 [name] 的安装。%n%n是否立即重启？

; Installing
WizardInstalling=正在安装
InstallingLabel=正在将 [name] 安装到您的计算机上，请稍候。

; Setup completed
FinishedHeadingLabel=[name] 安装完成
FinishedLabelNoIcons=[name] 已安装到您的计算机上。
FinishedLabel=[name] 已安装到您的计算机上。您可以通过已创建的快捷方式启动此程序。
ClickFinish=单击“完成”退出安装程序。
FinishedRestartLabel=需要重新启动计算机才能完成 [name] 的安装。是否立即重启？
FinishedRestartMessage=需要重新启动计算机才能完成 [name] 的安装。%n%n是否立即重启？
ShowReadmeCheck=查看自述文件
YesRadio=是，立即重启计算机(&Y)
NoRadio=否，稍后手动重启(&N)
RunEntryExec=运行 %1
RunEntryShellExec=查看 %1

; Next disk
ChangeDiskTitle=插入磁盘
SelectDiskLabel2=请插入磁盘 %1，然后单击“确定”。%n%n如果文件位于其他位置，请输入正确路径，或单击“浏览”选择。
PathLabel=路径(&P)：
FileNotInDir2=在“%2”中找不到文件“%1”。请插入正确的磁盘或选择其他文件夹。
SelectDirectoryLabel=请指定下一张磁盘所在的位置。

; Installation phase
SetupAborted=安装未完成。%n%n请修复问题后再次运行安装程序。
AbortRetryIgnoreSelectAction=请选择操作
AbortRetryIgnoreRetry=重试(&T)
AbortRetryIgnoreIgnore=忽略错误并继续(&I)
AbortRetryIgnoreCancel=取消安装
RetryCancelSelectAction=请选择操作
RetryCancelRetry=重试(&T)
RetryCancelCancel=取消

; Installation status
StatusClosingApplications=正在关闭应用程序...
StatusCreateDirs=正在创建文件夹...
StatusExtractFiles=正在解压文件...
StatusDownloadFiles=正在下载文件...
StatusCreateIcons=正在创建快捷方式...
StatusCreateIniEntries=正在写入 INI 配置...
StatusCreateRegistryEntries=正在写入注册表...
StatusRegisterFiles=正在注册文件...
StatusSavingUninstall=正在保存卸载信息...
StatusRunProgram=正在完成安装...
StatusRestartingApplications=正在重新启动应用程序...
StatusRollback=正在撤销更改...

; General errors
ErrorInternal2=内部错误：%1
ErrorFunctionFailedNoCode=%1 失败
ErrorFunctionFailed=%1 失败；错误代码 %2
ErrorFunctionFailedWithMessage=%1 失败；错误代码 %2。%n%3
ErrorExecutingProgram=无法执行文件：%n%1

; Registry and INI errors
ErrorRegOpenKey=打开注册表项时出错：%n%1\%2
ErrorRegCreateKey=创建注册表项时出错：%n%1\%2
ErrorRegWriteKey=写入注册表项时出错：%n%1\%2
ErrorIniEntry=在文件“%1”中创建 INI 条目时出错。

; File errors and verification
FileAbortRetryIgnoreSkipNotRecommended=跳过此文件(&S)（不推荐）
FileAbortRetryIgnoreIgnoreNotRecommended=忽略错误并继续(&I)（不推荐）
SourceIsCorrupted=源文件已损坏。
SourceDoesntExist=源文件“%1”不存在。
SourceVerificationFailed=源文件验证失败：%1
VerificationSignatureDoesntExist=签名文件“%1”不存在
VerificationSignatureInvalid=签名文件“%1”无效
VerificationKeyNotFound=签名文件“%1”使用了未知密钥
VerificationFileNameIncorrect=文件名不符
VerificationFileTagIncorrect=文件标签不符
VerificationFileSizeIncorrect=文件大小不符
VerificationFileHashIncorrect=文件哈希值不符
ExistingFileReadOnly2=现有文件为只读，无法替换。
ExistingFileReadOnlyRetry=移除只读属性并重试(&R)
ExistingFileReadOnlyKeepExisting=保留现有文件(&K)
ErrorReadingExistingDest=读取现有文件时出错：
FileExistsSelectAction=请选择操作
FileExists2=文件已存在。
FileExistsOverwriteExisting=覆盖现有文件(&O)
FileExistsKeepExisting=保留现有文件(&K)
FileExistsOverwriteOrKeepAll=对之后的所有冲突执行相同操作(&D)
ExistingFileNewerSelectAction=请选择操作
ExistingFileNewer2=现有文件比即将安装的文件更新。
ExistingFileNewerOverwriteExisting=覆盖现有文件(&O)
ExistingFileNewerKeepExisting=保留现有文件(&K)（推荐）
ExistingFileNewerOverwriteOrKeepAll=对之后的所有冲突执行相同操作(&D)
ErrorChangingAttr=更改现有文件属性时出错：
ErrorCreatingTemp=在目标文件夹中创建文件时出错：
ErrorReadingSource=读取源文件时出错：
ErrorCopying=复制文件时出错：
ErrorDownloading=下载文件时出错：
ErrorExtracting=解压文件时出错：
ErrorReplacingExistingFile=替换现有文件时出错：
ErrorRestartReplace=重启后替换文件的操作失败：
ErrorRenamingTemp=重命名目标文件夹中的文件时出错：
ErrorRegisterServer=无法注册 DLL/OCX：%1
ErrorRegSvr32Failed=RegSvr32 执行失败，退出代码为 %1
ErrorRegisterTypeLib=无法注册类型库：%1

; Uninstall display name
UninstallDisplayNameMark=%1（%2）
UninstallDisplayNameMarks=%1（%2，%3）
UninstallDisplayNameMark32Bit=32 位
UninstallDisplayNameMark64Bit=64 位
UninstallDisplayNameMarkAllUsers=所有用户
UninstallDisplayNameMarkCurrentUser=当前用户

; Post-installation errors
ErrorOpeningReadme=无法打开自述文件。
ErrorRestartingComputer=无法重新启动计算机，请手动重启。

; Uninstaller
UninstallNotFound=文件“%1”不存在，无法卸载。
UninstallOpenError=无法打开文件“%1”，无法卸载。
UninstallUnsupportedVer=当前卸载程序无法识别卸载日志文件“%1”的格式，无法卸载。
UninstallUnknownEntry=卸载日志中包含未知条目（%1）。
ConfirmUninstall=是否完全删除 %1 及其所有组件？
UninstallOnlyOnWin64=此程序只能在 64 位 Windows 上卸载。
OnlyAdminCanUninstall=卸载此程序需要以管理员身份登录。
UninstallStatusLabel=正在从您的计算机中删除 %1，请稍候。
UninstalledAll=%1 已从您的计算机中成功删除。
UninstalledMost=%1 已卸载。%n%n部分内容未能删除，请手动删除。
UninstalledAndNeedsRestart=需要重新启动计算机才能完成 %1 的卸载。%n%n是否立即重启？
UninstallDataCorrupted=文件“%1”已损坏，无法卸载。

; Shared files and uninstall progress
ConfirmDeleteSharedFileTitle=删除共享文件
ConfirmDeleteSharedFile2=系统报告以下共享文件已不再被任何程序使用。是否删除此文件？%n%n如果其他程序仍在使用此文件，删除后可能导致该程序无法正常运行。如果不确定，请选择“否”。保留此文件不会造成问题。
SharedFileNameLabel=文件名：
SharedFileLocationLabel=位置：
WizardUninstalling=正在卸载
StatusUninstalling=正在卸载 %1...

; Shutdown block reasons
ShutdownBlockReasonInstallingApp=正在安装 %1。
ShutdownBlockReasonUninstallingApp=正在卸载 %1。

[CustomMessages]
NameAndVersion=%1 版本 %2
AdditionalIcons=附加快捷方式：
CreateDesktopIcon=创建桌面快捷方式(&D)
CreateQuickLaunchIcon=创建快速启动栏快捷方式(&Q)
ProgramOnTheWeb=%1 网站
UninstallProgram=卸载 %1
LaunchProgram=启动 %1
AssocFileExtension=将 %2 文件与 %1 关联(&A)
AssocingFileExtension=正在将 %2 文件与 %1 关联...
AutoStartProgramGroupDescription=启动：
AutoStartProgram=自动启动 %1
AddonHostProgramNotFound=在所选文件夹中找不到 %1。%n%n是否仍要继续？
