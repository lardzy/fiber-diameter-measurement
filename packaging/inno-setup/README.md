# 中文安装器

`fdm_installer.iss` 使用仓库内的 [`languages/ChineseSimplified.isl`](languages/ChineseSimplified.isl)。安装、升级和卸载均使用简体中文，不显示语言选择窗口，也不沿用旧安装记录中的英文语言设置。通过 Python 脚本、BAT 包装脚本或 Inno Setup IDE 编译都会加载同一份语言文件，不需要修改编译器目录。

界面中的软件名称是“特纤通用测量工具”。AppId、默认安装目录、开始菜单文件夹和产物文件名沿用现有值，以便继续识别旧版本并使用原有路径。版本号仍由 `src/fdm/version.py` 统一管理。

## 构建

建议使用 Inno Setup 6.5 或更新的 Unicode 版本。在 Windows 仓库根目录运行：

```powershell
python scripts/build_windows_installer.py
# 或使用已有的入口（参数会转交给 Python 脚本）：
.\scripts\build_windows_installer.bat
```

上面两个命令择一执行即可。默认重建 full onedir，再生成安装器。公开版本继续使用 `--public-release`；复用已验证的 onedir 使用 `--reuse-onedir`。

编译器未被自动发现时，可显式指定路径：

```powershell
python scripts/build_windows_installer.py --compiler "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
```

仅刷新版本文件时仍可使用 `--sync-only`，此模式不要求 Inno Setup、onedir 或语言文件存在。正常构建会检查中文语言文件是否存在，避免先执行耗时的 onedir 构建后才发现文件遗漏；直接编译 `.iss` 时也有对应的缺失检查。

## 语言文件维护

- 文件使用 **UTF-8 with BOM**，简体中文语言 ID 为 `$0804`，代码页为 `936`，界面字体为 Microsoft YaHei UI。
- 包含 281 条 `[Messages]` 和 12 条 `[CustomMessages]`，覆盖向导、按钮、安装进度、错误、下载、解压、卸载和重启提示。
- 结构参考所提供的 `Japanese.isl`，并对照官方 [6.5.0 Default.isl](https://github.com/jrsoftware/issrc/blob/is-6_5_0/Files/Default.isl) 补齐 `HelpTextNote`。已核对 [6.7.3](https://github.com/jrsoftware/issrc/blob/is-6_7_3/Files/Default.isl) 和 [7.1.0](https://github.com/jrsoftware/issrc/blob/is-7_1_0/Files/Default.isl) 的消息键与占位符；三份模板一致。这是语言数据核对，不代表已经在所有编译器版本上构建或运行。
- 编辑时保留 `%1`、`%2` 等替换参数、`%n` 换行以及 `[name]`、`[name/ver]`、`[gb]`、`[mb]`。`(&N)` 等文本中的 `&` 用于快捷键。
- `AboutSetupNote`、`TranslatorNote`、`BeveledLabel`、`HelpTextNote` 与官方模板一样留空；其余消息均提供译文。
- 升级 Inno Setup 后，核对其 `Default.isl` 的新增消息和编译日志中的缺失消息警告。语言文件的接入和消息参数规则见官方 [Languages 文档](https://jrsoftware.org/ishelp/topic_languagessection.htm) 与 [Messages 文档](https://jrsoftware.org/ishelp/topic_messagessection.htm)。

## Windows 验证

在目标 Windows 环境重新生成安装包后，检查：

1. 全新安装：许可协议页标题、安装目录、开始菜单、附加任务、安装进度、完成页和启动选项显示中文，中文字符正常显示。
2. 覆盖旧英文安装包：识别原安装位置，界面改用中文；桌面快捷方式和文件关联仍正常。
3. 取消安装、程序占用文件及卸载确认等提示显示中文；卸载器中的名称与安装界面一致。
4. 在不同系统语言和常用显示缩放下检查文字换行、按钮、复选框是否被截断。

许可协议正文仍为仓库原有 `LICENSE`，并未随界面本地化翻译。Windows UAC 等系统界面由 Windows 自身决定语言；部分系统消息框按钮也使用系统语言，参见官方 [FAQ](https://jrsoftware.org/isfaq.php)。这些内容不由 `.isl` 全部控制。
