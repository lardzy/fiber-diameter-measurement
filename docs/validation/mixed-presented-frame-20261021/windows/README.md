# Windows 替换试测包验证（2026-10-21）

本轮仅修复混合模式在“新帧到达、旧帧尚在屏幕上”期间静默丢失落点的问题。版本保持 **0.4.10**，两点与拖线的既有规则不变。

## 构建身份

- 基线提交：`dc25a4031009a7dd9cdf1646cafa9f089fb746d9`，包含本轮未提交修复。
- `live_measurement_canvas.py`：41,480 bytes；SHA256 `eb74c4b97dbbc829d4ef2b01e26ad2af6dc483dc98c4fc7d6ce19c6882ab0f07`。
- 构建 ID：`5bdd8be20cdb4647aab36c391b08a781`。
- 安装包：`dist/internal/mixed-presented-frame-fix-20261021/FDM-0.4.10-mixed-frame-fix-5bdd8be20cdb.exe`。
- 安装包大小：398,846,679 bytes。
- SHA256：`6d36be1b4bba5b0ecfad3febb4c58ddacfc35f30b692060587ec9d2c4d49d5c1`。
- 构建配置：internal `--exclude-area-models`，保留 Microview 与内部报告模板；未使用 public-release。
- 源码快照清单 SHA256：`f416ca38b56b732c3ca089c3988ad9753be4ed8498af24a768d5f9d34c5841c6`。`source_commit=unknown` 是独立源码快照不带 `.git` 的结果，基线及文件哈希在清单中另行固定。

## 本轮结果

| 检查 | 结果 | 边界 |
|---|---|---|
| Windows 针对性 pytest | **77 passed，14.29 秒** | 包含 6 项实际 Workspace/live 模拟帧间隙回归，以及混合输入、round7 画布和 round6 首点回归 |
| 实际导航 source-quick | 通过，8.52 秒 | 模拟相机，真实分析、候选点击、账本与证据流程；3 次接受、余项不整组隐藏、点选标记保留通过 |
| 安装包提取与哈希 | 通过 | 4,818 个资源文件、436 个产品源码/运行资源匹配；图标、Microview x64 DLL、报告模板在包中 |
| 提取后 self-check | 通过 | 冻结程序与工作进程自检；独立对齐模块仅在离线自检使用，不表示采集启用了对齐任务 |
| 实际安装 | 通过 | 独立目录安装；原应用四项文件哈希未变 |
| 普通用户 self-check | 通过 | `elevated=false`，安装 EXE 与包提取 EXE 哈希一致 |
| 普通用户 Explorer 启动 | 通过 | `Explorer → 隔离 cmd → 安装 EXE`；窗口标题正确、正常退出 code 0，无强制终止 |
| 安装版可见窗口复核 | 通过（第二次捕获） | 重新启动同一 EXE，确认前台 HWND、DPI-aware 坐标及 PrintWindow；实际图像为完整 FDM 主窗口，未见异常对话框 |
| 原安装注册恢复 | 通过 | 仅恢复身份校验通过的 FDM 注册项，旧安装文件不变；未运行卸载程序 |

所有鼠标输入回归属于 **Windows 虚拟机／模拟输入**。实体 Microview 相机、真实鼠标操作及现场频繁触发是否消失，仍待用户实机确认。本轮没有重跑旧视觉矩阵或全量历史业务验收。

`package/installer-identity.json` 在实际安装前生成，因此其 `actual_installation_tested=false` 保留历史原值；后续实际安装、普通用户启动的证据见 `actual-installation.json`、`installed-selfcheck/result.json`、`explorer-start/result.json`，最终交付状态见安装包目录的 `delivery-manifest.json`。

## 安装与回退位置

- 本轮实际安装验证路径：`C:\tmp\fdm-cu-live-mixed-frame-fix-20261021\dist\windows\installed-candidate`。
- 恢复注册后保留路径：`C:\tmp\fdm-cu-live-mixed-frame-fix-20261021\dist\windows\installed-candidate-validated`。
- 第七轮回退包保持原位：`dist/internal/acquisition-seventh-round-20261021/FDM-0.4.10-round7-0b4a03782cd9.exe`。
- 第七轮回退包 SHA256：`3298883840527ba419a2c32dfb3db1fab11923078e6f5c5d81862799c52dce59`。
- 未清理旧包、未修改产品版本、未提交或推送。

## 阶段耗时

原生扩展 20.19 秒；目标 pytest 14.29 秒；source-quick 8.52 秒；PyInstaller＋封装 615.41 秒（其中 PyInstaller 日志 415.03 秒）；包提取和自检 48.13 秒；实际安装 34.70 秒；普通用户 self-check＋Explorer 启动/截图/关闭合计 22.44 秒；注册恢复 0.77 秒。

`phase-timings.json` 保留原始文件时间推导值，`installation-phase-timings.json` 是外部单调时钟计时。Explorer 的独立日志未产生有效写入间隔，不使用其中的 0 秒判断实际运行时间；精确起止时间见 `explorer-start/result.json`（约 4.98 秒）。

## 构建辅助问题

首次本机私有网络监听受到沙箱限制，执行授权后继续。一次 Parallels 短 `-Command` 调用返回 255 参数错误，临时构建适配器改用编码传输后成功。这些只影响构建工具，不涉及产品代码。

安装及启动均通过后，证据汇总脚本曾用 Windows 默认 GBK 读取 UTF-8 源码清单，触发 `UnicodeDecodeError`。只修正临时汇总脚本为显式 UTF-8 并重新汇总，安装包未重建。保留失败阶段记录及当时终端错误摘录。

## 启动截图复核

首次 `explorer-start/installed-window.png` 保留为**无效截图**：根线程视觉检查发现其内容是其他窗口，不能作为应用可见性证据。该捕获未校验前台窗口，且在 200% DPI 下使用了虚拟化坐标（813×536）。原运行日志只证明标题、进程与正常退出，不证明该图片内容正确。

随后重新以普通用户启动相同 `installed-candidate-validated` 中的 EXE，使用独立配置，设置捕获进程 DPI aware、前置窗口并验证前台 HWND 与主窗口一致，同时使用 PrintWindow 按指定 HWND 捕获。`explorer-visual-recheck/installed-window-printwindow.png` 为 **1626×1071** 的完整 FDM 主窗口：标题“显微测量工作台”、工具栏、空白工作区及右下角 v0.4.10 均可见，未见异常对话框。另有同次前台屏幕捕获 `installed-window-screen.png` 作对照。窗口仍正常关闭、exit 0、无强制结束。未修改产品、未重建安装包。

本地图片与补充 ZIP 均保留。原 `windows-validation-evidence.zip` 不被覆盖；第二次视觉证据单独保存为 `windows-visual-recheck.zip`，两者哈希分别写入交付清单。媒体由此目录的 `.gitignore` 排除，不纳入源码变更。
