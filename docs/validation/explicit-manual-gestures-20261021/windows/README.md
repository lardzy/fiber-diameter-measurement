# Windows 明确手势试测包验证（2026-10-21）

本轮把手工“两点／拖线”明确分开，保留混合模式中的候选短点击；“贴边／自由”算法选择不变。版本保持 **0.4.10**。实际改动、macOS 375 项回归和主题检查见上级报告；本页仅记录 Windows 虚拟机的定向门禁及交付。

## 构建身份与交付

- 基线：`dc25a4031009a7dd9cdf1646cafa9f089fb746d9`，包含本轮未提交修改及上一轮帧间隙修复。
- 构建 ID：`2527fbdb485c4d419ffb7c8ccf42a5c8`。
- 安装包：`dist/internal/explicit-manual-gestures-20261021/FDM-0.4.10-explicit-gestures-2527fbdb485c.exe`。
- SHA256：`fd631229a9e74b61cf654d4e96271d50f482e02d425c4c48f48fd5397a17c833`；398,846,891 bytes。
- 配置：internal `--exclude-area-models`，保留内部模板、Fluent 图标和 Microview 运行资源；未使用 public-release。
- 源码快照清单 SHA256：`b2d2afd83e76999d61ca5600ac0f90fc2362757a08850ed98a27fc68fb2f7890`。快照不含 `.git`，因此包中 `source_commit=unknown`；基线、五项产品修改及全部 296 项 src/native 文件已用独立清单固定。
- 最终源 ZIP SHA256：`5012326d71ec8d7863451212bdd6bd63b6d7731b861d4c7a7d0cfd1773b387c0`，同步并逐文件校验 2,831 项。

构建完成后再次核对全部 296 项冻结文件：当前工作区与 Windows 源码清单均一致，见 `source-freeze-verification.json`。本轮没有修改版本、产品配置或旧安装包，未提交、未推送。

## 已通过的门禁

| 检查 | 结果与证据 |
|---|---|
| Windows 定向 pytest | **146 passed in 27.98s**；完整运行新增 39 项明确手势测试，并包含混合画布、真实帧间隙、辅助设置 UI、第七轮画布和第六轮首点回归。选取清单见 `test-selection.json` |
| 真实导航与输入 | source-quick 通过，8.468889 秒；由主导航进入含量试验，真实全图分析、候选点击、账本、证据及换场保留流程；模拟相机，不注入伪造候选或记录 |
| 两点手势 | 实际点击手势菜单；按下第一点后持键移动和释放均不提交，100 次无按键移动不提交，明确第二次按下才产生记录 |
| 拖线手势 | 短点击不留手工第一点；持键拖动后释放产生记录；切换手势清除未完成两点草稿 |
| 混合候选 | 两种手工手势下短点击候选均正常计入；共 7 条已接受记录，无残留手工草稿；点选不重建批次、余项保持显示 |
| 候选生命周期 | 独立逐帧对齐 `poll=0、compare=0`；真实新视场才产生第二次分析。此快速场景不替代历史长时压力测试 |
| 包提取、自检与资源 | 4,818 项资源、436 项产品源码/运行资源校验通过；图标、Microview x64 DLL、内部报告模板在包中；冻结 EXE 和工作进程自检通过 |
| 实际安装 | 独立候选目录安装成功；原安装四项关键文件哈希保持不变 |
| 普通用户自检 | `elevated=false`；已安装 EXE 与提取包 EXE 哈希一致，构建 ID 一致 |
| 普通用户启动与可见窗口 | Explorer → 隔离 cmd → 安装 EXE，标题正确，前台 HWND 与主窗口一致；DPI-aware PrintWindow 图像显示完整 FDM 主窗和 v0.4.10，无异常对话框；正常退出 code 0，无强制终止 |
| 恢复注册 | 原 FDM 注册项恢复，四项旧文件哈希再次通过；未运行卸载程序，未回滚整台虚拟机 |

手工实际运行验收使用真实的**自由测量**算法，以隔离手势和贴边识别拒绝；单元回归覆盖手工算法接口。真实相机中的贴边识别及现场鼠标事件仍需用户按 `OFFLINE_TEST.md` 试测。本轮没有重跑全套历史视觉或业务矩阵。

`package/installer-identity.json` 在安装前生成，因此保留当时的 `actual_installation_tested=false`。安装与启动的后续证据分别是 `actual-installation.json`、`installed-selfcheck/result.json`、`explorer-start/result.json`；最终状态见 `delivery-manifest.json`。

包 self-check 会单独检查对齐模块的离线能力，不能据此推断采集重新启用了逐帧对齐；实际导航结果明确记录 `independent_alignment_calls` 均为 0。

## 可见性证据与安装位置

- 实际安装位置：`C:\tmp\fdm-cu-live-explicit-gestures-20261021\dist\windows\installed-candidate`。
- 恢复注册后保留位置：`C:\tmp\fdm-cu-live-explicit-gestures-20261021\dist\windows\installed-candidate-validated`。
- 本轮普通用户启动时间：2026-10-21 15:01:31～15:01:36（北京时间），PID 5632，构建 ID `2527fbdb485c4d419ffb7c8ccf42a5c8`。
- 窗口截图：`explorer-start/installed-window-printwindow.png`，1626×1071；另保留同次前台屏幕截图作对照。
- 手势流程截图：`source-quick/explicit-gestures.png`，实际客户区 1600×900，可见“自由／两点”、新视场、保存和 Excel 常驻入口。

已实际查看两张关键图片。启动截图与上一包的干净空态像素相同，因此 SHA256 相同；本次是对新构建进程进行的新捕获，有独立进程 ID、时间、EXE 路径、构建 ID 及前台 HWND 记录，未复制旧图。媒体由 `.gitignore` 排除，仍保留在本地验证目录及证据 ZIP 中。

## 耗时与构建辅助说明

原生扩展 19.33 秒；目标 pytest 本体 27.98 秒（进程门禁 28.81 秒）；实际导航 8.47 秒（进程门禁 9.16 秒）；PyInstaller＋安装包封装 593.57 秒，其中 PyInstaller 394.45 秒；包提取与自检 48.28 秒；实际安装 33.79 秒；普通用户自检＋启动截图及关闭 22.47 秒；注册恢复 0.76 秒。

`phase-timings.json` 以日志创建和状态写入计时，`installation-phase-timings.json` 与 `delivery-phase-timings.json` 使用外部单调时钟。Explorer 日志没有有效写入间隔，因此其中的 0 秒不代表实际运行时长；其独立启动约 4.86 秒，见 `explorer-start/result.json`。

macOS 临时验收首次受到共享内存沙箱限制；授权后继续。临时菜单模拟使用 Home/Down/Return 在 offscreen 菜单中无法按期完成，改为点击实际菜单项几何并加两秒超时保护；只更改验证脚本，未修改产品。最终脚本先在 macOS 通过，再重新同步 Windows 并通过。验证脚本与改造来源哈希保留在 `validation-tools/` 和 `adapter-plan.json`。

原 PowerShell 重定向日志为 UTF-16，原件保留；额外生成 `.utf8.log` 便于阅读。原始证据 ZIP 保持不变。

## 回退与边界

两个旧包均保留，并重新核对 SHA256：

- `dist/internal/mixed-presented-frame-fix-20261021/FDM-0.4.10-mixed-frame-fix-5bdd8be20cdb.exe`：`6d36be1b4bba5b0ecfad3febb4c58ddacfc35f30b692060587ec9d2c4d49d5c1`。
- `dist/internal/acquisition-seventh-round-20261021/FDM-0.4.10-round7-0b4a03782cd9.exe`：`3298883840527ba419a2c32dfb3db1fab11923078e6f5c5d81862799c52dce59`。

全部 Windows 结果来自 **Parallels Windows 11 虚拟机／模拟相机／Qt 鼠标输入**。本机未连接实体 Microview；现场概率故障是否彻底消失仍待用户实机确认。安装包已在最小门禁通过后先行提供，完整定向证据随后补齐。
