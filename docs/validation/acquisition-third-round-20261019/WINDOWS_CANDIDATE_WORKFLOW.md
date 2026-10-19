# 第三轮 Windows 候选包工作流

产品版本保持 **0.4.10**。本轮是内部候选，采用 `--exclude-area-models`，保留 CU 模板；通过构建 ID、源码清单、日期和 SHA-256 区分同版本候选，不能覆盖第二轮或其他既有安装包。

## 准备状态

2026-10-19 初始只读检查确认 Parallels `Windows 11` 处于关机状态，摄像头自动共享开启。随后已在授权范围启动 VM 并只读复查：原 0.4.7 安装登记保持，CurrentUser 为 RemoteSigned、其他策略作用域 Undefined；Excel 已注册、没有运行中的 FDM 或 Excel。Python、编译工具、Inno 和提取工具均存在。见 [环境基线](windows/environment-before.json)及[构建工具](windows/build-tools.json)。已完成两次中间源码同步和原生扩展编译。中间 Windows 测试在旧 fixture 清理路径触发 Qt access violation，日志单独保留；随后暂停安装器阶段，等待 fixture 与低高度布局修复的最终同步。随后完成最终同步，Windows 291 项测试通过；内部候选构建和 4818 个载荷 / 425 个生产源文件核验通过。实际安装已在独立目录成功（退出 0），GUI / Excel 及检查点恢复结果另见本轮 Windows 证据。

准备检查未修改已有安装、来宾策略或相机共享设置。正式安装验证采用可回滚 VM checkpoint：临时关闭宿主相机共享，在新目录安装。证据回收后已恢复 checkpoint，并复查原 0.4.7 安装登记、执行策略、相机共享均恢复，没有遗留 FDM / Excel 进程；随后正常关机，VM 回到初始 `stopped`，仅删除本轮临时 checkpoint。见[恢复记录](windows/checkpoint.json)、[恢复后的环境](windows/environment-after-restore.json)。

候选安装包：`FDM-0.4.10-acquisition-third-round-20261019.exe`；构建 ID：`e1cf7774e0c1453796f966d222e2cf88`；398,674,244 字节；SHA-256：`790b16093df9948ac8a566e4b283cee56192b3f65d5d290a1591cb00f2889d8f`。

## 固定隔离范围

| 用途 | 位置 |
| --- | --- |
| 新 Windows 源码与构建目录 | `C:\tmp\fdm-cu-live-third-round-20261019` |
| 借用的现有 Python 环境 | `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe` |
| 原生编译工具 | `C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat` |
| 已固定 SHA-256 的提取工具 | `C:\tmp\fdm-cu-live-final-r4-20261015\build\validation-tools\innounp.exe` |
| 本轮构建日志、测试与提取证据 | 新目录内 `build\third-round\` |
| 本机交付目录 | `dist/internal/acquisition-third-round-20261019/` |
| 预定交付安装包 | `FDM-0.4.10-acquisition-third-round-20261019.exe` |

## 执行顺序

1. 代码稳定并确认启动 VM 后，用 `probe_acquisition_third_round_environment.ps1` 只读记录安装登记、策略、已有 FDM / Excel 进程及工具路径。它不会打开 Excel、相机或既有应用。记录启动前后的 VM 状态。
2. 在本机运行 `.venv/bin/python scripts/validation/sync_acquisition_third_round_windows.py --build`。只把源码及内部运行资源同步到上述新目录，并用清单逐项核对；现有 Python 仅提供解释器和依赖。`--build` 在隔离目录编译原生扩展，不是生成安装包。
3. 在隔离目录运行 `scripts/validation/build_acquisition_third_round.ps1`。默认包含第三轮工作流、记录窗、紧凑布局、主题及既有显示 / 报告相关测试；其他专项可用 `-AdditionalTests` 显式补充。可使用 `-TestsOnly` 先只执行测试；脚本保存确切测试清单，全部通过后使用 `scripts/build_windows_installer.py --exclude-area-models` 构建。已有同名第三轮安装包时拒绝覆盖，须先将其保留到独立历史位置再重建。
4. 脚本继续运行 `verify_third_round_installer.py`：核对提取工具身份、实际安装器全部载荷、CU 模板原始哈希、Microview DLL、排除组件、冻结程序 `--self-check --json`、构建 ID，以及生产源码在构建期间是否变化。该检查仅提取文件，不运行安装器。
5. `receive_acquisition_third_round_artifact.py` 只允许回传本轮隔离目录中的文件，经私有 Parallels 链路传输并比对两端 SHA-256；本机目标已存在时拒绝覆盖。回传新安装包、包身份、发布清单、源码清单和日志。
6. 在 checkpoint 内运行第三轮 `run_acquisition_third_round_frozen.py`。它调用新的 `probe_acquisition_third_round_frozen.ps1`，支持空类别入口和返回、四主题八类别、全部操作者记录、底部 / 左侧 / 浮窗、固定 Excel 入口、真实导航及 1067×600 可用尺寸。实际 Windows 安装版执行了下述四个浅色场景；其他主题由本机视觉矩阵覆盖。旧探针只复用 PID 限定的 UIA/窗口帮助函数，不执行其旧导航流程。
7. 实际安装使用专用 `dist/windows/installed-candidate` 目录与 `/NOICONS /TASKS=""`；固定 AppId 会暂时更新卸载登记，必须在 checkpoint 中执行。回收安装日志、GUI 证据及普通用户 Explorer 启动结果后恢复 checkpoint，再复查原安装、策略和相机共享。

## 实际安装与 GUI 结果

[安装汇总](windows/installed-validation.json)记录候选身份、四个成功场景、Excel 校验和恢复结果。[安装日志](windows/selected/actual-installation.log)及[安装结果](windows/selected/actual-installation.json)确认退出码 0；[普通用户 Explorer 启动](windows/selected/explorer-start.json)确认非提权用户、父进程 `explorer.exe`、RemoteSigned 环境下正常打开和退出安装版。

| 安装版场景 | 结果 | 证据 |
| --- | --- | --- |
| 浅色 1600×900 空入口 | 未预填类别；设置类别和返回入口可见；15 次真实子页选择，正常退出且无空访问保存提示 | [摘要](windows/frozen-gui/20261019-070904Z-48bd25d6/summary.json)、[截图](windows/installed-empty-content.png) |
| 浅色 1600×900 八类别记录 | 显式构造八类别；三位操作者、两个方向共 150 条记录；默认全部操作者 / 全部方向；底部、左侧、浮窗及返回恢复通过 | [摘要](windows/frozen-gui/20261019-071226Z-0cb30229/summary.json)、[截图](windows/selected/installed-content-1600.png) |
| 浅色 1600×900 固定 Excel 入口 | 在“操作”页点击固定 Excel 按钮，完成真实 Excel 16.0 COM 导出；未进入报告页 | [摘要](windows/frozen-gui/20261019-072421Z-34040aac/summary.json)、[成功截图](windows/selected/installed-excel-success.png)、[数据校验](windows/selected/excel-verification.json) |
| 浅色 1067×600 记录窗 | 底部、左侧、浮窗及展开样品信息后客户区尺寸保持，固定底栏与 Excel 入口可达；窄小空间使用局部滚动 | [摘要](windows/frozen-gui/20261019-072631Z-08068479/summary.json)、[底部停靠截图](windows/selected/installed-records-bottom-1067.png) |

GUI 记录的系统 DPI 为 **192**，探针显式使用 **`QT_SCALE_FACTOR=0.5`**。上述尺寸是实际观测的物理客户区尺寸，只代表这个缩放组合下的 Windows 原生控件布局，不能表述为系统 100% DPI 或跨显示器验收。1067×600 截图保留了“位置”选择菜单展开状态；固定底栏仍可见。

Excel 场景显式创建两类、三位操作者和经纬两方向，与空入口不预填类别的检查分开。实际报告保留编号字符串 `0000123`；校验通过 52 个输入单元格、144 个直径、12 项计数结果、486 个原有公式带单元格，以及 12 张工作表的名称和打印区域、VBA 存在状态。源模板哈希未变。[生成的测试报告](windows/selected/fixed-entry-report.xls) SHA-256：`3a8bed5faac859116b1793571c8a1f483f8b31b9292f9d0924715574d43bdcf3`。

探针曾出现两项自身问题，原始日志保留：空入口结果被重复输出了两个完全相同的 JSON，修正解析器为仅折叠一致副本后恢复原成功结果；首次 Excel 探针误按 UIA Edit / Button 寻找原生保存窗，实际控件以 Pane 暴露，后按已观测 Win32 控件类与 ID 并校验 PID / 父子关系完成保存。后一失败仅清理该次自有 FDM 进程，未开始 Excel 导出；修正后真实导出成功。它们不记为产品失败。

[完整运行证据 ZIP](windows/runtime-evidence.zip)共 4,279,746 字节，SHA-256 为 `75d8270f8757bdb6d1e57f50d494f7c095245eb346225ce836ec4962ac88b096`。关键截图和 JSON 平铺在 `windows/selected/`；深层解包原始树保留于忽略目录 `.tmp/acquisition-third-round-20261019/windows-runtime-evidence`，避免 Windows Git 路径过长。五次探针摘要与标准输出仍保留在 `windows/frozen-gui/`，其中一次为上述保存窗探针失败过程。

## 验收与交付边界

- 提取校验通过不等于已实际安装；源码视觉探针不等于冻结 Windows GUI；模拟系统主题不等于真实跨显示器切换。
- 真正连接 Microview 的验证仍由离线设备完成，不能把共享宿主摄像头或模拟帧当作 Microview 验收。
- 候选目录需保存安装包 SHA-256、构建 ID、源清单哈希、实际执行的测试及剩余限制。既有第二轮包继续保留，不改其版本记录与验证结论。
- 执行策略使用每个受控 PowerShell 进程的启动参数，不修改 CurrentUser、LocalMachine 或组织策略。冻结自检不发起相机采集。
