# Windows 截图快捷键优化与验证（2026-10-14）

本轮完成注册诊断、录入兼容和截图调度，并将源码同步到 Parallels Windows 11 项目进行原生窗口验证。继续使用 Windows `RegisterHotKey`；当前测量没有显示需要增加 C++ 输入模块。直接按 PrintScreen 录入在本虚拟机中受限，已提供并验证特殊键菜单路径。

## 实现范围

| 功能 | 当前行为 |
| --- | --- |
| 注册诊断 | 每个模式显示请求组合、实际生效组合、注册状态、原生错误码及最近接收时间；冲突后可重试。整组更新支持交换绑定，失败时尝试恢复。恢复失败按仍有效的原生绑定报告，不承诺操作原子性。 |
| 录入隔离 | 输入框获得焦点后通过租约暂停本工具的全局绑定，确认释放后录入；离开、隐藏、关闭和保存时恢复。租约在客户端消失后最多 15 秒失效。 |
| 键值兼容 | Windows 保存原生虚拟键和修饰位，保留旧配置兼容；处理 PrintScreen 只有释放事件、SysReq、重复按键、AltGr 和焦点结束。系统不送达 PrintScreen 时可从“特殊键”菜单选择。 |
| 截图调度 | 快捷键、托盘和 IPC 统一进入最多 8 项的队列；原生回调只接收请求。抑制长按、合并尚未派发的同模式快捷键、按 IPC 请求编号去重，并记录阶段耗时。 |
| 输出调度 | 图片编码与文件写入移到顺序工作线程，最多接受 4 项未完成输出；图片和设置使用快照，剪贴板留在 GUI 线程。慢保存不阻塞后续快捷键消息。 |
| 失败与退出 | 保存失败保留标注以便重试；退出等待已接受的保存。标注保存失败会取消退出。较旧的慢保存不会覆盖后来成功截图的“上次区域”。 |

操作说明见 [Windows 截图快捷键](../../screenshot-hotkeys.md)。截图抓取本身仍在 GUI 线程，不在本轮扩展为原生输入钩子或新截图后端。

## Windows 原生验证

环境：Parallels Windows 11 build 26200，Python 3.12.13 AMD64，PySide6 6.10.2，Qt `windows` 平台，物理截图尺寸 2582 × 1644。两端基础 HEAD 均为 `391d9262f99ed39d4b052433ffe6819b4f6415d5`，实施验证时本轮变更尚未提交，实际受测工作区源码以 [source-manifest.json](source-manifest.json) 为准。

探针通过真实 `RegisterHotKey → SendInput → WM_HOTKEY → WindowsScreenshotBackend → PNG 保存` 链路运行，核对图像解码、尺寸和摘要。临时图片验证后删除，只保留结果及摘要。未修改用户截图设置、登录自启或 Windows PrintScreen 首选项。

| 场景 | 结果 |
| --- | --- |
| 原生注册冲突、其它模式继续可用、释放占用后重试 | 通过，真实冲突返回 1409 |
| 正常连续 30 次全屏捕获 | 30 次消息、30 张有效 PNG |
| 录入暂停与恢复 | 通过，暂停时实际注册数为 0，恢复后触发截图 |
| 整组快捷键互换 | 通过 |
| 普通组合键录入 | 通过，失焦后仍保存 VK 120 和修饰位 7，注册并实际捕获 |
| 同次按下中的重复 keydown | 通过，仅触发一次；松开再按可以再次触发 |
| GUI 同步慢保存参考 | 完成，测到明显阻塞，用于受控对照 |
| 后台慢保存 | 通过，首张仍在保存时能接收下一次输入 |
| PrintScreen 全局注册 | 通过，真实接收 WM_HOTKEY 并保存 |
| PrintScreen 直接录入 / 特殊键菜单 | 直接录入受限；菜单选键、原生元数据、注册、WM_HOTKEY 和 PNG 完整通过 |

原始报告 [windows-final.json](windows-final.json) 的总体验收标记为 `passed_with_limitations`。最后一个场景的 `ok=true` 表示探针和菜单替代路径完成，**不表示直接按 PrintScreen 录入已通过**。该场景没有任何 PrintScreen 按下／释放事件到达录入框；前台窗口类别变为 `XamlWindow`，没有据此推断具体拦截程序。

## 性能与是否增加原生输入模块

只使用 `normal_capture_latency` 中的 30 次顺序捕获统计正常耗时，采样期间未并行运行测试。P50 为中位数，P95 为最近秩百分位。起点是应用收到 `WM_HOTKEY`，不包含实体键盘、宿主机或虚拟机输入转发延迟。

| 起止阶段 | P50 | P95 | 最大值 |
| --- | ---: | ---: | ---: |
| 接收 → 派发 | 0.571 ms | 0.704 ms | 0.769 ms |
| 接收 → 图像就绪 | 63.681 ms | 68.380 ms | 72.136 ms |
| 接收 → 输出完成 | 160.698 ms | 168.520 ms | 172.446 ms |

慢保存使用实际 PNG 写入前额外等待 300 ms 的受控场景，同步参考与后台输出各运行一组。它证明 GUI 阻塞机制，不能解释成正式的 30 次配对加速比。

| 受控慢保存指标 | GUI 同步参考 | 后台输出 |
| --- | ---: | ---: |
| SendInput 发送 → 收到消息 | 361.069 ms | 0.621 ms |
| GUI 心跳最大间隔 | 413.269 ms | 21.062 ms |

现有原生注册入口之后的调度已低于 1 ms 量级，耗时主要发生在捕获、编码和写盘。后台保存解决了已复现的消息阻塞；额外 C++ 输入模块不会让未被系统送达的按键自动到达，也不应以钩子抢占代替注册诊断。因此本轮不新增模块。

后续只有在实体 Windows 设备上复现“注册成功、没有输出阻塞，但仍持续丢失或显著延迟输入”，并排除系统保留键、其它软件占用、权限和宿主转发后，再评估独立原生消息线程。不同键盘布局若出现确定的 Qt 原生键值损失，也可单独评估更薄的原生录入层；不默认启用全局低级键盘钩子。

## 自动化回归与基线

| 最终回归 | 结果 | 记录 |
| --- | --- | --- |
| Windows 原生 Qt，全部截图相关测试、注册管理和设置导航 | **331 passed、1 skipped、46 subtests passed** | [windows-final-tests.log](windows-final-tests.log) |
| macOS offscreen，相同范围加原子 JSON/IO 测试 | **340 passed、46 subtests passed** | [macos-final-tests.log](macos-final-tests.log) |

Windows 跳过项为仅适用于非 Windows 的延迟导入保护检查。macOS 最终命令在允许本地 IPC 的执行环境运行；受限沙箱曾使同一套测试的单实例检查失败（Unix socket bind 返回 EPERM，Qt 报 `Unknown error 1`），解除该执行限制后整套通过，未为此改动产品代码。失败记录保留于 [macos-ipc-transient.log](macos-ipc-transient.log)。

完整 macOS 测试曾运行一次：**3843 passed、6 failed、1601 subtests passed**。其中本轮 Windows 探针缺少 `json.dumps(..., allow_nan=False)` 的 1 项序列化约束失败已修复，最终回归包含整个 `test_atomic_io.py`。其余 5 项在本轮开始前的 Git 暂存区源码快照中逐项复现：

- 导出选项页小窗口适配，dark / light 两项。
- 属性检查器区块连续性。
- 属性检查器固定区块顺序。
- 工作台响应式几何矩阵，1536 × 864 子场景。

保留 [full-suite.log](full-suite.log) 和 [macos-index-baseline.log](macos-index-baseline.log)。没有把后续局部通过写成全量通过，也没有将已有数字切片／工作台改动重置为 HEAD。

Windows 首轮暴露的旧测试问题已分开处理：主窗口测试的 fake client 缺少自启动参数，现隔离自启动同步；覆盖层像素测试按 DPR 采样；悬停测试显式使用 Qt `QWindow` 事件路径，保留候选与点击断言。原始索引快照中 6 个相关节点有 5 个失败、1 个通过，见 [windows-index-baseline.log](windows-index-baseline.log)。诊断发现 `QWidget` 路径在此虚拟机移动了光标但未向覆盖层送达 MouseMove；`QWindow` 路径可以送达。测试调整没有修改生产覆盖层，**不能据此宣布实体鼠标或 Windows 分层窗口命中问题已验收**。

本轮还增加了注册更新和回滚双重失败、暂停未成功、租约过期、超时后清理、重复 IPC、异常后继续调度、输出队列满、线程退出、保存失败重试、较旧输出完成顺序、录入失焦保留键值及禁用状态等回归。

## 复现与验收边界

Windows 项目根目录执行：

```powershell
$env:PYTHONPATH = 'src'
$env:QT_QPA_PLATFORM = 'windows'
$tests = @(Get-ChildItem tests\test_screenshot*.py | ForEach-Object { $_.FullName })
.venv\Scripts\python.exe -m pytest -q @tests tests\test_windows_global_hotkey.py tests\test_settings_dialog_navigation.py
.venv\Scripts\python.exe scripts\validate_screenshot_hotkeys_windows.py --output build\hotkey-check.json --samples 30
```

macOS 对应回归：

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_screenshot*.py tests/test_windows_global_hotkey.py tests/test_settings_dialog_navigation.py tests/test_atomic_io.py
```

Windows 项目位于 `C:\Users\lishuyang\Downloads\fiber-diameter-measurement`。同步前先核对旧内容、保存备份，再覆盖本轮相关文件；同步后逐个校验源码摘要。两端原有 15 个暂存路径均完整保留，暂存 diff SHA256 保持 `4d5deb4821e4f88794d1f0b336219e70a2e04895bd4debacb3100a2f60907b18`，证据见 [workspace-preservation.json](workspace-preservation.json)。这是实施验证时的现场记录。

本轮是源码环境验证，没有重新制作安装包。以下仍需独立验收：实体键盘与 Parallels 宿主拦截、非美式布局和 AltGr 实机、管理员前台程序、安全桌面、多显示器不同缩放，以及实体鼠标／Windows 分层窗口命中。上述限制均不因单元测试或 SendInput 探针通过而消失。

## 提交前审阅

审阅覆盖原生注册与回滚、录入生命周期、租约、请求去重、后台输出与退出、配置持久化，以及 Windows 验证脚本与测试隔离。未发现需要阻止本次提交的新问题。已清理归档日志的行尾空格，原始输出仍保留在本地 `build/hotkey-2026-10-14/`；没有改写失败记录或测试结论。

此前验证时的 `main_window.py` 包含原先已暂存的数字切片改动，因此上述源码摘要属于当时的工作区快照。提交使用独立 Git 索引，仅保留本次截图接入的两处修改；数字切片、IDE、研究文档和 egg-info 等原有暂存内容继续留在工作区。提交前回归另外加载候选索引中的源代码，确认截图功能不依赖这些未提交修改。

候选源码回归：**340 passed、46 subtests passed**，见 [commit-review-tests.log](commit-review-tests.log)。隔离加载的是 `fdm.ui.main_window` 和 `fdm.ui.digital_slide_canvas`；其它截图相关源文件、测试文件与候选索引逐一核对。Windows 探针记录的 6 个核心文件 SHA256 均与本次审阅源码一致，原有平台限制继续保留。
