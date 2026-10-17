# 工作区重构验收

本目录记录真实 `MainWindow` 与模拟采集后端的界面验收。所有运行使用独立临时配置；相机后端、电机串口和截图工具 IPC 均被替换，不访问用户设备或配置。

## 证据状态

最终结果：**44 项显示/工作区回归通过，25 组主题/尺寸/DPI 独立进程验收通过**。每组均执行 203 次子页转换和 20 次退出返回，并检查所有后续截图尺寸。没有非预期顶层窗口、Qt slot 异常或空试验 dirty 状态。25 组使用相同源码指纹，运行中没有源码变化，整理时与当前相关源文件一致，包含最终隐藏页面布局、关闭取消生命周期、显示中心恢复和小客户区导航显隐顺序修复。自动测试用时 23.42 秒，见 [tests.log](tests.log)；完整矩阵见 [matrix.json](matrix.json)。

| 主题 | 物理客户区 | DPR | 实际逻辑尺寸 | 结果 |
| --- | --- | --- | --- | --- |
| dark / light / navy / warm | 1366×768 | 1 | 1366×768 | 4 / 4 通过 |
| dark / light / navy / warm | 1366×690 | 1 | 1366×690 | 4 / 4 通过 |
| dark / light / navy / warm | 1280×670 | 1 | 1280×670 | 4 / 4 通过 |
| dark / light / navy / warm | 1920×1080 | 1 | 1920×1080 | 4 / 4 通过 |
| dark / light / navy / warm | 3840×2160 | 2 | 1920×1080 | 4 / 4 通过 |
| dark | 1920×1080 | 1.25 / 1.5 | 1536×864 / 1280×720 | 2 / 2 通过 |
| dark | 3840×2160 | 1 / 1.25 / 1.5 | 3840×2160 / 3072×1728 / 2560×1440 | 3 / 3 通过 |

本目录保留 134 张 PNG：四主题 1366×768 的全部 26 个页面/状态，1080p 150% 和 4K 200% 各三张代表图，以及四主题在两个新客户区下的实时/含量/直径三页（24 张）。各组完整报告位于 `reports/`；未重复存入仓库的全套图片仍位于 `/tmp/fdm-workflow-redesign-final`。

视觉抽查涵盖四主题的试验准备/错误状态、八个全局设置页、数字切片与 4K 含量界面：导航与常用表单可见，重量字段常驻，记录/报告不遮挡画面，小尺寸溢出内容留在滚动区域。可直接对比 [深色含量](dark-1366x768-1/content-operation.png)、[浅色含量](light-1366x768-1/content-operation.png)、[深蓝断线](navy-1366x768-1/trial-capture-error.png)、[暖灰设置](warm-1366x768-1/settings-0.png) 和 [4K 含量](dark-3840x2160-2/content-operation.png)。

`intermediate/` 中直接存放的旧报告和 `intermediate-matrix.json` 是补齐严格尺寸断言之前的中间证据，**不作为最终验收通过依据**。这些运行曾发现窗口高度 775 / 777 的膨胀，以及 150% 下回测量的瞬时 750 高度。保留历史用于说明检查如何发现问题；最终矩阵已在修复隐藏页面尺寸提示和导航显隐顺序后重新完整运行。

[Windows 导出间距微调前的报告](intermediate/before-windows-export-1px-fix/HISTORY.md) 另行保留了此前严格验收记录。Windows 原生检查随后发现导出窗口 900×544 在四主题下多出 1 像素滚动条，`ExportOptionsDialog` 页布局间距由 5 调整为 4。当前矩阵和过程录像已在该微调后重跑；此工作区矩阵本身不覆盖导出对话框，导出专项验收由对应 Windows 报告说明。

[小客户区修复前的报告](intermediate/before-small-client-route-fix/HISTORY.md) 保留了之后 Windows GUI 发现的边界：请求 1366×690 / 1280×670 时进入试验曾撑高到 715；macOS 新增回归同样失败。原因包括先显示试验页、后隐藏旧实时工具栏，以及同步工作区时先 show 后 hide 测量工具栏。修复后直接应用最终显隐状态，且进入试验前先隐藏旧工具栏。新增两个客户区各四主题完整跑过；测试检查每次导航后的实际窗口尺寸，既不以最终 minimumSizeHint 代替，也不在导航后 resize 窗口掩盖膨胀。新尺寸的九类别滚动可达性回归也通过。

过程录像的视觉复核另外发现：没有用户平移时，离页布局变化曾将归一化中心从 `(0.5, 0.5)` 错写成约 `(0.712, 0.603)`。修复后，布局等待恢复期间不采集中心，页面显示/尺寸变化强制重新恢复变换；最终矩阵逐次检查中心偏差，允许的误差仅为整数滚动条对应的一个逻辑像素。新增真实鼠标中键平移回归也验证跨页、窗口 resize、采集重启及主窗口销毁重建确实保留用户中心，避免用禁止所有位置保存来掩盖问题。

## 切换过程录像

[打开过程录像](recording/workflow-offscreen.mp4)（H.264、1280×702、20 fps、311 帧、15.55 秒）记录同一个真实 `MainWindow` 的连续导航：自动进入采集、含量/直径/实时采集切换、记录页、12 次快速切换、退出返回、停止重启、模拟断线重试和共享显示方式。每帧顶部明确标记 `OFFSCREEN · 模拟相机`；下方 1280×670 是窗口客户区，覆盖本轮新发现的最小边界。视频已完整解码检查，并抽查记录页与断线/恢复阶段；[媒体参数](recording/media.json) 来自 `ffprobe`。

[过程事件与帧时间](recording/recording.json) 保存每步入口、视频帧号、实际墙钟时间和全局顶层窗口 Show 事件。没有非预期窗口或 Qt slot 异常；源码指纹与最终 25 组矩阵一致。恒定帧率编码会使播放时长与真实操作时长略有不同，JSON 保留了真实时间。20 fps 视频不能单独排除帧间短暂窗口，因此仍结合 Show 事件观察器判断。此录像来自 Qt offscreen 渲染和确定性模拟帧，不是 Windows 桌面或真实相机录像。

## 可重复执行

```sh
.venv/bin/python -m pytest -q tests/test_live_display_preferences.py tests/test_acquisition_workflow_redesign.py
.venv/bin/python scripts/validation/validate_workflow_redesign.py --theme dark --size 1366x768 --scale 1 --output /tmp/fdm-workflow-final-dark
.venv/bin/python scripts/validation/validate_workflow_redesign.py --theme light --size 1920x1080 --scale 1.5 --output /tmp/fdm-workflow-final-150
.venv/bin/python scripts/validation/validate_workflow_redesign.py --theme navy --size 3840x2160 --scale 2 --output /tmp/fdm-workflow-final-4k
.venv/bin/python scripts/validation/validate_workflow_redesign.py --theme light --size 1366x690 --scale 1 --output /tmp/fdm-workflow-final-client-690
.venv/bin/python scripts/validation/validate_workflow_redesign.py --theme dark --size 1280x670 --scale 1 --output /tmp/fdm-workflow-final-client-670
.venv/bin/python scripts/validation/record_workflow_redesign.py --size 1280x670 --output /tmp/fdm-workflow-recording-final
```

每个 DPI / 主题组合使用独立进程。`--size` 指窗口客户区的物理像素尺寸，逻辑尺寸为物理尺寸除以 `--scale`。屏幕基线是 1366×768、100%，另外用 1366×690 / 1280×670 客户区覆盖标题栏、任务栏及较小工作区；1080p、150% 对应 1280×720。这些小客户区验收滚动可达性，不要求九个类别全部同时可见。截图输出的实际逻辑尺寸、像素尺寸和 DPR 均进入 JSON。可选录像脚本要求 PATH 中有 `ffmpeg`；它同样只使用隔离配置和模拟设备。

每份 JSON 记录实际解析的字体、字号和中文样本文字度量。当前矩阵使用 macOS `.AppleSystemUIFont`（9 pt、12 px、行高 15），没有声明等价于 Windows 字体。脚本支持诊断参数 `--font-family` / `--font-point-size`，并明确记录字体是否安装和实际回退家族；本机没有 Segoe UI / Microsoft YaHei，原生 Windows 字体与冻结程序仍以 Windows GUI 专项证据为准。

## 自动检查

- 使用实际主窗口 QAction 和 QTabBar 鼠标事件进入实时采集、含量试验、直径试验；默认 200 次子页切换、20 次退出返回。后端只能维持一个当前会话，页面缓存固定为三页和两个试验实例。
- 用全局 Qt Show 事件观察器捕捉一闪而过的顶层窗口；检查导航可见区域、原生目标隐藏、根工作区没有被提升为原生窗口。
- 项目与类别、测量概览、结果面板按用户选择恢复；空试验保持项目干净，真实关闭前置检查不弹保存提示。
- 实际像素、适合窗口、指定比例、物理倍率跨三个页面、停止重开及主窗口销毁重建恢复。百分比与倍率保留独立最后数值；缺少标定时保留用户选择并回退实际像素。
- 无帧超时、完整帧恢复、设备错误、内联重试、异步停止期间快速重入，以及旧会话迟到的帧和错误均单独验证。
- 记录和报告切换前后图像区域矩形不变；九类别在小窗口中均可通过滚动到达。
- 截图包括测量分析、实时采集、两个试验的操作/记录/报告/高级选项、数字切片、八个全局设置页、无设备和采集错误状态。
- 捕捉 Python Qt slot 异常，避免回调报错但外层脚本误判通过。每次页面转换和截图均检查窗口未突破请求尺寸。

## Windows 实机验收仍需执行

这些结果不证明真实 SDK、相机、混合显示器或设备吞吐量。发布前应在装有受支持 Microview 设备的 Windows 环境检查：

1. 在 1366×768、100%，1080p 的 100%/125%/150%，4K 的 100%/125%/150%/200% 分别运行脚本；对有原生标题栏的系统另检查实际可用工作区和任务栏。
2. 使用真实 SDK 连续执行同样的进入、三页切换、停止、退出返回、快速重入路径，确认设备只打开一次，停止完成后才能重开，旧 HWND 仍由同一会话拥有且始终不可见。
3. 对真实 768×576 及设备实际支持的高分辨率图像检查帧率、CPU/GPU、内存、后台停止时间和断线重连；模拟帧的 UI 测试不能替代这些指标。
4. 跨不同 DPI 的显示器拖动同一窗口，核对实际像素 1:1、指定比例、视图中心；更换屏幕或物理分辨率后物理倍率应提示重新校准屏幕标线，原始测量坐标和已接受记录不变。
5. 对准备中、候选待确认、超限确认、异步图像保存中的操作执行阻止/恢复检查，确认提示可操作，关闭过程不会丢失已接受结果。
