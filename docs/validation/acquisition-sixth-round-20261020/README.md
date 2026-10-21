# 第六轮采集与试验回归

版本保持 **0.4.10**。本轮修复混合模式手工输入的按下／移动／松开边界、异步失败反馈归属、自动分析周期与候选增量发布；证据相册入口改为 40×22 逻辑像素的图标分裂按钮。没有修改现有视场稳定判定、测径算法、对齐阈值或帧龄预算。

**Windows 连续点选的“正常负载不隐藏余项”验收未通过。** 轻量追踪仍在 100 次点选中观察到一次约 335 ms 的安全暂停：主线程完整 GC 耗时 26.281 ms，使核验完成耗时 40.916 ms 超过原有 27.962 ms 预算。本轮没有禁用 GC 或放宽预算来隐藏此问题。候选包已完成构建、实际安装和验证，用于离线验证，不能视为本轮全部验收完成。

## 结果与证据

| 范围 | 结果 | 证据与边界 |
|---|---|---|
| 仓库回归 | 初始全仓 5,534 通过、35 失败、13 跳过、1,653 子测试；失败逐项处理，最终相关 410 通过 | [测试说明](tests/README.md)。33 项是受限环境的 IPC 权限问题，实际权限重跑通过；另 2 项旧即时刷新断言调整为下一轮事件发布，并补充时序边界。不是一次最终全仓运行全绿。 |
| 手工首点与晚到反馈 | 混合／仅手工、自动开／关、首次／连续测量交叉覆盖；每组首点后 1,000 次无按键移动不提交 | [手工输入](manual-input/README.md)。Qt 事件注入与有界轨迹，不等同于实体鼠标、Microview 实机。 |
| 自动周期 | 30 Hz 合成输入持续 60.001 秒仅分析一次；独立对齐失败后恢复原批次；显式新视场只新增一次 | [运行时回归](runtime/README.md)。768×576 合成图，实际导航、分析进程和严格核验。 |
| 延迟注入 | GUI 消费延迟 40／80／200 ms 不产生显隐转换；工作完成延迟各一次暂停／恢复，分析次数与批次不变 | [运行时回归](runtime/README.md)。故障注入运行，不作为隔离性能对照。 |
| macOS 正常负载点选 | 12 批次，100 接受／100 JPEG；其他候选整组隐藏 0 次 | [macOS 回归](macos/README.md)。实际分析和画布点击；合成相机、Qt offscreen。 |
| macOS 性能 | FPS 30.3036 → 30.3045；P95 帧龄增加 7.933 ms，小于一帧 32.999 ms | [性能与录像](macos/README.md)。同机隔离、固定候选样例；不将此算法样例性能称为实际全图算法或相机性能。 |
| 视觉 | 四主题，1600×900 的 100／125／150% 及 1366×768，共 16 组 | [视觉记录](visual/README.md)。图标、箭头、状态和类别区检查；150% 的可见类别数如实记录为 1，不声称所有比例均显示四类。 |
| Windows 选定回归 | 首轮 911 通过、50 子测试；记录发布最终定向 7 通过 | [Windows 日志与归因](windows/README.md)。原发布夹具有 5 项失败：Windows 同时间戳触发合法 ID 排序，与夹具插入次序假设不符；仅修正确定性时间戳并新增同时间戳排序测试，产品未修改。两个测试集合不累加为独立覆盖数。 |
| Windows 实际导航 | cycle60 通过；轻量 click100 为 12 批次、100 入账／100 JPEG，一次余项安全暂停约 335 ms | [Windows 诊断](windows/README.md)。严格显隐验收未通过；暂停不丢批次和剩余 3 条候选，恢复后继续接受。 |
| Windows 性能 | FPS 30.2636 → 30.3026；P95 帧龄增加 9.953 ms，小于一帧 33 ms | [同机对照](windows/performance-final-comparison.json)。此指标通过不替代显隐体验；固定候选样例，非相机或全图算法性能承诺。 |
| Windows 安装版、原生 UIA | 实际安装、普通用户自检、Explorer 启动、9 组原生 UIA 通过 | [安装证据](windows/README.md)。四主题、相册、存档、记录停靠和 Excel 导出；原生缩放覆盖至 125%，150% 来自 macOS 视觉检查。 |
| Microview 实体设备 | 待验收 | 没有接入真实相机；[离线检查步骤](OFFLINE_MICROVIEW_CHECKLIST.md)。 |

## 验证失败及保护边界

Mac 独立录像曾出现一次第 4 次点击超时，原始事件证据不足，保留为未完全归因。补充诊断的另一轮录像证实合成点击恰好位于新画布帧尚未实际绘制的间隙，呈现令牌与画布帧不同，产品精确帧保护拒绝手势。最终验证只等待真实绘制完成再发 Qt 点击，没有强行公布帧或放宽产品保护；12 次独立录像通过。详情与失败证据见 [macOS 说明](macos/README.md)。这些结果不能解读为实际用户每一次点击必然入账。

Windows 新增诊断完整记录核验算法耗时、工作提交至完成耗时、GUI 消费延迟和暂停调用点。初始截图确有一次测试自身造成的阻塞，轻量正式对照在截图后等待真实恢复才开始点击，仍复现了完整 GC 引起的安全暂停。即使 100 条记录均正确写入，余项短暂隐藏仍单独作为未通过项处理，不将其隐去或归为“通过”。

## 验证与交付工具

- `scripts/validation/validate_acquisition_sixth_round.py`：真实导航、真实全图工作进程、严格对齐、画布点击、账本及 JPEG；支持周期、点击和区分 GUI／工作任务的延迟注入。每轮保留产品指纹、脚本哈希和不覆盖的输出目录。
- `scripts/validation/sixth_round_windows.py`：将固定名单中的既有第五轮交付脚本适配到本任务 Windows 隔离目录，添加第六轮测试并强制可信 PowerShell 文件采用 `-File` 传输，记录源／生成脚本哈希。产品源码不做文本替换。
- Windows 工作目录固定为 `C:\tmp\fdm-cu-live-sixth-round-20261020`。预定内部构建参数为 `--exclude-area-models`，保留内部报告模板及 Microview 运行文件。

## 候选包

[Windows 内部离线候选安装包](../../../dist/internal/acquisition-sixth-round-20261020/FDM-0.4.10-round6-862b9fc24cf2.exe)，**398,835,901 字节**。

- 版本：`0.4.10`；构建 ID：`862b9fc24cf248af817271cb159fbbbb`。
- 安装包 SHA256：`7227c9d6709e81f854c0aca26e3f3d867f4326acce310cf56a67f1d9abda5a7b`。
- [交付清单](../../../dist/internal/acquisition-sixth-round-20261020/candidate-manifest.json)与[全部交付文件哈希](../../../dist/internal/acquisition-sixth-round-20261020/SHA256SUMS)。
- [原始源码清单](../../../dist/internal/acquisition-sixth-round-20261020/source-manifest.json)、[原始载荷清单](../../../dist/internal/acquisition-sixth-round-20261020/release-manifest.json)：构建端逐项核验 435 个产品源码及 4,818 个载荷文件；安装版自检实际启动了对齐子进程。
- [Windows 完整证据归档](../../../dist/internal/acquisition-sixth-round-20261020/windows-validation-evidence.zip)：15,686,243 字节，SHA256 `18fea4cbf6cfff05b126fae479f81a965a4c31e0911a9259c1e9e5d6d9db830a`；362 个文件，压缩包 CRC 检查通过。

源码为未提交工作区快照，包内 `source_commit=unknown`；`source_git_baseline=6278af2bb86f70f880332bfa84342a6436783813` 只说明 Git 起点，不代表本轮修改已提交。运行时验证脚本的单文件更新作为清单覆盖记录保留，产品文件没有改变。

第五轮候选及第四轮回退安装包保留。构建门禁和最终清单明确记录 `strict_windows_no_hide_passed=false`，不会将未通过验收改写为通过。旁置离线清单包含已知 335 ms 暂停限制，须随包交给实体设备测试人员。

## 环境恢复

限定 FDM 注册项已恢复；仅索引 0 的卸载注册项发生改变，恢复后逐项签名匹配，原安装的四个关键文件哈希不变。候选目录保留在本任务隔离路径 `installed-candidate-validated`，没有整体回滚虚拟机或运行卸载器。见[恢复结果](windows/registration-restored.json)。

Parallels `Windows 11` 已优雅停止，自动共享 Mac 相机恢复开启，与[开始状态](windows/vm-before.json)一致；见[结束状态](windows/vm-after.json)。没有遗留本任务运行中的验证或构建进程。

所有验收结果区分 macOS、Windows 虚拟机源码、Windows 安装版原生输入与实体 Microview。实体相机至少 30 分钟动态采集、原始帧上的对齐阈值、实体多显示器／DPI 切换仍须由离线设备执行。
