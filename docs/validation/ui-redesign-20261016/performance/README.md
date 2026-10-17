# UI 重构性能与资源打包验证

2026-10-16，macOS 27.0.1 / arm64、Python 3.13.12、Qt 6.10.2、offscreen。正式采样在根任务确认集成测试退出、Windows 任务暂停构建之后串行完成；原始 JSON 含源码摘要、原生构建摘要、依赖版本、窗口尺寸、输入摘要和全部尝试。

**结论：8 个短程组合中 6 个通过既有门槛，4K 动态输入的 2 个组合失败。不能将本轮描述为性能全面通过。**

## 试验页面的连续帧与测量响应

使用实际 QWidget/QPainter、Qt 两次点击、原生测量内核、30 Hz 合成输入；每组预热 5 次、正式尝试 30 次，深色主题、1366×768、适应窗口。下表使用原项目 [固定性能门槛](../../cu-live-measurement-2026-10-15/performance-acceptance-rules.md)，未调整阈值或补测失败样本。

| 输入 | 尺寸 | PNG | 成功/正式尝试 | 结果可见 P95 / P99 (ms) | 不同帧绘制 FPS | 结论 |
|---|---:|---|---:|---:|---:|---|
| 静止 | 2048 | 关 | 30/30 | 8.38 / 8.81 | 30.34 | 通过 |
| 静止 | 2048 | 开 | 30/30 | 10.07 / 10.14 | 30.53 | 通过 |
| 静止 | 4096 | 关 | 30/30 | 9.17 / 12.93 | 30.19 | 通过 |
| 静止 | 4096 | 开 | 30/30 | 7.57 / 10.43 | 30.37 | 通过 |
| 逐帧变化 | 2048 | 关 | 30/30 | 26.91 / 27.42 | 30.24 | 通过 |
| 逐帧变化 | 2048 | 开 | 30/30 | 24.45 / 24.99 | 30.37 | 通过 |
| 逐帧变化 | 4096 | 关 | 30/30 | 565.01 / 806.07 | 30.09 | 延迟失败 |
| 逐帧变化 | 4096 | 开 | 28/30 | 1047.81 / 1397.29 | 30.09 | 延迟、拒收失败 |

所有组合草稿显示 P95 ≤ 2.80 ms，GUI 同步发布最大值 ≤ 1.92 ms。4K 动态 PNG 开启时，正式尝试有 2 次“视场核验超过两秒”拒收，另有 1 次预热拒收；表中成功样本分位数不能掩盖这些失败。

[原始静止数据](cu-ui-stable.json)、[原始动态数据](cu-ui-dynamic.json)、[现有审计器生成的逐项结论](absolute-threshold-audit.json)。`diagnostic-concurrent-*` 是发现并发集成测试后保留的非正式数据，未纳入结论。

## 真实主窗口、导航与坐标

[主窗口连续流记录](main-window-stream.json) 使用后台生产线程，经真实 CaptureSessionManager 输入，再走 MainWindow 的实时采集、含量试验、直径试验导航。

- 2K 的六段页面流程完成，采集后端仅启动一次。试验页首次创建约 106–130 ms，缓存后切换约 12–26 ms。GUI 10 ms 心跳的 P95 为 11.11 ms，最大值 135.30 ms；初次构建存在同步延迟，不能宣称完全没有停顿。
- 试验页不同帧绘制约 28.49–30.50 fps。四次测量的 200 ms 草稿阶段各继续绘制 6 帧；记录均使用第一击锁定帧，四次原生测径均为预期 19 μm。
- 输入坐标最大误差 0.919 原图像素，小于该适应比例下 Qt 整数点击的 1.230 像素舍入容差；未观察到坐标或来源帧串用。
- 4K 主窗口用例在第二击后的入账等待超时，测试中止，未得到完整该尺寸报告。保留退出码 1 的 [原始日志](main-window-stream.log)，总体 `correctness_checks_passed=false`。原始探针未记录本次超时的具体拒收事件，不能仅凭另一个基准的结果断言同一根因。脚本现已补充失败持久化与拒收事件收集，未用重跑结果替换本次失败。

## 比较边界与打包

[历史基线评估](baseline-assessment.json) 说明没有可隔离归因于本次 UI 重构的同配置前后配对。此前 [r4 审计](../../cu-live-measurement-2026-10-15/performance-audit-short-final-r4.md) 已记录 macOS 4K 动态输入长尾与 28/30 入账；本轮仍未解决这类问题，不据不同配置计算提升百分比。

[wheel 资产检查](wheel-assets.json) 通过：隔离源码副本离线构建 wheel，36 个官方 Fluent SVG、MIT LICENSE、NOTICE、manifest 与源文件逐字节一致；包含主题、图标和 WorkspaceStack 模块。使用 `FDM_BUILD_NATIVE=0`，未安装 wheel，此检查仅验证资源包含关系，不是发布包或原生运行时验收。

资源采样由独立进程执行，仅统计目标进程 RSS/CPU，明确排除子进程。没有运行后台全图算法、一小时联合负载、真实相机、Windows 桌面或物理屏幕扫描验收；这些不能由本次合成 macOS offscreen 数据代替。

复现命令（正确性检查与构建结束后串行执行）：

```sh
.venv/bin/python scripts/benchmark_cu_live_ui.py --sizes 2048 4096 --warmup 5 --samples 30 --evidence both --display-mode fit --window-width 1366 --window-height 768 --theme dark --resource-parent-only --motion-input stable --output /tmp/cu-ui-stable.json
.venv/bin/python scripts/benchmark_cu_live_ui.py --sizes 2048 4096 --warmup 5 --samples 30 --evidence both --display-mode fit --window-width 1366 --window-height 768 --theme dark --resource-parent-only --motion-input lsb-noise --output /tmp/cu-ui-dynamic.json
.venv/bin/python scripts/validation/benchmark_workflow_redesign.py --sizes 2048 4096 --phase-seconds 2 --output /tmp/main-window-stream.json
```
