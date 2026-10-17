# 第二轮采集界面视觉与过程回归

2026-10-17 最终结果：**25 / 25 组通过，600 张截图、2 段过程录像，488 次三页切换、5,000 次视场结束与状态更新检查。** 所有已记录的客户区尺寸、截图像素尺寸均等于请求值；最终各组的 60 项生产源码及图标指纹与当前文件一致，没有运行中源码变化、未处理 Qt 槽异常、意外模态提示或多余顶层窗口。

本轮运行环境为 macOS 27.0.1 ARM64、Python 3.13.12、Qt `offscreen`，字体解析为 `.AppleSystemUIFont` 9 pt。使用真实 `MainWindow`、真实导航和隔离配置，通过 `WorkflowHarness` 的模拟采集后端提供测试帧。**本报告不是 Windows、Microview 真机、Excel 或冻结安装包验收。**

## 最终矩阵

每一行均覆盖实验室浅色、石墨深色、深蓝、暖灰四个主题：

| 请求截图 / 客户区像素 | Qt 缩放 | 实际逻辑客户区 | 结果 |
| --- | --- | --- | --- |
| 1366 × 768 | 100% | 1366 × 768 | 4 / 4 |
| 1280 × 670 | 100% | 1280 × 670 | 4 / 4 |
| 1920 × 1080 | 125% | 1536 × 864 | 4 / 4 |
| 1920 × 1080 | 150% | 1280 × 720 | 4 / 4 |
| 3840 × 2160 | 150% | 2560 × 1440 | 4 / 4 |
| 3840 × 2160 | 200% | 1920 × 1080 | 4 / 4 |

另加石墨深色 1280 × 670、侧栏宽 390 px：1 / 1。其余组侧栏宽 432 px。

这里的尺寸是 Qt 客户区与截图像素，使用 `QT_SCALE_FACTOR` 验证布局和栅格缩放。`offscreen` 虚拟屏幕不能代表真实操作系统任务栏、标题栏或跨显示器 DPI 切换。

## 检查内容与结果

- 通过实际导航依次进入实时采集、含量试验、直径试验。石墨深色 1366 × 768 连续切换 200 次，其余 24 组各 12 次；三页图像区域边界一致，导航始终可见，采集后端保持单次启动。
- 每组分别对含量和直径执行 100 次视场结束、稳定 / 核验状态交替更新。图像、侧栏、操作 / 记录 / 报告区域、样品输入与保存按钮的矩形不变，已准备完成的提示不会反复占用高度。
- 触发当前控制器的 `no_complete_crossing` 拒绝通路，检查本次未通过与未计入记录提示、具体落点建议、无内部错误码泄露、账本无变化、提示卡片完整位于画面内且允许点击穿透。
- 在 390 px 侧栏检查上一条有效测量 `123.456 μm`：含量数值可用宽 231 px、所需 127 px；直径可用宽 334 px、所需 127 px。失败卡片正文所需高度 34 px，实际 34 px，没有截断。
- 将目标设为 2 条，验证第二条入账时显示 `2 / 2` 和达标反馈。保留卡片可见，实际点击撤销：显示 `1 / 2`、最近有效值 `123.456 μm`、旧卡片消失；再次撤销后显示 `0 / 2`、`— μm`、空记录说明。实际重做两次后恢复 `2 / 2` 和 `20.000 μm`，不重放已通知的卡片。25 组共执行 200 次真实撤销 / 重做按钮点击。
- 修改目标时清除已经过期的达标卡片，并显示 `3 / 4`。含量试验通过真实补计 `+` / `−` 按钮达到总目标、减回目标以下、再次达到同一目标：数量同步刷新，旧卡片消失，再次达到不重放。共执行 100 次真实增减按钮点击。操作、记录、报告之间切换不改变图像大小。
- 所有导航入口和数量设置图标均非空。目视检查最终浅色小窗、深色窄栏、深蓝 150%、暖灰 200% 的数值、图标与反馈文字，并对最终录像解码抽帧复核。小客户区的操作内容使用正常纵向滚动；底部保存入口固定可达。

为验证反馈呈现，`123.456`、`20.000` 与 `22.000 μm` 是带 `validation_fixture` 标记的账本测试数据，由正常控制器发布信号刷新界面。失败原因在现有拒绝边界注入。该过程不证明测径数值准确性；5,000 次检查验证状态呈现的几何稳定性，不是视场检测算法精度、实时吞吐量或相机延迟测试。

## 发现并关闭的问题

初次目视复核发现：有效结果已经入账且显示数量达标提示后，底部仍短暂保留上一条失败信息。主实现已在单条 / 批量接受记录时更新底部反馈，并更新批量最新值。本脚本增加“达标时底部包含已计入记录且不含未识别到”的断言，随后重新执行完整 25 组矩阵，全部通过。最终截图中的底部信息为“本次有效测量 20.000 μm 已计入记录。”

后续独立审查还发现：撤销、减计或修改目标后，旧达标卡片可能保留过期数量，记录撤销后最新值也需同步重建。生产实现已按当前有效记录重建数值，并在原达标快照失效时清除卡片、保留通知去重历史。之前脚本在撤销前主动关闭卡片，没有覆盖这个场景；本次已改为卡片可见时直接点击真实按钮，并增加上述历史状态断言与每组 10 张截图。完整 25 组重新通过。

最终结果仅来自 `final-current-after-history/`。早期 `final/`、`settled-final/`、`post-dialog-fix/` 与 `final-current/` 原始证据均保留在本机；最后一组是历史反馈修复前的结果。全部中间结果都不计入最终通过数。

## 证据与复现

- [结构化结果及 SHA-256 清单](visual-results.json)：25 组结果路径、600 张截图校验、2 段录像校验、源码指纹与验证边界。
- [验证脚本](../../../scripts/validation/validate_acquisition_second_round.py)。脚本支持 Windows 平台参数，但本轮只执行了 macOS `offscreen`。
- 本机原始证据：`.tmp/acquisition-second-round-20261017/final-current-after-history/`；截图、隔离配置、日志和录像均留在被 Git 忽略的目录。

关键截图与录像：

- [390 px 侧栏与 123.456 μm 失败反馈](../../../.tmp/acquisition-second-round-20261017/final-current-after-history/dark-1280x670-1p0-sidebar390/content-invalid-measurement.png)
- [浅色 1280 × 670 数量达标与更新后的底部反馈](../../../.tmp/acquisition-second-round-20261017/final-current-after-history/light-1280x670-1p0/content-quantity-target.png)
- [实时采集右侧功能区](../../../.tmp/acquisition-second-round-20261017/final-current-after-history/dark-1366x768-1p0/realtime.png)
- [暖灰 4K / 200% 数量达标](../../../.tmp/acquisition-second-round-20261017/final-current-after-history/warm-3840x2160-2p0/content-quantity-target.png)
- [真实撤销后剩余一条记录](../../../.tmp/acquisition-second-round-20261017/final-current-after-history/light-1280x670-1p0/content-target-undo.png)
- [全部撤销后的空记录状态](../../../.tmp/acquisition-second-round-20261017/final-current-after-history/light-1280x670-1p0/content-target-undo-empty.png)
- [修改目标后清除旧达标卡片](../../../.tmp/acquisition-second-round-20261017/final-current-after-history/navy-1920x1080-1p5/diameter-target-changed.png)
- [石墨深色过程录像](../../../.tmp/acquisition-second-round-20261017/final-current-after-history/dark-1366x768-1p0/process.mp4)：H.264，1366 × 768，15 fps，154 帧，10.267 秒。
- [实验室浅色小窗过程录像](../../../.tmp/acquisition-second-round-20261017/final-current-after-history/light-1280x670-1p0/process.mp4)：H.264，1280 × 670，15 fps，128 帧，8.533 秒。

录像来自实际导航过程中的 Qt 窗口抓帧；原始抓帧时间戳保留于每组 `result.json`，15 fps 编码时长不作为性能结论。两段录像均经过 `ffprobe` 视频流元数据核对，浅色录像另抽取第 2 秒画面目视确认。

从项目根目录复现基线：

```sh
.venv/bin/python scripts/validation/validate_acquisition_second_round.py \
  --theme dark --size 1366x768 --scale 1 --platform offscreen \
  --cycles 200 --motion-cycles 100 --video \
  --output .tmp/acquisition-second-round-repeat/dark-1366x768
```

录像需要 `ffmpeg` 位于 PATH。其余矩阵组使用 `--cycles 12`；窄栏检查增加 `--sidebar-width 390`。运行采用独立临时配置，不操作用户项目或真实采集设备。
