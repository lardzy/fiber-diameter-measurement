# 快速测径：减少误拒绝与细化失败原因

日期：2026-09-15。基于[首轮提速实现](quick-diameter-implementation-2026-09-15.md)继续调整，本轮测径自检修订号为 `2`。后续扩展像素功能修复及修订号 `3` 的验证见[像素修正记录](quick-diameter-pixel-correction-2026-09-15.md)。本文保留修订号 `2` 时的验证结果。

## 1. 调整后的行为

这轮主要解决“纤维已经高亮，但没有可用直径线”的几何判断问题：

- **短而完整的纤维**：在整体形状具有明确长轴、近似实心凸形时，使用原分辨率掩膜的主轴补充骨架方向。要求主轴与最小包围矩形方向一致，并验证相邻三个横截面的完整性和宽度稳定性。
- **靠近图像边框的完整纤维**：依据横截面能否找到两侧真实边界判断，不再仅凭距离边框近就拒绝。修复形态学清理把距边框 2 px 的完整区域扩到边框的问题。
- **纵向穿过视野的纤维**：只要横截面完整，允许测量；真正缺失横截面一侧边界时仍然拒绝。去边产生的人工边界也不能当作纤维边界。
- **很长、很细的区域**：缩小骨架工作图时兼顾前景宽度和总像素预算，避免仅为满足 640 px 长边限制而把细纤维压成一两个像素。
- **短交叉分支**：修剪时避免把所有分支一起删光。正常候选不足时，在较小的交叉排除范围内再尝试一次，并要求相邻三个截面稳定。
- **候选选择**：先剔除置信度不足的候选，再选择接近中位宽度的代表线，避免一个低分候选使其它可用候选一起失效。

保留编译版 Zhang 骨架、后台预热、原图端点搜索、取消和过期结果抑制。所有恢复路径都有次数或候选数量上限，使用原有 3 秒几何期限。个别本地库调用执行期间仍不能即时中断。

## 2. 合成样本结果

下面的“调整前”是首轮编译骨架版本，“调整后”是本轮修订；并非最初的纯 Python 骨架版本。

| 样本组 | 调整前成功 | 调整后成功 | 总数 |
| --- | ---: | ---: | ---: |
| 不同宽度、角度的短纤维 | 30 | 42 | 45 |
| 细纤维 | 3 | 4 | 5 |
| 靠近图像边框 | 0 | 4 | 4 |
| 沿长度方向穿过视野 | 0 | 1 | 1 |
| 大跨度斜纤维 | 0 | 1 | 1 |
| 4096 px 图像中的长细纤维 | 0 | 1 | 1 |
| 短十字交叉 | 3 | 5 | 6 |
| **合计** | **36** | **58** | **63** |

已成功的 36 例全部保留。成功样本相对于生成参数的最大宽度误差为 **2.0 px**，有单一已知方向的样本最大方向偏差约 **4.0°**。断言容差为 2.1 px，包含离散像素及端点取样误差。

仍拒绝的 5 例：3 个仅约 8 px 宽、长宽比 1.2 的低分辨率短块方向不明确；1 个标称 4 px 的水平细条实际只有 3 行前景、像素中心间宽度不足 3 px；1 个短交叉缺少足够方向信息。这些样本未强行生成线段。

另用 30 例短交叉组合检查不同角度与分支长度：25 例成功，最大宽度误差 2.0 px、最大方向偏差约 8.3°；5 例宽 16 px、总长度约 3 倍宽度的短交叉继续拒绝。该组与前述交叉样本有重叠，不合并为独立样本总数。短分支方向回归限值为 10°；原有较长纤维的严格方向检查继续保留。

原始数据：[调整前](research/quick-diameter-robustness-2026-09-15/before.json)、[调整后](research/quick-diameter-robustness-2026-09-15/after.json)、[短交叉补充检查](research/quick-diameter-robustness-2026-09-15/cross_stress.json)。这组成功数不能代替真实显微图像上的成功率或计量精度验收。

## 3. 失败提示与诊断

几何服务新增 `FiberQuickGeometryError`，同时携带用户提示和可记录的原因码。界面继续显示 worker 返回的提示。

| 原因码 | 实际判断 | 提示的处理方向 |
| --- | --- | --- |
| `incomplete_boundary` | 搜索到图像外缘仍未遇到背景，或端点落在去边产生的人工边界 | 移动视野，取得完整的两侧边界 |
| `target_too_thin` | 区域或候选线的像素宽度不足 | 使用更高分辨率图像 |
| `ambiguous_direction` | 局部骨架或整体形状不能提供稳定方向 | 选择更完整的纤维段 |
| `unstable_width` | 恢复路径的相邻横截面不完整，或宽度变化过大 | 选择边界更清晰的纤维段 |
| `width_mismatch` | 横截面长度与局部距离变换估计明显不符 | 补点排除交叉、粘连 |
| `no_boundary` | 无法获得完整的掩膜边界端点 | 补点修正纤维区域 |
| `target_too_small` / `target_too_large` | 可用面积不足 32 px，或前景覆盖达到图像的 40% | 扩大有效目标，或排除背景和相邻纤维 |
| `no_usable_branch` / `no_usable_skeleton` | 没有可用分支或方向候选 | 缩小到单根纤维，或选择完整纤维段 |
| `low_confidence` | 过滤后没有达到最低评分的候选 | 补点选择稳定截面 |

空掩膜、清理后为空、超时、组件加载失败也保留各自提示。一个请求有多种候选拒绝原因时，界面显示出现最多的原因；它是几何判断结果，不能据此断言真实样本一定存在粘连等物理问题。界面收到缺少有效线段的异常成功结果时，仍保留“未找到可靠直径线”的兜底提示。

失败日志标题为 `Quick diameter failed`。几何拒绝会记录原因码、候选拒绝计数、尝试路径、面积/工作区域、边缘处理标志、请求编号、排队与计算耗时；字段随失败阶段而异。过期或取消的任务不写失败日志。

- Windows 默认位置：`%LOCALAPPDATA%/FiberDiameterMeasurement/logs/startup.log`，环境缺失时沿用现有 APPDATA / 用户目录回退规则。
- macOS：`~/FiberDiameterMeasurement/logs/startup.log`。

## 4. 耗时检查

本轮观察到明显的运行时波动，单独运行 6 例的几何中位数为约 7–226 ms。因此增加同一进程、预热后、前后版本交替顺序的比较，每个版本每例预热 2 次、记录 7 次。

| 合成目标 | 首轮版本中位数，ms | 本轮版本中位数，ms |
| --- | ---: | ---: |
| 小矩形 | 5.12 | 5.60 |
| 横向 60 px | 57.67 | 59.19 |
| 斜向 40 px | 18.94 | 19.22 |
| 斜向 120 px，2048×1536 | 35.62 | 38.55 |
| 弯曲 48 px | 17.45 | 18.10 |
| 十字交叉 | 4.51 | 4.53 |

在这次交替比较中，新增判断带来的中位耗时差约 0.02–2.93 ms。6 例均正常返回，代表宽度未改变。不能把不同时间运行的绝对数值直接换算成可靠的速度比，也不能将它们当成首次点击或完整界面的延时保证。

数据：[单次版本基准](research/quick-diameter-robustness-2026-09-15/performance.json)、[同进程交替比较](research/quick-diameter-robustness-2026-09-15/paired_performance.json)。后者记录两个服务源文件的 SHA-256，首轮到本轮的几何差异另存为 [robustness.patch](research/quick-diameter-robustness-2026-09-15/robustness.patch)。该补丁使用无上下文格式，复现时通过 `git apply --unidiff-zero` 应用。

## 5. 打包与冻结验证

继续使用首轮补齐的正式 PyInstaller 钩子和锁定依赖，本轮没有引入新的库。发行自检现在要求 `geometry_revision == 2`，防止安装器复用只通过旧版测径探针的目录。

`functional_checks.fiber_quick_geometry` 实际运行 5 个正例和 1 个拒绝检查：

- 直线、斜线、十字、短纤维、距图像边框仅 2 px 的完整纤维。
- 缺失横截面一侧边界时，必须返回 `incomplete_boundary`。
- 同时验证本地编译扩展及长条骨架连通性。

**实际完成 macOS ARM64 冻结构建及运行**：PyInstaller 6.19.0、hooks-contrib 2026.3、Python 3.13.12、scikit-image 0.26.0、SciPy 1.18.1。进程从 `/tmp` 运行，移除 Python 路径/虚拟环境变量；探针将标准输入、输出、错误流设为 `None` 后运行几何检查。包内扩展确认通过，5 个正例测得约 39 / 24 / 40 / 15 / 39 px，拒绝检查通过。

首次冻结启动曾在几何计算前等待数分钟。进程采样显示停留在 Qt 动态库导入时的 dyld 映射/签名相关 `fcntl` 调用，未据此修改系统设置；不能把这段等待归因于测径算法。随后同一可执行程序完整进程再次启动、运行全部检查并退出耗时约 **2.80 秒**。这仍是探针启动检查，不是完整产品启动验收。

证据：[构建日志](research/quick-diameter-robustness-2026-09-15/frozen_build.log)、[冻结结果](research/quick-diameter-robustness-2026-09-15/frozen_result.json)、[再次运行](research/quick-diameter-robustness-2026-09-15/frozen_repeat_result.json)、[首次启动采样](research/quick-diameter-robustness-2026-09-15/frozen_startup_sample.txt)。

**尚未在 Windows 真机生成并安装完整安装包。** Windows 本地库加载、完整界面、模型推理和用户实际样本仍需随目标安装包验证。当前结果不能代替 Windows 安装后验收。

## 6. 回归结果与复现

- 测径、后台任务、打包、发行自检和训练导出集成：**122 passed，217 subtests passed**。
- 快速测径界面用例：**8 passed**，覆盖异步分发、提前确认、快捷键和线程关闭。
- `uv lock --check`、`git diff --check` 通过。
- 单独复核原有 JSON 序列化契约测试：仍有 **1 项既有失败**，对应轮廓比较功能的 6 处缺少 `allow_nan=False`。涉及三个文件均逐字节确认与 HEAD 相同。本轮新增失败日志显式使用 `allow_nan=False`。本轮未重新跑整个测试套件；首轮整套测试结果保留在首轮记录中。

日志：[相关测试](research/quick-diameter-robustness-2026-09-15/tests_related.log)、[界面测试](research/quick-diameter-robustness-2026-09-15/tests_ui.log)、[既有失败](research/quick-diameter-robustness-2026-09-15/tests_existing_json_check.log)。[验证汇总](research/quick-diameter-robustness-2026-09-15/validation_summary.json)包含失败文件列表及源文件校验值。

从仓库根目录复现主要检查：

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync python -B -m pytest tests/test_fiber_quick_geometry.py tests/test_background_task_controllers.py tests/test_build_support.py tests/test_build_windows_onedir.py tests/test_build_windows_installer.py tests/test_release_self_check.py tests/test_training_dataset_integration.py -q -p no:cacheprovider
uv run --no-sync python -B docs/research/quick-diameter-robustness-2026-09-15/benchmark.py --output /tmp/fdm-quick-robustness.json
uv run --no-sync python -B docs/research/quick-diameter-robustness-2026-09-15/benchmark.py --cross-stress --output /tmp/fdm-quick-cross-stress.json
uv run --no-sync python -B -m PyInstaller --noconfirm --clean --distpath /tmp/fdm-quick-frozen/dist --workpath /tmp/fdm-quick-frozen/build docs/research/quick-diameter-2026-09-15/frozen_probe.spec
```

Windows 的完整构建和安装后自检命令沿用[首轮记录](quick-diameter-implementation-2026-09-15.md#4-回归与复现)，当前源代码会执行修订号 2 的检查。

测量口径仍是掩膜边界像素中心间的代表距离；尚未加入原图灰度亚像素边缘修正，也未用真实样本验证标定后的误差。
