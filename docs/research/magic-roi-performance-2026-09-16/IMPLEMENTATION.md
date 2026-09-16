# 魔棒 ROI 优化实施与验收记录

日期：2026-09-16。旧版基线：`d9a0dd7`。新版：当前工作区。本轮未更换模型、降低 1024 输入精度、量化、调整 CPU 线程或引入 GPU 依赖。

## 已实现

1. **分段计时**：记录来源、会话、请求编号、模型、实际裁剪框、缓存命中/未命中、编码/解码次数、扩框次数与停止原因。分开统计来源准备、排队、session 初始化、裁剪/RGB 转换、编码准备、编码、解码、掩膜恢复、几何处理、UI 投递和结果应用。总耗时达到 100 ms 时，沿用运行日志输出一条 `Magic segmentation request` JSON 汇总；预热初始化另记一次。统计信息在主窗口应用结果前移除，不写入测量数据。
2. **扩框去重**：每次普通魔棒请求最多处理 4 个不同区域。相同实际裁剪框采用同一缓存键，完整约束区域算过即返回，取消重复的 `|fallback` 编码；空结果也不重复处理完整区域。保留原扩框倍率、面积阈值、触边接受规则、剔除主体约束和数字切片有效覆盖裁剪。
3. **稳定工作区**：添加、剔除各自保存工作区。来源、模型、约束及增强配置相同且有效提示点仍在其中时复用特征；越界时选取包含全部有效提示点的后续候选框。来源版本改变会失效，切换图像、焦层、提交或取消清理草稿。模型切换保留可编辑草稿及采样点，清理工作区并使旧请求失效。
4. **有界缓存**：每个 ONNX 服务按 embedding 实际 `nbytes` 管理 32 MiB LRU；不再限定 ROI 最多 4 项，非 ROI 仍保持原条数限制。当前模型单个 embedding 为 4 MiB。键包含来源版本、模型、预处理版本、目标输入尺寸和裁剪框，点提示不参与特征键。
5. **按需准备图像**：先选框和查询缓存；命中时不裁剪、不转换像素。未命中只对实际 ROI 进行 QImage 裁剪及 RGB 转换。普通图片的 SHA-256 按文档和 QImage 版本复用，像素变化或关闭文档后失效。
6. **异步初始化**：进入普通魔棒时通过现有工作线程准备当前模型的 encoder/decoder session；成功后重复进入不重新初始化，不进行预编码，不增加推理线程。预热失败记录日志，实际请求仍走原错误提示。
7. **打包门槛**：`--self-check --json` 增加 `functional_checks.magic_segmentation`；发行配置排除该功能时明确跳过。启用时用随包两套 ONNX 模型执行真实 ROI 推理、复用缓存、比较掩膜，并确认 CPU 运行时。Windows 构建脚本拒绝缺失、失败、重复编码或运行时不完整的自检结果；原快速测径和画布自检继续保留。

## 根据回放增加的边界保护

旧规则只在触及至少两条裁剪边或面积占比过大时继续扩框。因此，“只触及一条内部裁剪边”的结果虽可返回，却不适合固定为后续工作区：补点后保留旧框可能继续截断目标。

本轮没有更改这条分割接受规则，而是增加**工作区复用资格**：内部裁剪边仍截断目标时，返回现有结果，下次继续采用原来的选框方式；已经覆盖完整约束范围，或仅接触原图/约束本身的外边界时仍可复用。这类边缘补点优先保持原边界行为，不承诺零次新增编码。

初版回放暴露的问题和数值保留在 [quality-before-boundary-guard.json](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/magic-roi-performance-2026-09-16/implementation-runs/quality-before-boundary-guard.json)。最终回放见下文。工作区策略会改变裁剪上下文，不能把“相同实际输入逐像素一致”解读为“所有不同裁剪上下文输出都一致”。

## 同机性能对照

环境：Apple M5 Pro、18 核、macOS 26.6.2 ARM64、ONNX Runtime 1.24.4、CPUExecutionProvider、EdgeSAM-3x。旧版与新版分别在独立进程串行执行，每个稳定场景 30 次。先完成模型初始化和两次预运行，随后计时；RSS 采样、图片生成及掩膜保存不计入服务耗时，基准期间不并行执行其他测试或构建。

测试范围为 `PromptSegmentationService.predict_polygon`，不包括主窗口准备来源、Qt 排队、GUI 绘制和整机冷启动。新增加的生产日志覆盖这些阶段，目标 Windows 电脑需用实际操作日志复核端到端体验。合成图用于可重复计时；显微图片用于额外效果回放。

| 场景（各 30 次） | 旧 P50 / P95 ms | 新 P50 / P95 ms | 旧 / 新总编码次数 | 旧 / 新最大缓存 MiB |
|---|---:|---:|---:|---:|
| 全新 ROI（2048×1536） | 103.00 / 131.58 | 100.48 / 105.91 | 30 / 30 | 16 / 32 |
| 相同 ROI 重复点击 | 16.60 / 17.14 | 14.91 / 17.89 | 0 / 0 | 4 / 4 |
| 工作区内连续正/负补点 | 103.09 / 105.74 | 14.69 / 16.30 | 30 / 0 | 16 / 4 |
| 长条目标重复扩框 | 494.78 / 576.01 | 28.18 / 29.10 | 150 / 0 | 16 / 16 |
| 贴原图边缘目标重复点击 | 17.06 / 17.84 | 15.22 / 16.28 | 0 / 0 | 4 / 4 |
| 大图新 ROI（6000×4000） | 115.07 / 117.09 | 102.02 / 104.11 | 30 / 30 | 16 / 32 |

| 场景 | 旧 / 新采样 RSS 最大值 MiB |
|---|---:|
| 全新 ROI（2048×1536） | 820.4 / 768.8 |
| 相同 ROI 重复点击 | 847.6 / 789.5 |
| 工作区内连续正/负补点 | 856.9 / 790.3 |
| 长条目标重复扩框 | 989.5 / 896.8 |
| 贴原图边缘目标重复点击 | 989.6 / 909.3 |
| 大图新 ROI（6000×4000） | 1218.5 / 1099.5 |

首次长条目标另外记录编码次数：旧版 5 次，新版 4 次；完整区域只计算一次。相同 ROI 的旧版本就能命中，因此该场景只预期节省图像准备开销。

本机稳定复核中，连续补点 P50 降低 **85.8%**，重复扩框 P50 降低 **94.3%**，均达到 50% 目标；全新 ROI P95 变化 **-19.5%**，大图新 ROI P95 变化 **-11.1%**，均未超过允许的 +10%。首次长条请求单次记录为 489.59 → 364.15 ms，此项只有一次，不作为 P50/P95 结论。

计时过程曾出现一轮明显的时序波动：基线新 ROI P50 达 806 ms，随后连续补点又回到 95 ms。该轮全部原始数据保留在 [variable-timing-run/paired.json](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/magic-roi-performance-2026-09-16/implementation-runs/variable-timing-run/paired.json)，不据此声称有数量级的首次编码提升。上表采用在停止其他测试和构建后，以 **新版先运行、旧版后运行** 的独立进程顺序重新执行的完整 30 次复核，原始样本未删点。全新小 ROI 的核心编码成本仍约 100 ms，优化收益主要来自少编码与不重复准备整图。

逐次耗时、编码次数、裁剪框、缓存占用和进程驻留内存保存在 [paired.json](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/magic-roi-performance-2026-09-16/implementation-runs/paired.json)。缓存预算只限制保存的 embedding；模型 session、临时输入/输出、QImage、几何数据和进程分配器不属于这 32 MiB。RSS 是逐请求采样最大值，不是整个进程瞬时峰值。

## 效果与回归

- 固定实际裁剪框与提示点：圆形、带孔目标、边缘目标、显微碎片的原始模型掩膜、最终几何选定掩膜、面积及轮廓哈希逐项比较。
- 稳定工作区回放：合成圆、带孔目标、显微碎片、纤维、误点背景间隙，各回放首次点击和两次补点。背景间隙样本用于暴露选框行为，不作为正确识别目标的证据。
- 显微图来源为本地既有 `.tmp/fiberseg/image/58.jpg` 第三方示例；没有人工标注真值，也不是用户生产样本。回放可检查形状、孔洞和边界变化，不能代替实际计量精度验收。
- 覆盖同框补正/负点、扩框去重、完整约束、空结果、工作区越界、LRU、像素转换一致、来源哈希版本、模型切换、过期会话/请求取消、主窗口到工作线程的工作区传递、真实模型自检和构建拒绝条件。共享服务的快速测径、小目标剔除、坐标转换、覆盖裁剪、孔洞处理采用既有测试回归。

**固定输入检查全部逐像素一致，最终面积和轮廓哈希也一致。** 最终工作区回放结果如下；IoU 为新旧掩膜交集/并集，反映输出差异，并非相对人工真值的精度。

| 回放样本 | 3 次请求的最小 IoU | 主体/孔洞与边界检查 |
|---|---:|---|
| 合成圆 | 0.997158 | 主体一致，轮廓有少量像素差异 |
| 带孔目标 | 0.992017 | 保留内孔，轮廓仅有少量像素差异 |
| 显微碎片 | 0.974242 | 主体一致，轮廓有少量像素差异 |
| 误点背景间隙 | 1.000000 | 3 次均逐像素一致，沿用原选框及孔洞结果 |
| 显微纤维 | 1.000000 | 3 次均逐像素一致，沿用原选框及孔洞结果 |

两版本各执行 30 次启动前取消请求，均没有编码/解码；工作线程另有在编码后取消、同编号请求被新 generation 替换的功能测试，确认不继续解码、不把旧结果应用到新草稿。取消在 ONNX 单次调用结束后的检查点生效，不强行中断正在运行的原生调用。

[轮廓回放对照图](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/magic-roi-performance-2026-09-16/implementation-runs/microscopy-replay.png)：绿色是各自的轮廓，青色表示仅旧版包含的像素，紫色表示仅新版包含的像素。

验证记录：

- 最终相关服务、坐标、几何、设置与构建测试：**211 passed、232 subtests passed**。
- 最终 UI 回归（魔棒、快速测径、数字切片及相关设置）：**80 passed、210 deselected**。
- 补齐排队取消日志后再验 worker、坐标和新增 ROI 测试：**55 passed**；这批与前述测试有重叠，不能相加为独立测试总数。新增 ROI 测试文件最终包含 35 个参数化测试用例。
- 全量测试一次：**2901 passed、1569 subtests passed、1 failed**。发生在最终边界保护及日志收尾前，后续变更由上述专项回归覆盖。唯一失败为 `tests/test_atomic_io.py::test_production_json_serialization_explicitly_rejects_non_finite_values`；发现的 6 处缺少 `allow_nan=False` 调用全部位于未修改的轮廓比较文件，已逐文件确认与 HEAD 相同。证据见 [JSON 契约审计](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/magic-roi-performance-2026-09-16/implementation-runs/json-contract-audit.json)。本次新增生产 JSON 日志满足该约束。
- `git diff --check` 通过。测试原始记录：[全量](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/magic-roi-performance-2026-09-16/implementation-runs/full-pytest.log)、[相关服务与构建](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/magic-roi-performance-2026-09-16/implementation-runs/related-pytest.log)、[UI](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/magic-roi-performance-2026-09-16/implementation-runs/ui-pytest.log)、[日志收尾](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/magic-roi-performance-2026-09-16/implementation-runs/worker-pytest.log)。

## 打包自检记录

- 源码环境：两组实际发布模型自检通过。
- macOS 原生 PyInstaller 冻结探针：沿用生产 hooks 和 ONNX 动态库收集策略，把两组模型打入包内；在 `stdin/stdout/stderr=None` 的条件下完成推理及快速测径自检。
- 两个模型首次编码均为 1，重复编码均为 0，重复解码均为 1，掩膜逐像素一致，实际后端为 CPUExecutionProvider。
- ONNX Runtime 扩展和 skeleton Cython 扩展均从冻结包中加载。快速测径仍为 `skimage_zhang`、`geometry_revision=3`，包含延长/收缩及不完整边界拒绝检查。
- 原生探针不是完整 FDM 安装包，未对它声称已完成画布渲染的冻结自检；生产构建脚本仍强制要求原有画布自检通过。
- 记录：[frozen-native.json](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/magic-roi-performance-2026-09-16/implementation-runs/frozen-native.json)、[构建记录](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/magic-roi-performance-2026-09-16/implementation-runs/build-record.json)。

**待 Windows 实机验收**：完整 onedir 构建及其 `--self-check --json`；安装包制作和安装后再自检；冷启动、新 ROI、连续补点、扩框、取消、切图/切焦层/切模型；目标电脑上每场景至少 30 次的 P50/P95、编码次数与内存记录。当前没有目标 Windows 机器，不将 macOS 的毫秒数、原生冻结探针或合成图结果当作 Windows 验收结论。

## 复现

在项目根目录运行；不修改项目依赖或模型资产：

```sh
QT_QPA_PLATFORM=offscreen uv run --no-sync python docs/research/magic-roi-performance-2026-09-16/paired_performance.py --order current-first
uv run --no-sync --with matplotlib python docs/research/magic-roi-performance-2026-09-16/render_replay.py
uv run --no-sync pyinstaller --noconfirm --clean --distpath .tmp/magic-roi-frozen/dist --workpath .tmp/magic-roi-frozen/build docs/research/magic-roi-performance-2026-09-16/frozen_probe.spec
QT_QPA_PLATFORM=offscreen .tmp/magic-roi-frozen/dist/fdm-magic-roi-probe/fdm-magic-roi-probe
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q tests/test_magic_roi_performance.py
```

`paired_performance.py` 从指定 Git 基线提取旧服务，模型和环境保持相同，输出逐次数据及压缩掩膜。`--replay-only` 可只更新效果回放并保留已有计时。第三方显微示例不存在时会跳过对应回放，不会下载其他图片替代。
