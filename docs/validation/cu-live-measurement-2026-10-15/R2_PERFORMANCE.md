# CU 实时试验 R2 正式性能报告

本报告只读取已经到达且符合本页版本命名的 JSON。缺失项明确列为待验收；不使用早期 `performance-*.json`，也不把早期无真实画布的一小时控制器测试算作最终 UI 联合验收。

**源码身份：**本页独立读取 `final-r2-{macos,windows}-*.json` 的六组新版短程结果及 `cu-live-workspace-hour-windows-final-r2.json`。首轮 [PERFORMANCE.md](PERFORMANCE.md) 与原 JSON 保留；新版结果不覆盖旧失败。三个尺寸的独立全图算法未改，单列引用 R1 的六份报告，明确是原构建证据，不伪称重新跑过 R2 全图。配置修正见 [overlay](configuration-rebuild-overlay.json)，各运行的实际源码／原生摘要以原始 JSON 为准。

**新版完整验收尚待所有分项、失败分母与一小时证据共同核对。** 逐项成功、失败与未完成以以下表格为准；脚本退出 0、快速原生内核或静止画面通过，均不能替代动态画面和联合负载验收。

## 固定口径与覆盖

硬门槛以 [采样前固定规则](performance-acceptance-rules.md) 为准。每个短程条件预热 5 次、正式尝试 30 次；不补测凑满成功数。成功延迟只以实际成功且绘制的样本计算，失败分母始终保留 30 次。预热拒收单列。`—` 表示没有有效样本或未测，不是零耗时。

短程 UI 的 `interval_ms=100` 是每次结果成功／失败后再等待 100 ms，属于串行连续测量，**不是固定每秒 10 条**。输入请求 30 Hz；不同帧绘制 FPS 与重复重绘次数分开。结束区间边界计数可能使短窗口 FPS 略高于 30，不代表请求速率提高。

表内延迟单位均为 ms，四元组依次为 **P50 / P95 / P99 / 最大值**。30 次中的 P99 仅为本批估计；失败越多，可用分位样本越少，不能解释为长期尾延迟保证。

GUI 发布计时覆盖 `_accept` 和同步 UI 信号，每次最大值须 ≤ 4 ms；不能用接受后重绘时间替代。端到端结果可见由实际 QTest 第二击计到包含该记录的 `paintEvent` 完成。

| 项目 | 固定门槛 |
|---|---|
| 第一击草稿可见 | P95 ≤ 16.7；P99 ≤ 33 |
| GUI 同步发布 | 每次 ≤ 4；报告所有超限次数 |
| 第二击结果可见 2K／4K | P95 ≤ 50；P99 ≤ 100 |
| 第二击结果可见 6K | P95 ≤ 100；P99 ≤ 150 |
| 预览 | 请求 30 Hz；实际不同帧绘制 ≥ 27 FPS |
| 换场可见清除 | 2K／4K P95 ≤ 150；6K P95 ≤ 250 |
| PNG＋后台全图相对影响 | 匹配无 PNG、无后台基线；结果可见 P95 增量 ≤ max(基线×20%, 10)，绝对门槛仍须通过 |
| 稳定输入中的拒收／超时 | 任何一次都记失败；不可删除失败后只比较成功延迟 |
| 真实 Qt 一小时 | ≥3600 s、≥10000 实际入账；输入30 Hz、尝试4 Hz、PNG开启、传统全图每10秒提交 |


| 正式报告 | 到达情况 |
|---|---|
| [final-r2-macos-manual-statistics.json](final-r2-macos-manual-statistics.json) | 待验收／尚未到达 |
| [final-r2-macos-motion.json](final-r2-macos-motion.json) | 待验收／尚未到达 |
| [final-macos-automatic-2048.json](final-macos-automatic-2048.json) | 已读取 |
| [final-macos-automatic-4096.json](final-macos-automatic-4096.json) | 已读取 |
| [final-macos-automatic-6144.json](final-macos-automatic-6144.json) | 已读取 |
| [final-r2-macos-ui-stable.json](final-r2-macos-ui-stable.json) | 待验收／尚未到达 |
| [final-r2-macos-ui-dynamic.json](final-r2-macos-ui-dynamic.json) | 待验收／尚未到达 |
| [final-r2-macos-ui-joint-stable.json](final-r2-macos-ui-joint-stable.json) | 待验收／尚未到达 |
| [final-r2-macos-ui-joint-dynamic.json](final-r2-macos-ui-joint-dynamic.json) | 待验收／尚未到达 |
| [final-r2-windows-manual-statistics.json](final-r2-windows-manual-statistics.json) | 待验收／尚未到达 |
| [final-r2-windows-motion.json](final-r2-windows-motion.json) | 待验收／尚未到达 |
| [final-windows-automatic-2048.json](final-windows-automatic-2048.json) | 已读取 |
| [final-windows-automatic-4096.json](final-windows-automatic-4096.json) | 已读取 |
| [final-windows-automatic-6144.json](final-windows-automatic-6144.json) | 已读取 |
| [final-r2-windows-ui-stable.json](final-r2-windows-ui-stable.json) | 待验收／尚未到达 |
| [final-r2-windows-ui-dynamic.json](final-r2-windows-ui-dynamic.json) | 待验收／尚未到达 |
| [final-r2-windows-ui-joint-stable.json](final-r2-windows-ui-joint-stable.json) | 待验收／尚未到达 |
| [final-r2-windows-ui-joint-dynamic.json](final-r2-windows-ui-joint-dynamic.json) | 待验收／尚未到达 |
| [cu-live-workspace-hour-windows-final-r2.json](cu-live-workspace-hour-windows-final-r2.json) | 待验收／尚未到达 |

覆盖边界：静止输入与逐帧唯一标记的低幅噪声动态输入分列；静止输入能命中像素相等缓存，不能代表相机每帧变化。短程联合为 PNG＋**持续至多一个 multiscale 任务**，一小时为 PNG＋**每10秒 traditional**；二者不是同等负载。短程 PNG＋持续 traditional、长期持续 multiscale、真实相机／采集卡与显微录像均不得由已有组合推定通过。

## 平台和资源口径

macOS UI 为真实 QWidget／QPainter／事件循环的 offscreen 渲染，不包含物理屏幕呈现。Windows UI 必须为 `qt_platform=windows`。本设备为 Parallels Windows ARM64、4 个虚拟 CPU、约16 GiB内存；x64 Python／原生扩展经仿真运行，不代表原生 Windows x64 用户设备。窗口尺寸、画布、DPI 按各 JSON 独立列出。

| 平台 | 系统 | Python | 依赖 | 基础版本来源（线程逐批另列） |
|---|---|---|---|---|
| macOS ARM64 | 正式环境待到达 | — | — | — |
| Windows ARM64／x64 仿真 | 正式环境待到达 | — | — | — |


| 平台 | 采样报告 | OpenCV请求 | 该脚本报告值 | 并行框架 | 该报告记录的环境变量 |
|---|---|---|---|---|---|
| macOS ARM64 | R2 [final-r2-macos-manual-statistics.json](final-r2-macos-manual-statistics.json) | 待到达 | — | — | — |
| macOS ARM64 | R2 [final-r2-macos-motion.json](final-r2-macos-motion.json) | 待到达 | — | — | — |
| macOS ARM64 | R2 [final-r2-macos-ui-stable.json](final-r2-macos-ui-stable.json) | 待到达 | — | — | — |
| macOS ARM64 | R2 [final-r2-macos-ui-dynamic.json](final-r2-macos-ui-dynamic.json) | 待到达 | — | — | — |
| macOS ARM64 | R2 [final-r2-macos-ui-joint-stable.json](final-r2-macos-ui-joint-stable.json) | 待到达 | — | — | — |
| macOS ARM64 | R2 [final-r2-macos-ui-joint-dynamic.json](final-r2-macos-ui-joint-dynamic.json) | 待到达 | — | — | — |
| macOS ARM64 | R1沿用 [final-macos-automatic-2048.json](final-macos-automatic-2048.json) | 1（脚本固定；JSON未单列请求字段） | 18 | 未记录 | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"} |
| macOS ARM64 | R1沿用 [final-macos-automatic-4096.json](final-macos-automatic-4096.json) | 1（脚本固定；JSON未单列请求字段） | 18 | 未记录 | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"} |
| macOS ARM64 | R1沿用 [final-macos-automatic-6144.json](final-macos-automatic-6144.json) | 1（脚本固定；JSON未单列请求字段） | 18 | 未记录 | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"} |
| Windows ARM64／x64 仿真 | R2 [final-r2-windows-manual-statistics.json](final-r2-windows-manual-statistics.json) | 待到达 | — | — | — |
| Windows ARM64／x64 仿真 | R2 [final-r2-windows-motion.json](final-r2-windows-motion.json) | 待到达 | — | — | — |
| Windows ARM64／x64 仿真 | R2 [final-r2-windows-ui-stable.json](final-r2-windows-ui-stable.json) | 待到达 | — | — | — |
| Windows ARM64／x64 仿真 | R2 [final-r2-windows-ui-dynamic.json](final-r2-windows-ui-dynamic.json) | 待到达 | — | — | — |
| Windows ARM64／x64 仿真 | R2 [final-r2-windows-ui-joint-stable.json](final-r2-windows-ui-joint-stable.json) | 待到达 | — | — | — |
| Windows ARM64／x64 仿真 | R2 [final-r2-windows-ui-joint-dynamic.json](final-r2-windows-ui-joint-dynamic.json) | 待到达 | — | — | — |
| Windows ARM64／x64 仿真 | R1沿用 [final-windows-automatic-2048.json](final-windows-automatic-2048.json) | 1（脚本固定；JSON未单列请求字段） | 1 | 未记录 | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"} |
| Windows ARM64／x64 仿真 | R1沿用 [final-windows-automatic-4096.json](final-windows-automatic-4096.json) | 1（脚本固定；JSON未单列请求字段） | 1 | 未记录 | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"} |
| Windows ARM64／x64 仿真 | R1沿用 [final-windows-automatic-6144.json](final-windows-automatic-6144.json) | 1（脚本固定；JSON未单列请求字段） | 1 | 未记录 | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"} |

UI／运动的默认 OpenCV 配置是这两类场景的主验收：`requested=null` 表示未主动调用线程限制，不能解释为1。手工／统计及独立全图使用另一采样脚本，其入口显式请求 `cv2.setNumThreads(1)`；旧JSON未单列请求字段，上表已区分代码依据与实际报告值。该脚本 SHA256 `ee3f3f8837b82ff2ac1712361c35cb1a83d46c7f16181c0f7703f7e44cbfa3fb` 与首轮905文件清单一致；新报告若改变脚本，应另核对该设置。

不能将某一行请求或报告值扩展到全场：首轮 Windows 手工／统计报告1，运动与四种UI报告默认4／Concurrency；首轮 macOS UI／运动为默认18／GCD，手工脚本即使请求1也报告18。报告线程数不等于实际同时运行的线程数；主进程环境记录也不能证明全图子进程采用同一个线程数。OMP／BLAS 等环境变量另列，不把其值1等同于OpenCV单线程。未到达的R2条件不沿用R1配置猜测。源码摘要、输入摘要、原生编译器／ABI 位于原始 JSON，不能只凭 Git HEAD 判断未提交工作树一致。

512 MiB 是去重后的保留帧及来源预留预算（含明确的 CPU／GPU 帧载荷），不是进程 RSS 上限。当前 ledger 不覆盖 `cv2.imencode` 的压缩输出缓冲、Qt／驱动全部分配，原计划的总缓冲预算尚未完整落实，不能因该账本低于512 MiB就称总内存已达标。RSS 另含 Python、Qt、算法、SDK、PNG 与子进程工作区；GPU 驱动不透明开销也不能由 RSS 完整覆盖。资源表按每个条件约100 ms采样的父／子／合计峰值列出，短于采样间隔的峰值可能遗漏；三者各自取峰，不要求父峰＋子峰等于同时观测到的合计峰。

## 手工原生服务与增量统计

这些是非 UI 分段计时：完整 BGR 快照复制、局部 ROI 服务及两者合计分别测量，不能替代第二击至结果可见门槛。已完成的原生手工样本要求固定已知宽度 19 px；复制阶段不产出测量状态。统计样本覆盖 1／100／1000／10000 条历史，追加和修改包含摘要与撤销；不包含界面表格或导出。此处没有另行设立事后提速阈值。

### macOS ARM64

待验收：正式报告尚未到达。

### Windows ARM64／x64 仿真

待验收：正式报告尚未到达。

## 运动核验分段

含缩小与运动比较，不含 GUI、捕获、分配和 PNG。基线关闭缓存并用原双精度归一化与 NumPy 峰值搜索；生产路径使用等价查表和 C++ 峰值扫描。结果差异数单列。P99／最大值从每行原始30个样本直接复算；分段未另设硬阈值，端到端门槛仍见 UI。

| 平台 | 宽×高 | 场景 | 样本／差异 | 基线 P50 / P95 / P99 / 最大 | 生产 P50 / P95 / P99 / 最大 |
|---|---|---|---|---|---|
| macOS ARM64 | 待验收 | — | — | — | — |
| Windows ARM64／x64 仿真 | 待验收 | — | — | — | — |

## 独立全图服务

所有行分别预热5次、正式30次；返回状态 `ok` 和 `empty` 都表示算法正常结束，空结果是按恢复规则未保留候选，不能当作运行错误。错误、超时、内存拒绝另列。worker 往返包含传递与等待，worker 内服务是诊断分段；相减不同分位数不能作为精确复制成本。30秒硬超时、2 GiB算法预算是资源保护，不是交互延迟验收目标。

**本节全部为 R1 原构建证据。** 全图算法代码未改，因此保留三个尺寸的原独立采样，不将其源码／原生摘要替换成 R2。新版端到端联合 UI 的全图负载另见本页 R2 条件；这两类证据不互相冒充。

### macOS ARM64

小图标准样本（每个大图批次单独重跑，不能合并成一批30样本）：

| 独立批次 | 输入 | 算法 | 状态／返回分母 | 往返 P50 / P95 / P99 / 最大 | 内服务 P50 / P95 / P99 / 最大 | 采样 RSS 父／子／合计 MiB | 父线程峰 |
|---|---|---|---|---|---|---|---|
| [2048批](final-macos-automatic-2048.json) | single8 320×240 | traditional | ok × 30 / 30；错误 0；候选 [1] | 3.152 / 3.181 / 3.238 / 3.260 | 1.941 / 2.109 / 2.123 / 2.129 | 71.1 / 97.2 / 168.3 | 3 |
| [2048批](final-macos-automatic-2048.json) | single8 320×240 | multiscale | ok × 30 / 30；错误 0；候选 [1] | 9.170 / 14.081 / 17.969 / 17.995 | 7.941 / 8.024 / 8.090 / 8.110 | 81.5 / 99.0 / 180.5 | 4 |
| [2048批](final-macos-automatic-2048.json) | single20 320×240 | traditional | ok × 30 / 30；错误 0；候选 [1] | 3.135 / 3.177 / 3.188 / 3.190 | 1.908 / 2.064 / 2.304 / 2.399 | 82.2 / 99.6 / 181.8 | 3 |
| [2048批](final-macos-automatic-2048.json) | single20 320×240 | multiscale | empty × 30 / 30；错误 0；候选 [0] | 10.215 / 11.584 / 13.589 / 14.111 | 8.954 / 9.190 / 9.239 / 9.255 | 82.5 / 99.6 / 182.1 | 4 |
| [2048批](final-macos-automatic-2048.json) | cross 640×426 | traditional | ok × 30 / 30；错误 0；候选 [2] | 7.711 / 11.454 / 14.591 / 14.983 | 6.738 / 7.044 / 7.085 / 7.098 | 84.4 / 137.6 / 222.0 | 4 |
| [2048批](final-macos-automatic-2048.json) | cross 640×426 | multiscale | empty × 30 / 30；错误 0；候选 [0] | 25.766 / 26.331 / 31.722 / 33.763 | 25.116 / 25.385 / 25.603 / 25.688 | 85.0 / 138.4 / 223.4 | 4 |
| [4096批](final-macos-automatic-4096.json) | single8 320×240 | traditional | ok × 30 / 30；错误 0；候选 [1] | 3.148 / 3.190 / 3.198 / 3.201 | 1.917 / 2.020 / 2.086 / 2.099 | 71.4 / 95.0 / 166.4 | 3 |
| [4096批](final-macos-automatic-4096.json) | single8 320×240 | multiscale | ok × 30 / 30；错误 0；候选 [1] | 9.185 / 14.311 / 18.214 / 18.457 | 8.062 / 8.634 / 8.795 / 8.848 | 83.6 / 99.4 / 183.0 | 4 |
| [4096批](final-macos-automatic-4096.json) | single20 320×240 | traditional | ok × 30 / 30；错误 0；候选 [1] | 3.146 / 3.184 / 3.192 / 3.196 | 1.900 / 2.009 / 2.056 / 2.073 | 83.6 / 101.5 / 185.1 | 3 |
| [4096批](final-macos-automatic-4096.json) | single20 320×240 | multiscale | empty × 30 / 30；错误 0；候选 [0] | 9.143 / 10.850 / 12.068 / 12.511 | 8.908 / 9.031 / 9.068 / 9.078 | 84.2 / 101.5 / 185.7 | 4 |
| [4096批](final-macos-automatic-4096.json) | cross 640×426 | traditional | ok × 30 / 30；错误 0；候选 [2] | 7.715 / 9.532 / 11.867 / 12.232 | 6.820 / 7.075 / 7.122 / 7.134 | 85.6 / 137.8 / 223.5 | 4 |
| [4096批](final-macos-automatic-4096.json) | cross 640×426 | multiscale | empty × 30 / 30；错误 0；候选 [0] | 25.813 / 30.991 / 34.007 / 34.336 | 25.031 / 25.251 / 25.271 / 25.272 | 85.7 / 138.7 / 224.3 | 4 |
| [6144批](final-macos-automatic-6144.json) | single8 320×240 | traditional | ok × 30 / 30；错误 0；候选 [1] | 3.115 / 3.177 / 3.182 / 3.184 | 1.936 / 2.121 / 2.166 / 2.177 | 71.3 / 99.1 / 170.5 | 3 |
| [6144批](final-macos-automatic-6144.json) | single8 320×240 | multiscale | ok × 30 / 30；错误 0；候选 [1] | 9.176 / 13.793 / 17.438 / 17.613 | 7.965 / 8.174 / 8.326 / 8.375 | 82.8 / 100.9 / 183.6 | 4 |
| [6144批](final-macos-automatic-6144.json) | single20 320×240 | traditional | ok × 30 / 30；错误 0；候选 [1] | 3.114 / 3.179 / 3.185 / 3.187 | 1.897 / 1.986 / 2.109 / 2.156 | 83.4 / 102.6 / 186.0 | 3 |
| [6144批](final-macos-automatic-6144.json) | single20 320×240 | multiscale | empty × 30 / 30；错误 0；候选 [0] | 10.663 / 14.694 / 18.980 / 20.334 | 9.106 / 12.126 / 12.172 / 12.176 | 85.7 / 118.5 / 204.2 | 4 |
| [6144批](final-macos-automatic-6144.json) | cross 640×426 | traditional | ok × 30 / 30；错误 0；候选 [2] | 7.703 / 9.804 / 13.701 / 14.610 | 6.766 / 7.185 / 7.226 / 7.239 | 87.1 / 148.3 / 235.3 | 4 |
| [6144批](final-macos-automatic-6144.json) | cross 640×426 | multiscale | empty × 30 / 30；错误 0；候选 [0] | 25.784 / 26.816 / 28.087 / 28.580 | 24.807 / 25.242 / 25.417 / 25.473 | 87.1 / 153.0 / 240.1 | 4 |

大图固定八根样本：

| 独立批次 | 输入 | 算法 | 状态／返回分母 | 往返 P50 / P95 / P99 / 最大 | 内服务 P50 / P95 / P99 / 最大 | 采样 RSS 父／子／合计 MiB | 父线程峰 |
|---|---|---|---|---|---|---|---|
| [2048批](final-macos-automatic-2048.json) | large_sparse8 2048×1365 | traditional | ok × 30 / 30；错误 0；候选 [8] | 252.935 / 274.451 / 287.540 / 288.948 | 250.553 / 272.044 / 284.913 / 286.329 | 113.9 / 440.5 / 554.4 | 4 |
| [2048批](final-macos-automatic-2048.json) | large_sparse8 2048×1365 | multiscale | ok × 30 / 30；错误 0；候选 [8] | 285.850 / 325.905 / 362.773 / 376.251 | 265.120 / 272.762 / 294.439 / 302.751 | 644.2 / 629.7 / 1266.0 | 4 |
| [4096批](final-macos-automatic-4096.json) | large_sparse8 4096×2730 | traditional | ok × 30 / 30；错误 0；候选 [8] | 1446.639 / 1497.995 / 1823.955 / 1949.804 | 1442.903 / 1493.911 / 1819.738 / 1945.761 | 162.8 / 559.8 / 721.4 | 4 |
| [4096批](final-macos-automatic-4096.json) | large_sparse8 4096×2730 | multiscale | ok × 30 / 30；错误 0；候选 [8] | 928.837 / 1032.177 / 1111.603 / 1129.209 | 882.867 / 911.306 / 932.273 / 937.229 | 1219.1 / 1020.8 / 2237.6 | 4 |
| [6144批](final-macos-automatic-6144.json) | large_sparse8 6144×4096 | traditional | ok × 30 / 30；错误 0；候选 [8] | 4071.747 / 4119.200 / 4137.250 / 4143.342 | 4064.546 / 4111.234 / 4129.218 / 4135.356 | 246.8 / 1042.6 / 1288.2 | 4 |
| [6144批](final-macos-automatic-6144.json) | large_sparse8 6144×4096 | multiscale | ok × 30 / 30；错误 0；候选 [8] | 1958.850 / 2094.931 / 2244.040 / 2281.507 | 1880.485 / 1916.342 / 1933.800 / 1939.909 | 1833.8 / 1739.0 / 3572.7 | 4 |

### Windows ARM64／x64 仿真

小图标准样本（每个大图批次单独重跑，不能合并成一批30样本）：

| 独立批次 | 输入 | 算法 | 状态／返回分母 | 往返 P50 / P95 / P99 / 最大 | 内服务 P50 / P95 / P99 / 最大 | 采样 RSS 父／子／合计 MiB | 父线程峰 |
|---|---|---|---|---|---|---|---|
| [2048批](final-windows-automatic-2048.json) | single8 320×240 | traditional | ok × 30 / 30；错误 0；候选 [1] | 11.678 / 17.028 / 20.505 / 21.803 | 8.641 / 9.198 / 9.288 / 9.321 | 64.8 / — / — | None |
| [2048批](final-windows-automatic-2048.json) | single8 320×240 | multiscale | ok × 30 / 30；错误 0；候选 [1] | 35.424 / 41.722 / 44.854 / 46.124 | 32.971 / 33.694 / 34.127 / 34.224 | 74.7 / — / — | None |
| [2048批](final-windows-automatic-2048.json) | single20 320×240 | traditional | ok × 30 / 30；错误 0；候选 [1] | 12.644 / 19.691 / 21.739 / 21.772 | 8.137 / 8.344 / 8.556 / 8.642 | 75.2 / — / — | None |
| [2048批](final-windows-automatic-2048.json) | single20 320×240 | multiscale | empty × 30 / 30；错误 0；候选 [0] | 41.641 / 49.583 / 50.325 / 50.366 | 38.082 / 38.491 / 38.686 / 38.763 | 75.2 / — / — | None |
| [2048批](final-windows-automatic-2048.json) | cross 640×426 | traditional | ok × 30 / 30；错误 0；候选 [2] | 33.346 / 38.000 / 38.162 / 38.170 | 29.305 / 29.675 / 29.842 / 29.908 | 76.6 / — / — | None |
| [2048批](final-windows-automatic-2048.json) | cross 640×426 | multiscale | empty × 30 / 30；错误 0；候选 [0] | 106.103 / 109.898 / 111.543 / 112.111 | 99.898 / 100.686 / 101.429 / 101.711 | 76.6 / — / — | None |
| [4096批](final-windows-automatic-4096.json) | single8 320×240 | traditional | ok × 30 / 30；错误 0；候选 [1] | 13.110 / 16.712 / 19.798 / 21.058 | 8.094 / 9.013 / 9.167 / 9.220 | 65.2 / — / — | None |
| [4096批](final-windows-automatic-4096.json) | single8 320×240 | multiscale | ok × 30 / 30；错误 0；候选 [1] | 34.808 / 40.945 / 41.869 / 42.170 | 31.652 / 32.134 / 32.224 / 32.252 | 74.6 / — / — | None |
| [4096批](final-windows-automatic-4096.json) | single20 320×240 | traditional | ok × 30 / 30；错误 0；候选 [1] | 14.109 / 21.478 / 21.611 / 21.637 | 7.946 / 8.169 / 8.348 / 8.419 | 75.4 / — / — | None |
| [4096批](final-windows-automatic-4096.json) | single20 320×240 | multiscale | empty × 30 / 30；错误 0；候选 [0] | 41.708 / 48.696 / 50.183 / 50.474 | 36.410 / 36.908 / 36.952 / 36.963 | 75.4 / — / — | None |
| [4096批](final-windows-automatic-4096.json) | cross 640×426 | traditional | ok × 30 / 30；错误 0；候选 [2] | 31.334 / 37.365 / 37.443 / 37.472 | 28.512 / 28.856 / 29.194 / 29.325 | 76.7 / — / — | None |
| [4096批](final-windows-automatic-4096.json) | cross 640×426 | multiscale | empty × 30 / 30；错误 0；候选 [0] | 102.933 / 108.714 / 110.351 / 110.386 | 96.708 / 97.619 / 97.678 / 97.693 | 76.8 / — / — | None |
| [6144批](final-windows-automatic-6144.json) | single8 320×240 | traditional | ok × 30 / 30；错误 0；候选 [1] | 10.700 / 16.671 / 16.791 / 16.826 | 8.175 / 9.170 / 10.059 / 10.379 | 64.7 / — / — | None |
| [6144批](final-windows-automatic-6144.json) | single8 320×240 | multiscale | ok × 30 / 30；错误 0；候选 [1] | 37.092 / 42.446 / 42.681 / 42.759 | 31.750 / 32.310 / 32.655 / 32.764 | 74.6 / — / — | None |
| [6144批](final-windows-automatic-6144.json) | single20 320×240 | traditional | ok × 30 / 30；错误 0；候选 [1] | 12.406 / 16.876 / 17.861 / 18.201 | 7.903 / 8.192 / 8.212 / 8.217 | 75.1 / — / — | None |
| [6144批](final-windows-automatic-6144.json) | single20 320×240 | multiscale | empty × 30 / 30；错误 0；候选 [0] | 41.329 / 48.663 / 50.243 / 50.350 | 36.462 / 37.285 / 37.872 / 38.051 | 75.4 / — / — | None |
| [6144批](final-windows-automatic-6144.json) | cross 640×426 | traditional | ok × 30 / 30；错误 0；候选 [2] | 33.308 / 37.223 / 37.330 / 37.364 | 28.247 / 28.919 / 29.062 / 29.083 | 76.7 / — / — | None |
| [6144批](final-windows-automatic-6144.json) | cross 640×426 | multiscale | empty × 30 / 30；错误 0；候选 [0] | 102.855 / 111.311 / 111.795 / 111.933 | 96.979 / 99.233 / 99.761 / 99.797 | 76.7 / — / — | None |

大图固定八根样本：

| 独立批次 | 输入 | 算法 | 状态／返回分母 | 往返 P50 / P95 / P99 / 最大 | 内服务 P50 / P95 / P99 / 最大 | 采样 RSS 父／子／合计 MiB | 父线程峰 |
|---|---|---|---|---|---|---|---|
| [2048批](final-windows-automatic-2048.json) | large_sparse8 2048×1365 | traditional | ok × 30 / 30；错误 0；候选 [8] | 1208.263 / 1219.102 / 1225.951 / 1228.270 | 1201.103 / 1212.327 / 1219.466 / 1221.830 | 105.4 / — / — | None |
| [2048批](final-windows-automatic-2048.json) | large_sparse8 2048×1365 | multiscale | ok × 30 / 30；错误 0；候选 [8] | 1021.499 / 1062.120 / 1092.495 / 1101.507 | 967.596 / 975.986 / 989.864 / 995.463 | 609.7 / — / — | None |
| [4096批](final-windows-automatic-4096.json) | large_sparse8 4096×2730 | traditional | ok × 30 / 30；错误 0；候选 [8] | 7298.491 / 7418.404 / 7448.290 / 7457.932 | 7278.697 / 7397.109 / 7425.718 / 7434.083 | 164.0 / — / — | None |
| [4096批](final-windows-automatic-4096.json) | large_sparse8 4096×2730 | multiscale | ok × 30 / 30；错误 0；候选 [8] | 3680.162 / 3763.572 / 3847.788 / 3879.900 | 3563.484 / 3644.794 / 3730.025 / 3763.263 | 1175.5 / — / — | None |
| [6144批](final-windows-automatic-6144.json) | large_sparse8 6144×4096 | traditional | ok × 30 / 30；错误 0；候选 [8] | 22330.265 / 22589.019 / 22706.137 / 22736.456 | 22287.113 / 22546.949 / 22667.893 / 22697.347 | 275.0 / — / — | None |
| [6144批](final-windows-automatic-6144.json) | large_sparse8 6144×4096 | multiscale | ok × 30 / 30；错误 0；候选 [8] | 8638.740 / 8785.018 / 9182.059 / 9323.931 | 8437.239 / 8521.449 / 8602.361 / 8634.153 | 1826.7 / — / — | None |

## 真实 Qt 短程交互

以下只用正式30次；成功须既接受又在画布绘制。拒收／未完成与成功分位数并列，预热拒收不混入正式分母。`PASS` 仅适用于本行固定合成条件；静止／动态、PNG开关与后台条件不可互相替代。

### macOS ARM64

待验收：正式 UI 报告尚未到达。

### Windows ARM64／x64 仿真

待验收：正式 UI 报告尚未到达。

### PNG＋持续 multiscale 的相对影响

仅与同平台、尺寸、输入、窗口／画布／DPI、源码和环境一致的 PNG关闭／无后台基线比较。相对P95通过不代表绝对门槛通过；任何一边有失败，禁止用成功样本比较宣称通过。

| 报告 | 宽 | 后台返回状态 | 相对判定 | 基线→联合P95 | 增量／允许增量 | P99增量（诊断） | 说明 |
|---|---|---|---|---|---|---|---|
| 待验收 | — | — | — | — | — | — | — |


| 报告／宽 | 预热正常完成 | 正式正常完成（旧版未分期） | 收尾正常完成 | 阶段响应数 |
|---|---|---|---|---|
| 待验收 | — | — | — | — |


| 报告／宽 | 取消总数 | 显式收尾取消 | 推断收尾取消 | 未归因取消 | 后台错误 | 阶段轨迹有效 |
|---|---|---|---|---|---|---|
| 待验收 | — | — | — | — | — | — |

后台 `empty` 是正常无候选返回。`cancelled` 是未完成任务，始终保留计数，不改成成功。R2 的预热结果单独存储；每条响应记录 `phase`、`task_id` 和接收单调时间，审计先核验阶段边界、时间顺序及任务 ID，再核对收尾取消是否属于明确请求关闭的任务。只有有效轨迹匹配时才列为**显式收尾取消**；正式正常完成只计 formal，预热与收尾正常返回另列。轨迹缺项／冲突不得退回旧版推断来豁免失败。

R1 无阶段记录，仍只在报告中harness源码SHA与已核查版本完全一致、整批唯一取消且为末项、显式 `error:null` 与 `candidates:0`、此前至少一次正常返回时，单列为**源码约束下推断的收尾取消**。已核查旧代码在 `finally` 关闭后台worker，后续 `wait_for` 仍可能向返回对象共享的列表追加取消；这不是旧JSON的运行时阶段证明。R1 的正常完成数没有预热／正式／收尾分期，不能冒充 R2 formal 数。所有未归因取消仍失败；两种归因都不改变手工测量失败、超时或背压分母。

## 一小时真实工作区联合运行

只接收本页版本加载清单中包含 `counters` 和 `latency_by_fifteen_minutes` 的真实 Qt 报告；R2 文件固定为 `cu-live-workspace-hour-windows-final-r2.json`。早期 `soak-one-hour-macos.json` 不进入本节。每15分钟独立看尾延迟与发布最大值；至少10000条实际入账、零串场、提交／接受／终结守恒、PNG摘要／像素、账本往返、撤销历史和关闭资源均是门槛。
**待验收：最终真实 Qt 一小时报告尚未到达。** 不以早期控制器测试或短程通过代替。

## 未验收事项与复算

缺失的平台／尺寸／负载组合维持待验收；单独内核、静止相等缓存、正常空候选和安全拒收有不同含义。不能通过降低运动阈值、放宽最新已显示帧核验、停止实时画面，或剔除失败样本换取通过。真实相机／采集卡、真实CU画面、曝光变化／失焦／振动和物理计量均需独立验证。

本页由同目录 `summarize_final_performance.py` 仅读取已有JSON生成，UI／一小时判定复用 `scripts/validation/audit_cu_live_performance.py`；运行聚合不会启动测试、Qt、基准、子进程或网络。新正式文件到达后重新生成，并保留所有失败。

```sh
python docs/validation/cu-live-measurement-2026-10-15/summarize_final_performance.py --revision r2
```

输入文件摘要（保证本页所引用的是这一批结果）：

| 文件 | SHA256 |
|---|---|
| [final-macos-automatic-2048.json](final-macos-automatic-2048.json) | 68437aecaf4c7df3bb6076ca50ccffc9afda64c975682ce73ba2d9565cdc0e1e |
| [final-macos-automatic-4096.json](final-macos-automatic-4096.json) | f86711030abb24fe244228b2c19801ce287103d72d01df639c55eba270c6ad1b |
| [final-macos-automatic-6144.json](final-macos-automatic-6144.json) | ab274915bb3d981cfa931e76b36d4b7866445bb4f0653b1dab794f15df03bc3f |
| [final-windows-automatic-2048.json](final-windows-automatic-2048.json) | 0ff2283534ef12518ba33005210060877bbcfdce87f2fdf3050f0c812153d0fb |
| [final-windows-automatic-4096.json](final-windows-automatic-4096.json) | 7cf8b489a943eeddef4825e3dfcf14ca77f9c24588b35571c07e2d719a513960 |
| [final-windows-automatic-6144.json](final-windows-automatic-6144.json) | aca6ed42bbc3b273d2a7f1aa1e52eaa82aaf2fabf3780fd685fab4a7c4e70a33 |

