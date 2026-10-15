# CU 实时测量性能验收规则（采样前固定）

本规则对应 `scripts/benchmark_cu_live_ui.py` 与 `scripts/validation/soak_cu_workspace.py` 的最终指标，固定判定口径，不代表当前已经达标。正确性、零串场和数据完整性是先决条件；不能通过修改运动阈值、放宽最新已显示帧核验、停止主画面或只挑成功样本取得性能通过。

| 验收项 | 判定指标与固定门槛 |
|---|---|
| 第一击草稿可见 | UI benchmark 的 `first_click_to_draft_paint_ms`；soak 的 `first_click_to_draft_paint`。P95 ≤ 16.7 ms，P99 ≤ 33 ms。 |
| GUI 同步发布预算 | `gui_publish_ms`：原 `_accept` 调用前后计时，包含同步 UI 信号处理；每次预算 ≤ 4 ms，同时报告 P50/P95/P99/最大值和超预算情况。不得用 `accept_to_record_paint` 代替。 |
| 第二击结果可见：2K／4K | `second_click_to_record_paint_ms`／`second_click_to_record_paint`。P95 ≤ 50 ms，P99 ≤ 100 ms。 |
| 第二击结果可见：6K | 同上。P95 ≤ 100 ms，P99 ≤ 150 ms。 |
| 预览 | 请求 30 Hz，报告实际生成 FPS；实际不同帧绘制 FPS ≥ 27。UI benchmark 使用 `painted_frames_per_second`，soak 使用 `distinct_painted_fps`，不得用重复重绘次数代替。 |
| 换场可见清除 | soak 的 `scene_input_to_clear_paint`：实际输入新场景至旧场结果清除后的 QPainter 完成。2K／4K P95 ≤ 150 ms，6K P95 ≤ 250 ms；同时报告 `scene_input_to_clear` 和检测／清场分段耗时。 |
| PNG＋后台全图 | 同尺寸、输入、环境下，以关闭 PNG 且无后台全图为基线；开启 PNG＋后台全图后，结果可见 P95 增量 ≤ `max(基线 P95 × 20%, 10 ms)`，P95、P99 均仍须通过以上绝对门槛。P99 增幅另报诊断，不作为额外相对硬门槛。 |

结果可见的起点是实际 QTest 第二击，终点是包含该记录的真实 `paintEvent` 完成；`second_click_to_accept`、计算完成至接受、工作队列等待、`accept_to_record_paint` 用于解释瓶颈，不能替代端到端门槛。GUI 同步发布包含脚本连接的同步观察回调，属于保守测量；它不包含之后等待 Qt 重绘的时间。最大值超过 4 ms 必须记为发布预算超限，不能仅以平均值或分位数掩盖。

**正式短程采样。** 每个尺寸、输入、负载条件预热 5 次、正式尝试 30 次；预热不进入正式延迟分布，但其错误和拒收必须披露。不得不断补测直到凑够 30 个成功结果。保存每次 trial、输入摘要、源码／原生构建摘要、Python／NumPy／OpenCV／Qt 版本、操作系统／CPU、窗口尺寸／DPI、Qt 平台以及线程配置。正式性能与构建、回归和其它性能采样串行执行。30 次的 P99 是本批样本的分位估计，必须同时给出样本数及最大值，不将其描述为长期尾延迟保证。

至少分别报告静止输入与 `--motion-input lsb-noise`。后者是两个固定低幅噪声状态轮换，并给每帧添加能穿过三倍缩小的 16 bit 唯一标记，避免采样恰好跨过偶数帧而命中相同像素缓存；它只证明这个固定合成动态输入的行为。静止图命中逐像素相等缓存是合法快路径，但不能代表每帧变化的摄像流。PNG 开／关、后台 traditional 与 multiscale 的结果分别列出；缺少某一组合时标为未验收，不由另一组合推断。短程脚本的后台全图持续保持至多一个任务在运行，其强度与 soak 每十秒提交一次不同，不混作同一负载。

新版短程报告为后台响应记录 `phase=warmup|formal|teardown`、`received_monotonic` 和 `task_id`，并记录阶段边界与关闭时的任务 ID；时间表示 GUI 收到响应，不代表 worker 完成时刻。预热响应单列于 `background_warmup_results`，只有 `formal` 正常返回计入后台正式完成。审计器仅对时间与任务身份匹配、唯一且位于末尾、`error:null`、零候选的明确收尾取消单列 `teardown_cancel_explicit`；取消仍属未完成任务。阶段信息缺失／矛盾或正式阶段取消均不得豁免。完全没有阶段字段的旧 r1 报告继续采用既有精确 harness SHA 与末项唯一取消条件，保留 `teardown_cancel_inferred`，不将旧数据改写成运行时证明。任何归因均不改变人工测量拒收、超时及背压分母。

**拒收和不完整样本。** 所有拒收、超时、未绘制、未提交及丢失都进入尝试／失败计数；成功延迟必须标明分母。稳定场景中的运动核验超时属于功能／性能失败，即使保留下来的成功样本很快也不能通过；不能把没有返回结果的样本当作零耗时或删除。真实换场导致的安全取消单列为“预期未完成尝试”，仍报告数量、原因及其占全部尝试的比例，不能加入成功数量。没有分位样本或运行中止应标为“失败／未完成”，不得写成通过。

**一小时联合运行。** 使用真实 `LiveExperimentWorkspace`、QPainter 和 QTest 两次点击，持续至少 3,600 秒且至少 10,000 条实际入账；输入 30 Hz、测量尝试 4 Hz、PNG 默认开启，独立 traditional 全图 worker 每十秒尝试一次。固定两场景竖向平移 12 px，并在运行前证实两方向均被生产 detector 判为不稳定。后台自动候选不进入账本。本脚本的重复固定场景以验证生命周期和账本压力为目的，不代替逐帧唯一动态的短程性能验收。

每十五分钟分别审查草稿可见、第二击至接受、第二击至结果可见以及 `gui_publish_ms` 的分位数和最大值，避免一万条历史增长被整小时汇总掩盖。报告实际生成／不同帧绘制 FPS、首次绘制帧龄、运动输入→检测→清场→可见清除耗时、后台任务状态、队列峰值及资源峰值。每个已观察场景转换必须有匹配的检测、清场和绘制事件；漏掉整个转换也不能只从已完成事件的低延迟推断通过。

零串场必须同时核对第一击的 field epoch／field id、记录来源场景与接受时当前实际显示场景，不能仅检查记录的 field id 等于控制器当前值。提交数应等于接受数加已归因终结数；接受数应等于账本数，并与已绘制或有明确换场原因的未绘制结果守恒。结束后排空任务和 PNG 保存，验证每个文件存在及 SHA256、原图尺寸／像素、无额外文件，完整账本 JSON 重载一致，撤销历史条数一致，租约归零，线程／worker 退出。安全换场清除场内覆盖层不得删除已确认历史。512 MiB 帧预算、CPU／GPU 来源预留与进程 RSS 分开披露；队列无界增长、预算超限或无法关闭均不通过。

**平台与报告边界。** Windows 正式 UI 数据必须来自 `qt_platform=windows` 的实际 Qt 窗口；UI benchmark 默认设置 offscreen，因此启动前需显式设置 `QT_QPA_PLATFORM=windows`。soak 可用 `--qt-platform windows`。macOS offscreen 是实际 QWidget/QPainter 与事件循环证据，不能替代 Windows 桌面，也不测物理屏幕扫描显示。Parallels 中 Windows ARM64 运行 x64 Python／扩展的结果须注明 x64 仿真、虚拟 CPU／GPU、DPI 和宿主负载；它可验证该设备路径，不能直接代表原生 Windows x64 用户设备性能。默认 OpenCV 配置作为主要验收，单线程另列压力条件，不把 GCD 的 `getNumThreads()` 当成精确实际并发数。

真实相机／采集卡、真实 CU 画面、曝光变化／失焦／振动和带时间戳的显微录像仍需独立验证。脚本退出 0 或 soak 的 `passed=true` 只表明其程序内完整性检查通过，**不自动断言本表的所有性能门槛通过**；最终须对原始 JSON 按尺寸、输入、负载、平台逐项判定，并保留失败和未验收项。

**复算现有结果。** 在项目根目录运行以下命令，将 `path/to/real-qt-one-hour.json` 替换为真实 Qt 一小时报告路径；多个报告可重复使用 `--soak`。工具默认读取本目录规定的八个 `final-{macos,windows}-ui-{stable,dynamic,joint-stable,joint-dynamic}.json`，缺失或不完整的数据会明确标记。它仅解析已有 JSON，不启动应用、测试或性能采样；实际 `display` 窗口／画布／DPI 信息也参与相对比较匹配。

```bash
python scripts/validation/audit_cu_live_performance.py \
  --soak path/to/real-qt-one-hour.json \
  --output-json docs/validation/cu-live-measurement-2026-10-15/final-performance-audit.json \
  --output-md docs/validation/cu-live-measurement-2026-10-15/final-performance-audit.md
```

命令退出码只表示解析程序是否执行完成；验收结论读取输出中的逐项 `status`／`overall_status`，不能将退出码 0 解释为性能通过。`--no-default-ui --ui <报告路径>` 可只审查指定的短程报告；冒烟记录仍按正式门槛显示不足的预热、样本数和运行时长。
