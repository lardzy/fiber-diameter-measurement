# 运动局部优化与 PNG 后续实验边界

本轮编辑下列四项，原生扩展已重新构建；已完成针对性正确性回归，性能仍待新版独占采样。优化前正式 `final-*.json` 保持不变；优化前五个源码文件保存在 `build/cu-live-measurement/pre-motion-remediation-sources/`，逐文件前后摘要见 [overlay](motion-remediation-source-overlay.json)。旧 UI benchmark／soak 脚本身份不变，但其下一次运行必须记录新的生产源码身份，不能沿用旧结果宣称新版本通过。

1. `controller.py`：manual／motion future 完成时发送合并的 Qt queued 通知；只在 GUI 线程 `poll()` 发布结果。保留 8 ms watchdog、空闲时的 40 ms 间隔和有 job 时的既有绕过规则；待确认的人工／全图结果同样绕过空闲间隔。新帧接纳、第二击提交与确认动作建立时尽早尝试安排运动计算。没有修改实际绘制节奏、最新 presented 门禁或两秒拒收时限。
2. `controller.py`：删除运动线程得出不稳定后立即建立、随后又被 `end_field`／下一 job reset 丢弃的参考 FFT。下一 job 的参考初始化仍由原 reset 流程负责。
3. `motion.py`：仅 uint8、非空 absdiff 的最大值 ≤8 时用已证明的整数上界直接返回 Sobel 比例 0；其它类型、强差值、`previous` 更新及判据保持原路径。
4. `native/kernels.cpp`：相关峰改为一次连续遍历，分别维护偶／奇列的第一个局部最大值；仍用 double 提升、原乘加和奇列正值并列优先，严格半宽回绕不变。

首轮联合回归为 807 passed、57 subtests passed、5 failed；失败均已定位并修复：四个确认门禁用例暴露人工／全图确认仍被空闲 40 ms 间隔推迟，现已与普通待提交 job 一样立即核验；一个预算用例原先假定 evidence 释放即无旧帧 owner，忽略新调度提前启动的运动降采样任务。预算用例现确定性保留旧 motion lease，要求它与 evidence 均释放后才能新建草稿，生产预算没有放宽。

修复后 `content_workspace`、`motion_remediation`、`motion_peak`、`motion_reference`、`runtime` 共 **251 passed**；见 `build/cu-live-measurement/r2-fix-targeted-macos.log` 与同名 XML。覆盖停止 watchdog 后完成通知仍发布于 GUI、通知合并、queued 回调关闭后无效、迟到移动帧不入账、下一场 reset、只读／切片输入、Sobel 8／9 边界及 70 附近结果、奇偶并列／奇数宽度／极值峰。其余整体回归、111 运动 gold／246 人工 gold 的新版独立验证与性能由统一验收继续推进；上述局部通过不替代它们。

随后独占执行新版阶段 profiler 与配对运动 benchmark，再跑真实 Qt 的 5 次预热／30 次正式条件。`benchmark_cu_motion.py` 的 baseline 使用独立保留的原 CV2 Sobel，实现不得共享新短路；新版阶段 profiler 的 Sobel 阶段明确包含短路，不称为“优化前阶段”。阶段总耗时、输出精确相等及各尺寸绝对门槛分别判断。没有数据前不承诺 4K／6K 能达到 ≥27 FPS 与严格入账延迟。

## PNG 流式写入：仅待实施实验方案

EvidenceStore 和 FrameBudget 生产代码保持原样。当前 512 MiB 统计帧像素 owner 与采集预留，未统计 `imencode` 输出压缩数组及 Qt／驱动全部分配；这不是总内存预算已完成验收。不能只在编码结束后记压缩数组，从而遗漏原图与输出同时存活的峰值；也不能释放仍在编码队列等待使用的原图。

后续若有必要，可在独立实验中比较现有 `imencode → Python write/fsync → SHA → replace` 与 `cv2.imwrite(临时 .png) → 释放原始帧/租约 → Python fsync → 分块 SHA → replace`。这项实验尚未实现或执行，前者仍是生产路径。

- 在 macOS、实际 Windows x64 Python／OpenCV 环境分别测试 ASCII、中文目录／文件名及非 BMP 字符路径。不能假定 OpenCV C++ 文件写入支持 Windows Unicode；任一合法路径不兼容即不能直接替换。
- 固定 2K／4K／6K 的均匀条带、一位噪声、自然梯度加噪声；参数保持 compression 1＋RLE，逐图解码逐像素比较并记录尺寸、文件大小、SHA。额外检查只读输入、灰度／BGR／BGRA 和大文件失败行为。
- 以独立进程对照分阶段耗时、RSS 峰值与原始帧弱引用释放时点，验证 `imwrite` 确实不保留完整编码输出，而非仅把缓冲藏入 OpenCV。峰值仍须涵盖编码期间的原始像素、编码器内部工作区及文件缓冲；不能仅用 Python 对象数量推断内存减少。
- 临时路径必须保留 `.png` 扩展以选择编码器，最后仍用同目录原子 replace。验证写入失败、fsync／hash／replace 失败均保留已接受记录的失败状态并清理临时文件；完成前不得标记 saved。分块 SHA 不可一次读回全文件。
- 若确需执行器 holder 清空，提交时仍由 holder 与 lease 持有尚未编码的原图，进入 worker 才转移所有权；编码完成后释放原图，不能留下 `_WorkItem.args` 隐式引用。关闭须排空或明确失败，不引入等待内存且无法释放排队原图的死锁。

即使流式写入成立，也只是候选生命周期／内存改进；PNG 编码吞吐、中文路径、持续预览和安全背压都要独立验证。本轮不以扩大内存上限、删除拒收、降低原图分辨率或有损编码换取达标。
