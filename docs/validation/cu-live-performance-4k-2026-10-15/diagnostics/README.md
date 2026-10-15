# 优化过程诊断

这里保留小样本诊断及失败尝试，不用于替代正式预热 5 次／采样 30 次结论。

- `windows-first-*`：只优化重心／LUT 后仍发生动态核验超时，说明原入账目标不断追赶最新显示帧是主要调度问题。
- `windows-pipeline-lsb-noise.json`：两阶段核验已消除采样中的两秒超时，但旧编码器仍造成缓冲压力。
- `windows-fast-joint.json`：分块快速编码后单独原图路径改善，并发全图仍暴露吞吐不足。
- `windows-background-joint.json`：试验 Windows PROCESS_MODE_BACKGROUND_BEGIN（更强后台模式），完整全图出现 30 秒硬超时，故撤回；未放宽超时，也未作为生产方案。

另一个逐像素融合“转灰度＋缩小”的 C++ 尝试因与 macOS OpenCV HAL 的舍入不一致而放弃；最终仍保留原 cvtColor 与整体 resize。未以放宽总误差掩盖数值差异。

正式生产采用 BELOW_NORMAL（回读 0x4000）；最终连续帧使用单行内存视图调用同一 cvtColor，非连续帧保留完整行块。具体正式结果位于上一级报告。

- `repeat/`：第二轮同版代码的三次独立进程复测全部保留，第三次出现 22/30 入账、9.05 FPS 和 8 次缓冲拒收，不用前两次覆盖。
- `contention/`：三轮加分段计时的诊断，以及逐秒进程 CPU／内存／磁盘观察；有监测开销，不作为正式性能结果。采样中的主要 CPU 消耗来自测量 Python 进程，未证明安全软件或其它进程是慢速轮次的原因。Windows `thread_time` 量化较粗，不从亚毫秒样本推导精确 CPU 占比。
- `flat-probe/`：保持同一 OpenCV 算法，仅测试连续帧视图的批量转灰度；正式采用后另做完整复测。
- `png-pack.json`：比较 NumPy 分通道与 cv2 通道重排；孤立写入仅小幅改善，未作为生产改动，以免扩大 OpenCV 工作线程竞争。
