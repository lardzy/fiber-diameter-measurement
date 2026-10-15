# 人工贴边、运动检测与 Windows 实时来源验证

本目录记录 2026-10-15 的专项证据。它不等同于真实显微样品计量验收、Microview 实物板卡验收或最终安装包验收；总体 UI、自动算法、长时运行和发布报告另列。

## 人工贴边内核

`native/cu_manual.cpp` 使用 C++17/STL 和 NumPy 缓冲区，不加载旧 CU/OpenCV DLL，不启动旧 EXE，不写临时图像。Python 入口为 `fdm.services.cu_manual_measurement.measure_manual`，输入为 uint8 灰度或 BGR 与两端点，输出状态、拒绝原因、像素直径、修正端点和有序边缘点。

保留了原 LineDDA 不含终点语义、整数 BGR 灰度公式、灰度变换、逐次直方图阈值、4 连通小组件清除、每边 21 个追踪点、角度拒绝和对称距离规则。只有划线附近区域参与计算；38 像素保护边保证小组件判定不会被 ROI 截断改变。ROI 超过 8 Mi 像素时在分配前明确拒绝，正常 6K 短划线只处理局部区域。只读输入的 ROI 复制与计算释放 GIL；检查点重新取得 GIL 并原样传播取消/超时异常。

- 最终 macOS 独立原生探针：246/246 录制对照通过；192 个成功例的直径 double 位级一致，最大差值 0；状态、端点和两条有序边缘点也一致。
- 最终专项测试：29/29 通过，含精度、边界、非连续输入、取消、超时、异常恢复、ROI 预算和 GIL。
- Windows 早期版本曾通过同一 246 组；最终统一 native 源摘要版本的 Windows 复核由发布报告记录，不能用早期结果替代。

最终统一生产 `_kernels` 也重新通过 246/246，192 个成功例 double 位级相等、最大差值 0；证据为 `manual-production-macos-final.json`。独立探针证据 `manual-macos-final.json`、`manual-macos-tests.xml` 保留作为历史交叉核对。JSON 保存源摘要、二进制摘要、每例输入/录制结果摘要及逐项判定。正式人工性能以总体配对报告为准。

## 运动检测

保留先转灰度再 1/3 线性缩小、暗像素质心门限、普通 float32 复数互相关、严格半宽回绕以及奇偶列并列峰规则。Sobel 强度使用无符号 `>70`，修正旧 signed-char 把 128–255 强变化漏掉的缺陷。旧 CU5/CU6/免脱色三套共 111 个数学探针全部通过，12 个强 Sobel 案例按已批准的修正规则由“稳定”改为“变化”。

几何缓存必须逐像素完全相等且参考视场未更换才命中，不凭时间戳或哈希。命中后仍计算与上一帧的 Sobel 差异，避免把首次外观变化与后续稳定帧混为一谈。灰度归一化使用 256 项 LUT，和原 double 除法再转 float32 逐值一致。缓存持有的缩略图计入 `retained_bytes`。

`motion-cache-exploratory-macos.json` 是 **2 次预热、10 次采样的探索性短测**，含 BGR 灰度、缩小和运动两阶段，不含采集、GUI、PNG 或帧分配。动态输入交替独立 1-bit 噪声，无法命中相等缓存。与禁用缓存的同算法配对输出全部一致。

| 输入 | 相同像素 P95 | 动态噪声 P95 |
|---|---:|---:|
| 2048×1365 | 0.81 ms | 9.32 ms |
| 4096×2730 | 2.59 ms | 37.18 ms |
| 6144×4096 | 7.47 ms | 76.71 ms |

这些时延来自后续原生峰值扫描优化之前，不能作为当前端到端性能验收。`motion-windows-before-native-peak.json` 和相应 XML 也明确保留该版本边界。最终生产原生峰值版重新通过 111/111，仍只有预期的 12 个 Sobel 修正变化，证据为 `motion-native-peak-macos-final.json`；人工、运动金标、峰值逐点 oracle 与大图阈值回归共 166/166，见 `manual-motion-highres-final.xml`。探索中的 SciPy FFT 在周期/平峰出现舍入选峰差异，未用于生产替换；生产仍使用原 cv2 DFT/idft。

## Windows 捕获与预算

WGC 使用 HWND 身份（PID、进程创建时间、root/child 关系）、CreateFreeThreaded、D3D11、单个帧池纹理。只裁剪目标子窗口的物理 client 像素，直接从映射的 BGRA staging 逐字节写入 BGR 输出；分辨率和三个颜色通道保持不变，没有中间 BGRA NumPy 全帧。像素相同不会被当作断流，帧到达按原生时间和序号判断。

Windows 11、200% DPI 受控窗口最终 9/9 检查通过：遮挡、尺寸变化、移动、最小化、恢复、关闭、时间/序号单调、完整 BGR 像素与缓冲计费。124 个已稳定帧的下半部完整像素区域均符合预期。证据：`wgc-bgr-budget-windows.json`。

`acquisition_reserved_bytes` 分别计入单个 **root** GPU 纹理 payload、CPU staging 的实际 `RowPitch × child_height` 和下一交付 BGR 数组；不再用 child 尺寸估 root 的 GPU 帧。已经交给父进程的 mailbox/current/presented/draft/PNG 帧由 controller 按底层 owner 去重计费，不能重复或遗漏。512 MiB 是完整帧预算，包含 GPU 帧 payload；motion/Qt 工作区、图形驱动不透明分配和进程 RSS 另报告。

6K child 与 root 都约 6144×4096 时，GPU 96 MiB + staging 96 MiB + next BGR 72 MiB = 264 MiB；三个独立父 BGR owner 再占 216 MiB，共 480 MiB。第四个独立父 owner 会超预算，必须暂停新增意图/等待证据写入，不能假换场或降低测量分辨率。实际 root 比 child 大时按真实值重算。

仅 WGC 能力在启动时不可用才允许 `printwindow_compat`：独立常驻 spawn 进程复用 PrintWindow，GDI 直接写共享内存，父进程有截止时间并可终止挂起子进程。没有桌面 BitBlt 回退。损坏 ABI、源摘要不匹配、运行中关闭/最小化不会切换到回退。兼容路径测试 8/8 通过，150 个稳定帧全正确；首次验证暴露移动中 Win32 几何读取竞态，现采用稳定快照重试并有回归。证据：`printwindow-windows.json`。

独立 PyInstaller onedir 捕获探针也实际运行通过：WGC 9/9、PrintWindow 8/8，分别 122 和 158 个稳定帧完整验证正确；冻结后的 spawn 子进程、共享内存、最小化后的助手重启与退出回收均执行。证据：`frozen-wgc.json`、`frozen-printwindow.json`。这是独立探针的冻结验证，最终 core/full 应用及安装包另行验收。

Microview 使用已有确认的 `MV_CaptureSingle` ABI、一个常驻设备进程、按实际尺寸动态分配的 SDK raw 和共享槽，默认最高 30 fps、单槽上限 128 MiB。GRAY/BGR/BGRA 直接写共享槽，RGB555/565 按行转换，无全帧 decoded 中间副本，无逐帧进程或 PNG。交接前停止既有原生预览，避免双开设备；是否能达到 30 fps 及真实 6K 格式必须在物理板卡上验收。

Windows 来源/运动测试 154/154、来源/locator 测试 55/55 通过。空 selector 不再匹配未知空进程名的系统窗口。只读真实窗树检查没有发现正在运行的 CU 预览，故 **真实 CU 画面、Microview 板卡、多物理屏幕 DPI 切换均未验收**。原 Windows 项目及其 Git 状态保持不动，验证位于 `C:\tmp\fdm-cu-live-20261015`。

捕获来源 `stop() -> bool` 只有在自有线程、helper 和自启 preview 全部确认退出后才返回 True。超时或 SDK 停止失败时保留所有者并返回 False；界面不得丢弃来源、切换设备或完成窗口销毁，可稍后重试。原先由别处打开的共享 preview 不属于该来源，退订时不会关闭。worker 的每次交付结束后立即释放局部 frame/packet，避免被预算拒收的上一帧跨到下一次分配而多占一个完整 owner。上述生命周期和弱引用释放回归包含在最终 macOS 来源 40/40 测试中；对应最终 Windows 应用复核以发布报告为准。

## 无损 PNG 策略短测

`png-strategies-6k-exploratory.json` 固定 6144×4096 BGR、种子、OpenCV 单线程；每种策略预热 5 次后交替采样 10 次，每次编码结果均解码逐像素验证。180 次回读全部一致，计时不包含解码。它是合成输入上的探索性编码实验，不是实际显微图像分布或完整 UI 的验收。

| 合成输入 | compression=1 默认 P50 / 大小 | compression=1 RLE P50 / 大小 |
|---|---:|---:|
| 均匀条带 | 201.6 ms / 0.344 MB | 196.2 ms / 0.080 MB |
| 独立 1-bit 噪声 | 751.6 ms / 21.26 MB | 653.1 ms / 17.10 MB |
| 自然梯度 + 噪声 | 947.6 ms / 53.23 MB | 647.6 ms / 49.30 MB |

RLE 在这三类中没有增加存储，噪声例分别缩短约 13% / 32% 编码时间，已由主流程采用并另加无损回归。HUFFMAN_ONLY 对噪声更快，但均匀条带扩大为 9.51 MB；compression=0 固定约 75.62 MB，未作为默认。绝对耗时受同期工作负载影响，正式端到端性能另测。

## 复现

```text
python scripts/validate_cu_manual_measurement.py --reference-root <cu_reverse> --output <manual.json>
python scripts/validation/validate_cu_motion_reference.py --output <motion.json>
python scripts/validation/benchmark_cu_motion.py --output <motion-performance.json>
python scripts/validation/benchmark_cu_png.py --output <png-strategies.json>
python scripts/validation/validate_windows_capture.py --backend wgc --output <wgc.json>
python scripts/validation/validate_windows_capture.py --backend compat --output <compat.json>
python scripts/validation/inspect_cu_capture_windows.py --output <existing-cu.json>
```

采集探针只创建自己的测试窗口并自动关闭；真实 CU 检查只枚举并尝试采集已有窗口，不启动、恢复或驱动 CU。`scripts/sync_cu_live_windows.py` 将源码快照通过 Parallels 私网单 ZIP 临时端点同步，接收后停服务，SHA 验证后可 `--build` 用目标 MSVC 编译。
