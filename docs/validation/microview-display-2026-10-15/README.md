# Microview 原生预览、处理后像素与传输修正

日期：2026-10-15。范围限于采集服务、原生帧桥、helper 和共享帧传输。本目录不替代完整主窗口、测径、标尺或真实 Microview 设备验收。

## 最终结果与边界

- 普通实时采集恢复 `native_embed`，由原 SDK 和专用原生 HWND 显示；试验画布取得 `AFTER_PROCESS=1` 的处理后像素。未改变场模式、图像处理参数，也未新增去隔行算法。
- 试验页隐藏但保留原生 HWND，不将 SDK 目标改成试验画布。回调 BOOL 只是请求；不能依赖它阻止所有随包 DLL 绘图。
- 修复 GUI 停顿后先闪回一个旧共享槽的问题。C++ 更新共享的最新有效序号；GUI 解码前只检查一次，确有更新帧才丢弃并立即 ACK，无固定帧龄阈值。
- 把 helper 的 3 ms 定时轮询改成原生帧与控制输入共同触发的自动重置事件。没有调用 `timeBeginPeriod` 或改变系统电源/定时策略。
- Windows 最终限定回归 **77 tests、3 subtests 通过**。原生自检 7 项全部为真。macOS 不依赖过期二进制的限定回归 **88 tests、3 subtests 通过**；最终 macOS 原生/整机回归由主任务统一执行。
- 最后一轮 250 帧：有效输入 21.276 FPS、订阅输出 21.189 FPS，传递率 **99.6%**；软件帧龄 P50/P95/P99/max 为 **7.647/9.334/9.904/10.395 ms**。前一轮事件版为 100%。这些是测试 DLL 的 callback→订阅者数据，不是传感器到屏幕延迟，也不是实际相机帧率。
- 真实 Microview 的连接、方向、裁边、完整帧、动态横纹、原生预览与试验画面的一致性，仍须真机核对。本测试没有相机、采集卡或真实运动录像，不能宣布这些硬件问题已验收。

## SDK 证据与实现选择

使用本地 Microview VER5 SDK 程序员手册，以及随包 `MVAPI.dll`/`saa7130.dll` 的静态反汇编。手册来源位于 `.tmp/Microview-ref/MVFG/document/MICROVIEW VER5.0 SDK程序员手册.pdf`；相关页为 4、6、19、36、51。

1. 手册定义 BEFORE 为原始像素、AFTER 为处理后像素，声明 AFTER 在显示前执行且 BOOL 控制显示。`SkipPixel` 是每行前缀。`WORK_SKIP` 是写入内存时的行穿插规则，不能据名称当作去隔行设置。
2. 随包 **SAA7130 实际顺序与通用手册不同**：它先处理并绘制，再调用 AFTER；调用后的 EAX 没有参与显示分支。不能用“回调返回 FALSE”证明原生显示被禁止。
3. `DISP_PRESENCE=0` 会跳过该 DLL 的显示/处理线程及 AFTER 回调，不能为隐藏试验页而关闭它。
4. 保留一个有效、专属、隐藏的原生 HWND，使试验页只有 Qt 画布可见；不要把 SDK 的目标 HWND 指向 Qt 试验内容。`native_preview_visible=False` 仍向遵守手册的 SDK 请求停止显示，但实际隐藏由 GUI 负责。
5. DLL 没有导入 `IsWindowVisible`/`IsIconic`，已检查路径没有根据绘图成功与否决定是否调用 AFTER。由此支持保留隐藏 HWND 的方案，但不能据静态检查保证所有板卡、DirectX 后端和驱动都支持。DirectDraw clipper 的解释属于推断；硬件边界保留。

静态证据定位（模块内虚拟地址）：

| 内容 | 定位 |
|---|---|
| MVAPI 回调转发 | `MVAPI.dll` `0x10003a60`，设备 vtable `+0x18` |
| SAA BEFORE/AFTER 保存位置 | `saa7130.dll` `0x1000af10`；kind 0 为 `+b8/c0`，kind 1 为 `+c8/d0` |
| SAA 显示线程和 presence 判断 | `0x100053f0`，检查对象 `+0x160` |
| 先处理绘图，再 AFTER | `0x100054ab` 调用 `0x100151a0`；之后 `0x100054b0` 取 AFTER 指针，`0x100054e1/550f/5545` 调用且忽略返回值 |
| GDI 绘制路径 | `0x100152ff → 0x10015080`；`StretchDIBits 0x10015116` / `SetDIBitsToDevice 0x1001516e` |
| DISP_PRESENCE 参数 6 | getter `0x100128b8`、setter `0x10012e07`，对应 `+0x160` |

已核对文件 SHA256：

- `runtime/camera/microview/x64/MVAPI.dll`：`1d22ce270ea4aa37fb1073943a707c82f44f255164e480e1877d606b531933ea`
- `runtime/camera/microview/x64/saa7130.dll`：`cbb60864f7c2526ce32515132bc04b89eb2c5e177ec5af62804e8b68e7cb6315`

这里没有把“576 行完整尺寸”当作“无时间梳状纹”的证明，也没有根据截图猜测并开启 `IMAGE_PROCESS`、`WORK_FIELD` 或单场放大。

## 所有权、最新帧与唤醒

原生回调仅校验元数据、向两个预分配槽之一复制像素、记录序号/时间、更新共享序号并触发事件；不进入 Python/Qt。回调使用 try-lock，消费者持锁时丢弃本次 tap，不阻塞 SDK 原生预览。

helper 将最新原生槽复制到一个共享像素槽。GUI 取得独立数组后 ACK；已被测量持有的像素不会被覆盖。GUI 停顿时 helper 保持 pending 槽不变，原生两槽继续更新；新增的 8 字节对齐序号标记独立于 ACK 推进。恢复后若标记较新，GUI 不解码旧槽，直接 ACK，使 helper 下一次立即取得最新槽。

这个检查只有一次，位于解码前。没有用 50/100 ms 等年龄阈值，低帧率或暂时不变的画面不会被误判为失效；也不会在大图解码后反复追逐新帧而饿死显示。标记由 Windows `InterlockedExchange64/InterlockedCompareExchange64` 跨进程读写，C++ 同时持有 Python owner 和实际 buffer export。重绑定在槽锁内完成；旧映射只有解除绑定后才能释放。

所有控制命令仍进入有序队列；自动重置事件只合并唤醒，不合并或丢弃命令。ACK 清除 pending 后，同轮调用 `publish()`。回调完成、容量不足、stdin 控制输入均触发事件。helper 使用 `get_nowait()`，100 ms 等待上限只用于断流看门狗。退出时先关闭控制入口，reader 临时持有 tap 引用；注销回调、停止/关闭设备并等待在途回调后才释放事件与映射。

## 逐轮性能数据（失败和干扰样本保留）

共同设置：768×576 BGRA；真实 Windows 测试 DLL，MSVC `/O2`；预热至少 5 帧，通常 250 个正式样本（legacy-awake 为 150）。测量原点为原生 AFTER callback **复制完成后**，终点为 `CaptureSessionManager` 的帧订阅回调。探针多加了分段时间字段和 JSON，不参与生产像素算法。

| 原始 JSON（microview-transport- 前缀） | 条件 | 样本 | 帧龄 P50/P95/P99/max ms | 有效源/订阅 FPS | 传递率 |
|---|---|---:|---|---|---:|
| [latency-build-interference](microview-transport-latency-build-interference.json) | API2，编译重叠/隐藏 QWait | 250 | 37.23/82.84/163.20/205.56 | 13.833/10.470 | 75.7% |
| [latency-hidden-qwait](microview-transport-latency-hidden-qwait.json) | API2，隐藏 QWait | 250 | 37.27/46.09/167.17/233.16 | 13.247/10.056 | 75.9% |
| [latency](microview-transport-latency.json) | API2，QEventLoop/桌面未核验 | 250 | 17.27/102.10/121.99/131.80 | 12.193/8.577 | 70.3% |
| [fixed-awake](microview-transport-fixed-awake.json) | API2，固定节拍/桌面已核验 | 250 | 7.45/9.32/44.32/80.34 | 21.585/19.537 | 90.5% |
| [legacy-awake](microview-transport-legacy-awake.json) | API2，工作+sleep 对照 | 150 | 12.44/68.23/74.19/79.27 | 15.455/10.241 | 66.2% |
| [api3-fixed-awake](microview-transport-api3-fixed-awake.json) | API3，序号标记/3 ms 轮询 | 250 | 7.18/15.88/50.62/80.68 | 21.689/19.949 | 91.9% |
| [api3-event-fixed-awake](microview-transport-api3-event-fixed-awake.json) | API3，事件/首轮序号探针错误 | 250 | 7.64/8.86/9.41/10.17 | 22.382/22.377 | 100.0% |
| [api3-event-final](microview-transport-api3-event-final.json) | API3，事件/最终正确序号 | 250 | 7.65/9.33/9.90/10.40 | 21.276/21.189 | 99.6% |

各轮解释：

- `latency-build-interference` 与父任务原生编译重叠，属于受干扰数据；`latency-hidden-qwait` 全隐藏并使用 `QTest.qWait(1)`。这两轮及首个 `latency` 轮次的 Windows 桌面状态没有当时核验。之后发现曾有黑屏，故不能作为正式流畅度验收。
- `fixed-awake` 开始确认桌面已唤醒、未锁屏，并采用正常 `QEventLoop` 与可见的独立空白窗口。SDK 专用 HWND 保持隐藏。源改为绝对 25 Hz 节拍与缓存的已处理图，过期节拍跳过，不补发旧帧。
- 原 `legacy` 源是“图像工作耗时 + sleep(40 ms)”，不是严格 25 FPS。该源 `/O2` 工作耗时 P95 约 3.14 ms，不能把全部低帧率归因于像素生成；Windows 实际调度和 sleep 超时也参与。所有报告都保留实际 callback 计数和时间窗口作为分母。
- API2 `fixed-awake` 在约 260 ms GUI 停顿后，首个共享槽帧龄 **206.56 ms**，第二帧才恢复新画面。这是明确的代码背压/旧槽问题。
- API3 轮询版 `api3-fixed-awake` 已丢掉那个 pending 旧槽，但传递率仍为 91.9%。该轮 `queue.get(timeout=3 ms)` 实测 **P50 16.407 ms、P95 56.688 ms、max 91.281 ms**，不能把名义 3 ms 当成实际等待。它的 ACK reader→helper P95 2.644 ms、stdout→GUI P95 1.584 ms，主要问题落在无帧轮询等待。
- 事件版第一轮 `api3-event-fixed-awake` 传递率 100%，但报告中的 `first_recovery_is_current_or_newer=false` 是**探针比较了不同序号域**：manager 对接收帧重新编号，标记使用原生 callback 序号。该原始报告未删。下一轮改为从帧消息读取原生 sequence，保留 manager sequence 供审阅；产品实现未因此修改。
- 最终 `api3-event-final` 传递率 99.6%。暂停实测 **283.93 ms**；恢复时最新标记 259，第一条接收原生序号 260，帧龄 **7.522 ms**、恢复后 **7.832 ms** 到达；旧共享槽丢弃 1 个。前/后半程帧龄 P95 9.462/9.065 ms，没有随队列时间增长。

最终轮分段 P95：

| 阶段 | ms |
|---|---:|
| AFTER 复制完成 → 共享槽就绪 | 0.741 |
| 原生槽 → 共享槽复制 | 0.302 |
| stdout 发布 → GUI 开始解码 | 1.571 |
| 解码成独立像素 | 3.106 |
| manager 图像转换/复制 | 2.981 |
| manager 排队发布 | 0.656 |
| 非阻塞控制队列读取 | 0.118 |

这些阶段百分位不能直接相加。事件版 `helper_poll_gap` 包含等新输入的间隔，不是收到新帧后的延迟；应看 callback→共享槽就绪 0.741 ms。模拟源的绝对节拍仍有 P95 67.93 ms 调度偏差，实际只有 21.276 FPS；传递率达标不代表输入达到请求的 25 FPS，更不代表真实 Microview 硬件验收。

## 回归、构建和重现

最终 C++ 源摘要：`d743608532e85f90e11170b87b9ec4fc0661065c866980129eccd23cf62c7b28`。

`_microview` 接口版本 **3**，`_kernels` 接口版本仍为 **1**；两者比较相同源码摘要和 Python ABI，不再错误要求两个不同接口版本相等。Windows Python 3.12.13 x64、pybind11 3.0.4、MSVC 1944，运行于 Parallels 的 Windows 11 ARM 虚拟机。VM 配置为 4 vCPU、16 GiB；不是 Windows x64 裸机成绩。

隔离目录：`C:\tmp\fdm-cu-live-microview-return-20261015`。没有修改用户原项目目录、驱动或默认设备配置。最终两个产物均在 `src\fdm\_native`：

- `_kernels.cp312-win_amd64.pyd` SHA256 `f3c4f59a12ae03ee742c3aa25fac3c73f7248a884898fa962532a04322eb0ddc`
- `_microview.cp312-win_amd64.pyd` SHA256 `25118ef78a6f655ecbb7b3a57571e7e48f478272d0f565d4a4aeb041b259fe7d`

完整构建信息和真实自检结果见 [final-build-info.json](final-build-info.json)。两份 pyd 构建、测试 DLL 编译、最终限定回归与两轮事件采样均退出 0。

主要真实 DLL 验证：大于 4 GiB 的指针/上下文，SDK 20 字节结构体，RGB/BGRA/行前缀，原图与 SDK 处理后像素区分，半高场拒收，BOOL 生效和 SAA 忽略 BOOL 两种顺序，隐藏但有效 HWND 与独立试验 HWND，20 轮设备/会话启停，helper 崩溃重启，标记 export 持有/解绑，ACK 中旧槽丢弃，容量切换，控制信号合并，40 轮无帧控制突发后安全停止。

最终命令（已设置 `PYTHONPATH=src;validation_vendor`、`FDM_TEST_MICROVIEW_DLL=build\test-sdk\MVAPI.dll`、`QT_QPA_PLATFORM=windows`）：

```powershell
python -m pytest -q tests/test_microview_native_windows.py tests/test_microview_display_routing.py tests/test_capture_and_cu_scale.py tests/test_live_experiment_sources.py
python scripts/validation/benchmark_microview_transport.py --sdk build/test-sdk/MVAPI.dll --output build/cu-live-validation/microview-transport-api3-event-final.json
```

首轮 API3 负例在 Windows 失败：重新打开映射时系统可能按页报告 `memory.size`，因此仅凭映射长度无法证明协议标记偏移正确。已改为要求偏移恰为声明像素容量的 8 字节对齐位置。首轮 **1 failed / 74 passed / 3 subtests** 日志保留在 `microview-api3-tests-first-failure.txt`；修复后 `microview-api3-tests` 为 75 passed，加入事件测试后 `microview-api3-event-tests` 为 77 passed。API2 的 63 项原始日志同样保留。

测试 DLL 只用于软件验证，不随安装包交付、不安装为驱动。其“处理”是人为 XOR 标记，用于证明选择了处理后回调，不能用它证明真实设备去隔行、色彩或几何正确。
