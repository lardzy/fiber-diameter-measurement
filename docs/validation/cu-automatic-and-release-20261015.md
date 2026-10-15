# CU 全图算法、工作进程与发布验证记录

本记录覆盖 `cu_automatic`、共享计算许可和发布构建。人工测量、相机真机、界面交互与含量公式的完整验收由各自报告负责。以下是可重复的阶段性证据；算法对照通过不代表真实显微样本的计量有效性或相机硬件验收。

## 已实现的路径

- 传统算法保留逐圆候选、`C²` 关系矩阵、原排序、连接、短/长纤维筛选及直径样点。输出动态分配；原 CU6 的 30 根输出截断不再存在。并未加入 kNN、近邻裁剪或改变阈值。
- 多尺度算法完整运行中等、粗、细三个 pass，保留增强、形态学、骨架、节点/线段连接、采样与拒绝规则。
- `native/cu_automatic.cpp` 运行高频循环；OpenCV 4.13 的批量算子通过 Python cv2 调用，不链接旧 DLL，不启动旧 exe，不进行 PNG 文件交换。
- `analyze_frame` 接受 `uint8 GRAY/BGR/BGRA`。输入转为任务独占连续快照，坐标保持原图坐标。返回 `status/reason/candidates/diagnostics`；候选保留 `start/end/width_px/path`，资源限制不返回部分可入账结果。
- 工作进程使用 Windows/macOS 均支持的 `spawn`，输入通过共享内存传输。一个对象复用同一常驻进程，只有一个在途任务。后台 reader 解码结果，GUI 只取本地队列。25 ms watchdog 检查硬超时，运行时最长 30 秒，取消/超时/崩溃后清理并在下一请求重建。
- 共享计算预算按 FIFO 只允许一个重任务运行：全图、魔棒分割、面积推理。人工测量不参与该许可。已有不可中断的 ORT/PyTorch 调用不会被虚称为可抢占；后台调用可等待并检查取消，GUI 主线程遇忙立即返回明确错误。全图排队期间不启动运行时超时。

`memory_budget` 覆盖输入快照、原生图像、实际 vector 容量（包括重新分配时新旧缓冲并存）、精确 `C²` 关系表，以及保守的 cv2 瞬时工作区、轮廓 Python 对象与返回对象预留。最低预算不再按 104 字节/像素重复预约。`native_peak_reserved_bytes` 是这些已计费项的峰值，**不是整个进程 RSS 的硬限**；解释器、第三方缓存和 IPC 仍以实测 RSS 单独报告。全图输入边长限制为 32766，避免旧定点绘线坐标溢出。

## 177 组金标现状

素材取自 `cu_reverse/docs/validation/diameter_demo_20261011.json`；只读取留存的 BMP 和 JSON，不运行原软件。脚本：

```bash
PYTHONPATH=src .venv/bin/python scripts/validation/validate_cu_automatic_gold.py \
  --cu-root /Volumes/LARD/cu_reverse \
  --output docs/validation/cu_automatic_gold_20261015.json
```

2026-10-15 隔离兼容修复的 177 组结果：

| 范围 | macOS ARM64 | Windows x64 / MSVC 1944 |
|---|---:|---:|
| 最终接受候选匹配 | 175 | 175 |
| 经另一原版结果验证的容量扩展 | 2 | 2 |
| 最终候选差异/运行错误 | 0 / 0 | 0 / 0 |
| 有中间图像差异的用例 | 0 | 0 |
| 有内部 pass 候选计数差异的用例 | 0 | 0 |

宽度比较使用 `atol=1e-8, rtol=1e-10`；候选顺序、传统圆索引和样点端点、现代线段点序列逐项比较。两项容量扩展均为 CU6 `many_35.bmp`：新输出 35、旧 CU6 输出 30；前 30 一致，全部 35 与相同图像/参数的原 CU6mt legacy 结果一致。验证脚本现在会因任意中间图像或内部 pass 计数差异返回失败，不再仅凭接受候选通过门禁。

初轮差异已按第一个分歧点修复，未更改筛选阈值：

- 传统 8 例各 1 像素的 contour 差异来自 OpenCV 1.0 灰度 AA 线段的旧裁剪和闭合轮廓绘制起点。移植这一窄原语，替换先绘制再清边的近似方法。
- `stress_connected_01` 的 coarse 首差在 `sift_holes`：旧最小外接矩形的宽高比略大于 3，现代 ARM 值为 2.9999998，导致 130 个孔洞像素、后续 4584 个 selection 像素和一个被拒绝候选不同。
- 同例 thin 在 `filter_segments` 首次少了 19 点的 segment 26：旧宽高比略大于 10，现代值为 9.999999。移植 OpenCV 3.3 旋转卡尺的 float 运算和顺时针凸包顺序，两处恢复原判断。
- 更深入的 coarse 孤立段方向标志差异来自 OpenCV 3.3/4.13 L1 `fitLine` 的最佳解更新时机；保留旧的外层迭代更新后，33 个完整中间阶段（图像、分段点序和全部候选方向/连接）一致。没有采用曾调查的排序补丁。
- 33 阶段证据见 `cu_automatic_stage_equivalence_20261015.json`。该额外诊断在独立 Windows 临时目录编译已经审计的源码 port，仅链接其 OpenCV 3.3 参考库，不加载 CU 算法 DLL；参考库与诊断可执行文件不进入项目源码或发行包。正式运行只使用当前原生扩展与 cv2 4.13。
- 三个窄兼容原语的 Intel 版权、完整许可证、来源与 SHA256 收录于 `fdm/_native/licenses/opencv-legacy.txt`，发布声明和资源清单摘要同步更新。

性能改进保持运算顺序与结果：7×7 整数和改为列滑窗；四连通 flood fill 改为扫描线；去掉重复原图复制；图像/图结构按实际生命周期计费。6K 普通稀疏样本在默认 2 GiB 预算下已完成 8 个候选；正式 5 次预热、30 次采样见最终性能报告，短诊断计时不作验收。

传统直方图加权累加额外消除一个大图容量缺陷：将 `hist[i]` 转成 double 后再乘灰阶索引，避免 4K 亮背景的 int32 有符号乘法溢出。正常域 177 金标保持原值。原反汇编 `preprocessing_and_thunks.txt` 的 0x10005ba7/0x10005bf7 为 32 位 IMUL，之后 0x10005bb5/0x10005c07 才执行 FILD，确认这是原 CU 的大图缺陷。`test_cu_automatic_high_resolution.py` 使用真实 4096×2730 服务路径和独立整型加权 oracle，验证 194 灰阶区不再被溢出误判为前景；原始适配构建失败该检查，修复后通过。

最终两平台原生公共源码摘要：`d3499aa56f20c64c4415adedd10e8d73b8653240806201cb1f3dac4967367086`（macOS CPython 3.13 / Clang 21；Windows CPython 3.12 / MSVC 1944）。

## 已完成的运行与发布验证

- macOS 发布门禁专项：108 tests、107 subtests 通过，包含真实 CU 探针、既有魔棒/水印门禁和双扩展回滚；日志 `build/cu-live-windows/macos-build-gates.log`。
- Windows 自动算法/真实 worker/计算许可/含量统计/模型/项目资产/人工/原生运动峰值专项：310 tests、7 subtests 通过，日志 `build/cu-live-windows/automatic-statistics-final.log`。
- Windows 最终独立 PyInstaller onedir worker 探针成功：冻结进程中两条算法实际执行，连续任务复用同一 PID，公共源码摘要与上文相同。报告 `build/cu-live-windows/automatic-frozen-final.json`。这是独立工作进程探针，完整安装包验收另列。
- Windows 严格入账屏障、同帧证据摘要缓存、退出生命周期与真实 4K 阈值专项 134 tests 通过；后选分类/操作者等专项 156 tests 通过；最终旧控件隐藏修复后的后选/工作区/过渡专项 90 tests 通过。日志分别为 `final-barrier-regression.log`、`final-postselection-regression.log`、`final-ui-hide-regression.log`。
- 最终 Windows 源码快照共 905 个文件：基础 ZIP SHA256 `df162f272e76f3ec3895c4f86f035441c1a879d9cb721e408d3fb0680f2ab058`，最后文案/文档/基准脚本增量 ZIP SHA256 `d3457d41a0907848c9f76a876db3e923a87cd44398c3c538cac3c3dac4e5c79e`。合并后逐文件清单 `build/cu-live-windows/final-source-manifest.json` 全部核验通过，SHA256 为 `33d042eb149cc37c2608c459f84ec5c4739f5ea2d649c0cb19ad4dc9448a9564`；工作区源码 SHA256 为 `2c0a807d1b56151625e423b1cf71d58a005daa5bc01badd944a63b5284040f04`。
- 最终 Windows Qt 实际 QTest 探针经过两击人工测量、右键后选和操作者改名，取得真实操作 ID；全部 5 项检查通过，9 类按钮和键盘模式均可见。`build/cu-live-windows/ui-probe-final/ui-probe.json` 与两张截图记录结果。1280×720 实际尺寸通过；请求 1330×900 被虚拟机工作区限制为 1293×802，全部控件仍可见。此为真实 Windows Qt 功能验证，不能代替正式帧率采样。
- 最终 core 已通过真实 windowed 冻结自检：全部 8 项 CU 检查通过，`errors=[]`、`warnings=[]`，清单含 4729 文件；直接读取其嵌入 PYZ 也确认新后选标签和待确认 tooltip 存在、旧“本场最后有效记录”文案消失。core 清单诚实记录缓存构建 `clean_build=false`。
- 最后一次 full 缓存尝试虽然通过普通功能自检，但被额外的实际 PYZ 文案门禁拒绝：EXE SHA 与文案更新前完全相同，仍含旧文案。源文件在前一轮构建期间修改，保留的 mtime 早于那轮 PYZ，缓存复用了旧归档。失败包、安装器和解包目录均标记 `rejected-stale-pyz`，不交付。未修改原打包脚本；它默认 `--clean` 可避免该情形。随后使用未修改的原脚本默认 clean 重建成功：最终 full 清单明确 `clean_build=true`，4766 文件，完整冻结自检与 EdgeSAM 实际运行通过；实际 PYZ 中新标签和 tooltip 存在，旧文案消失。新 EXE SHA256 为 `2c414d30dc4aafeedfbd85b47d7a35fd686f84f3642506a0820617592adcdd06`，full 清单 SHA256 为 `41dce53fe8270b2b235a1e62950323028d1ef6da7e9721494cd115859db8af96`，详见 `full-frozen-wording-final.json`。

- 早期 core/full 技术包及严格屏障后、后选修复前的 core 自检均已通过，但早于最终源码，统一标记为历史证据，不作为最终发行验收。对应 `*-technical.*`、`*-pre-barrier.*`、`*-pre-afterselection.*`、`*-pre-wording.*`；中断的 full hooks 构建没有生成安装包。
- Windows 独立快照中移走 capture 扩展，并保留过期 kernels，现有 `ensure_native_kernels` 自动准备临时构建依赖、重建两扩展并通过 ABI/源码摘要/真实计算检查。验收脚本只允许在 `C:\tmp\fdm-cu-live-*` 下运行，失败恢复备份。
- 整个验证使用独立 `C:\tmp\fdm-cu-live-20261015`；原 Windows 项目仅提供现有解释器和被读取复制的 runtime 资产，未覆盖旧源码、分支或 runtime。

最终内部安装包沿用 `full --exclude-area-models`，保留 4 个含量模板，不启用 `--public-release`。文件为 `build/cu-live-measurement/windows-release/fiber-diameter-measurement-setup-0.4.7-no-area-models.exe`，398,393,836 字节，SHA256 `8a37a58a382db75729ce8d33085df83445a3cfae299fbb622e79ddab2aa037be`；主机与虚拟机逐字节摘要一致。虚拟机原件位于 `C:\tmp\fdm-cu-live-20261015\dist\installer\fiber-diameter-measurement-setup-0.4.7-no-area-models.exe`。

对**这同一安装器**使用便携 [innounp 2.67.11](https://github.com/jrathlev/InnoUnpacker-Windows-GUI/releases/tag/oi_2_2_11) 解包到 `C:\tmp\fdm-cu-live-unpacked-20261015\{app}`；官方发布 ZIP SHA256 `851772538a041229102ad9964542d49dc00c74002e3091a70469c079ae368f52`、工具 EXE SHA256 `1c8453c198dd3fe7947f3705e59d2294ddce3dfa9f96a737e94449b119f44181`，详见 `innounp-provenance.json`。4766 个发布清单文件全部长度和 SHA256 一致，4 个私有模板保留，面积模型排除，实际嵌入 PYZ 的三项新文案检查全部通过。解包后真实 EXE 的完整 `--self-check --json` 也通过，报告 SHA256 `90836f77ce802a926ca88a9e2fb6d955c22a1b6f9a371b4f6a0921622ac4feb1`。

本轮未执行注册式安装/卸载：现有安装器使用固定 AppId，另选安装目录仍可能替换原卸载注册信息。依照授权采用同包解包运行验证，原 HKLM 安装记录前后完全相同，仍指向 `fdm-native-validation-installed-20261013`。这一证据覆盖同包负载与运行，不宣称已实装替换原应用。

实际解包 EXE 的最终 GUI 入口探针也已通过。早期探针只查 UIA 桌面直接子元素，并且只保存点击前的树，未能确认有 owner 的工作区与关闭提示；该失败记录保留。随后只修正验证工具，生产程序和冻结包均未变化：改为按本次 PID 枚举全部 owned HWND、在各自 UIA 子树查找，明确记录点击后的窗口和控件。

北京时间 03:41:05–03:41:14 在独占时段运行修正版一次，PID 1372，使用全新的 GUID `LOCALAPPDATA/APPDATA` 配置，并先确认“0 张图片 · 无缺失 · 尚未保存”。真实菜单打开“实时试验 · 纤维直径与含量”，11 个关键控件全部存在且可见。点击后和就绪时各保存 358 个 UIA 节点，均未截断；工作区 HWND 2884930 的 owner 为主窗口 HWND 21168870。先关闭工作区，再观察到“未保存的改动” HWND 2950466（同一 owner），保存 198 个节点及实际“保存／放弃／取消”按钮，仅对本次隔离空项目选择“放弃”。程序自然退出，退出码 0，`forced_cleanup=false`、`errors=[]`；收尾确认本次应用及探针进程均不存在。此证据覆盖同一安装包真实 EXE 的主窗口→菜单→实时工作区→正常关闭，不宣称执行了真实设备测量；完整源码 Qt/QTest 操作另见前述 Windows UI probe。

修正版脚本、解析结果、运行结果及五份 UIA 树的摘要单列于 `cu_windows_frozen_gui_overlay_20261015.json`，作为验证工具 overlay，不修改 905 文件源码冻结清单或安装包。PowerShell 解析为 2937 tokens、0 errors。最终 GUI 证据位于 `build/cu-live-windows/gui-corrected/`；该短探针结束后释放独占时段，随后才开始 Windows 正式性能采样。

机器可读发布证据见 [`cu_windows_release_20261015.json`](cu_windows_release_20261015.json)，包括 core/full 清单、自检、实际 PYZ 检查、安装包 SHA、同包解包、GUI 边界与已有安装记录。最后 `uv.lock` 的 benchmark/psutil 修订单列为元数据 overlay（SHA256 `40ba31b624bbfaa6373d8104f79cfeed57214e6a66606fb3dae7387aa919608e`），未改变构建环境、运行依赖或原 905 文件冻结清单。

## 构建接入

原 `.bat`/Python 打包入口保持不变。预检在独立 Python 中执行原生内核和新的 `cu_live_measurement` 真实功能探针；Windows 同时校验 `_windows_capture` 的类、接口/ABI/源码摘要。缺失或过期会重建，失败不继续打包。

两个扩展分别原子替换，并作为一组校验；第二个替换或安装后自检失败时恢复已替换的文件。Windows DLL 被别的进程锁住而无法恢复时，旧二进制保留到 `build/native-kernels/recovery-*` 并报告路径。这里是带回滚的多文件发布，不宣称操作系统提供双文件同时原子替换。

OpenCV 固定为 `opencv-python==4.13.0.92`。core/full 的 Windows 依赖声明及 PyInstaller hidden import 均包括 capture 扩展；扩展被排除出普通 package 的通用收集流程。
