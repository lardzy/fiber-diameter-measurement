# C++ 原生计算内核迁移验收记录

2026-10-05，在 `lard/native-kernel-migration` 完成首轮两项模块：轮廓边界／截线和快速测径短分支剪枝。三个 C++17 / pybind11 内核已接入生产 Python 服务、源码参考后端、setuptools 和冻结发行自检。本机正确性和约定的合成性能门槛均通过。Windows 二进制、安装器及真实显微样本尚未验收。

## 代码与现场

分支从 `d7c00be1281316550f6d98df69acc6fae73a948b` 创建。开始前记录 Git 状态、暂存区、13 个已有变更文件的 SHA256 和副本，保存在本机 `build/native-kernel-migration/workspace-before.json`、`index-before.patch`、`worktree-before.patch`、`original-files/`。

原有数字切片、主窗口、相关测试、研究文档和 egg-info 的内容均保留，见 [现场核对记录](workspace-preservation.json)。运行期间 IDE 自行更新 `.idea/modules.xml` 并出现新的 `.iml`，本次没有回退或纳入迁移修改。验证归档时迁移改动尚未提交，原有暂存状态保留；后续提交情况以 Git 记录为准。

主要入口：

- [`native/kernels.cpp`](../../../native/kernels.cpp)：两遍像素边界扫描、批量截线、八轮内短分支剪枝及 `build_info()`。
- [`fdm._native`](../../../src/fdm/_native/__init__.py)：后端选择、扩展／ABI／源码指纹验证、一次性输入快照。
- [`contour_comparison.py`](../../../src/fdm/services/contour_comparison.py)、[`fiber_quick_geometry.py`](../../../src/fdm/services/fiber_quick_geometry.py)：保留原 Python 算法作参考，通过批量接口调用原生实现。
- [`native_kernels_self_check.py`](../../../src/fdm/services/native_kernels_self_check.py)：真实执行三个内核，独立于已有 scikit-image 自检。

骨架化、几何修订号 3、测量候选评分／置信度／拒绝原因、项目格式和界面规则未因本次迁移调整。取消和超时仍由 Python 检查，保持先取消后超时；内核循环释放 GIL，检查点短暂取得 GIL 调用任务回调。

## 正确性与回归

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 原生、参考对照、取消、相关服务与轮廓 UI 专项 | 181 tests、138 subtests 通过 | [专项日志](targeted-final-native.log) |
| 强制 Python 参考后端及后端选择回归 | 79 tests、138 subtests 通过 | [参考后端日志](reference-final.log) |
| 构建及发行相关回归 | 92 tests、123 subtests 通过 | [构建回归](build-release-tests.log) |
| 全量运行后修正的发行测试夹具 | 13 tests 通过 | [夹具回归](release-fixture-regressions.log) |
| 配对基准工具自身 | 10 tests 通过 | [工具回归](benchmark-tool-tests.log) |
| 正式性能输入的逐对结果对照 | 49 场景、111 个服务／内核阶段全部一致 | [原始 JSON](paired-performance.json) |

归档 `.log` 仅清理了行尾空白，未修改测试结果或诊断内容；未清理的原始日志保留在本机 `build/native-kernel-migration/`。JSON 数据保持原样。

边界数组和剪枝掩膜要求形状、dtype、逐元素完全一致；截线区间数量、顺序、拓扑一致，浮点端点使用 `atol=1e-10, rtol=1e-12`。覆盖空形状、孔洞、多分量、触边、棋盘格及 200 万边上限，旋转／顶点／近水平／短区间，以及直线、阶梯、T/Y/十字、局部宽度和多轮修剪。还覆盖切片、翻转、只读／未对齐数组、非标准 bool 字节、错误类型、计算前／中取消、超时和异常后复用。

全量运行记录为 **18 failed、3716 passed、1601 subtests passed、1 warning**，见 [完整原始日志](full-suite-native.log)。其中 13 项是发行测试夹具未提供当前完整自检字段，已修正并单独全部复验通过；未放宽发行门槛。

其余 **5 项 UI 布局失败已在迁移前源码快照上复现**，见 [基线复现日志](baseline-ui-failures.log)：

- `test_image_page_fits_supported_small_workspace_without_scrolling` 的 dark／light 两项。
- `test_inspector_sections_remain_contiguous_and_expansion_pushes_down`。
- `test_measurement_inspector_uses_locked_section_order_and_defaults`。
- `test_responsive_geometry_matrix_keeps_explicit_commands_reachable` 的 `(1536, 864)` 子用例。

全量运行后段 Qt 全局样式处理耗时明显增大；当时发送过 SIGINT，随后进程输出上述完整失败摘要并以代码 1 退出，日志未出现 KeyboardInterrupt。这里保留原始执行事实，不宣称全量回归全绿。对五项失败的基线复验使用 HEAD 的归档源码加开始时已有修改，未用原生迁移后的代码代替基线。此次不改动这些 UI 功能来消除既有布局失败。

## 正式性能

环境：macOS 27.0.1 ARM64，CPython 3.13.12，NumPy 2.4.3，OpenCV 4.13.0.92，scikit-image 0.26.0，PySide6 6.10.2，pybind11 3.0.4，Clang 21.0.0。CPU 逻辑核心 18，OpenCV 线程数 18；原生内核不创建线程。详细环境、输入 SHA256、源码和二进制摘要都记录在原始 JSON。

正式执行一次，每个后端／每个阶段预热 5 次，交替执行 30 对采样；运行期间没有并行测试或构建。计时从生产 Python 入口开始，包含快照、结果转换与异常返回。逐对语义检查在计时之外。首次正式运行通过，未按结果筛选重跑。

| 预先固定的验收集合 | Python P50 / P95（ms） | Native P50 / P95（ms） | 验收结果 |
| --- | ---: | ---: | --- |
| 2K／4K／6K 正常轮廓完整对比 | 582.472 / 615.187 | 142.963 / 153.037 | 4.074 / 4.020 倍，目标 ≥1.5 倍 |
| 全部 36 个测径场景的剪枝入口 | 24.501 / 25.979 | 3.735 / 4.474 | 6.560 / 5.806 倍，目标 ≥3 倍 |
| 固定复杂分叉 9 场景完整测径 | 52.558 / 55.199 | 43.352 / 45.896 | 耗时降低 17.52% / 16.85%，目标 ≥10% |

表中集合指标按相同采样编号对各固定场景耗时求和，再取 P50/P95，是构造的配对集合分布，不是直接计时的一次整批调用，也不是逐场景提速倍率的平均值。复杂分叉子集在采样前固定为宽度 40、长宽比 3／4／5、角度 15°／25°／75°；没有根据测得速度挑选样本。

全部 **49 个场景** 的完整服务 P50 和 P95 均满足退化不超过 `max(5%, 0.5ms)`。这不表示每项都绝对变快：小矩形测径 P50 增加约 0.119 ms、P95 增加约 0.131 ms，另两项 P95 增加约 0.164 ms／0.015 ms，都处于事先定义的容差内。五个既有拒绝样本保留原拒绝代码。

13 个轮廓场景还包括 17° 轴旋转、孔洞和多分量。每个场景的边界、截线、剪枝及完整调用的 30 对耗时、独立 P50/P95 和输出状态均见 [原始 JSON](paired-performance.json)，[运行日志](paired-performance.log) 记录了完成顺序。性能边界为已有掩膜的几何服务，不包括分割／模型推理、队列等待、Qt 安装结果及屏幕呈现。

### 复制与返回值转换

另用 [`benchmark_native_overhead.py`](../../../scripts/benchmark_native_overhead.py) 对 14 个固定场景进行 5 次预热／30 次采样。诊断捕获生产包装层实际使用的输入和截线输出，输入摘要与正式基准逐一相同；[独立数据](native-overhead.json) 保存数组形状、dtype、布局、复制字节数和每次原始耗时。

| 场景 | 输入快照 P50 / P95（ms） | 两帧截线 tuple 恢复 P50 / P95（ms） |
| --- | ---: | ---: |
| 2K 正常轮廓的截线输入 | 0.01444 / 0.02291 | 0.18212 / 0.23141 |
| 4K 正常轮廓的截线输入 | 0.02460 / 0.03943 | 0.18419 / 0.29924 |
| 6K 正常轮廓的截线输入 | 0.02213 / 0.05673 | 0.18171 / 0.29892 |
| 小矩形剪枝 | 0.00246 / 0.00256 | — |
| 厚斜线剪枝 | 0.05198 / 0.08902 | — |
| 固定 9 个复杂分叉剪枝（逐场景范围） | 0.00338–0.01202 / 0.00352–0.02243 | — |

轮廓边界输入是只读连续掩膜，可借用且复制字节数为 0；两帧快照验证约 0.0016 ms。截线可写坐标／高度仍需复制，2K／4K／6K 两帧合计约 0.45／0.89／1.34 MiB。剪枝复制 bool 骨架和 float32 距离图，不进行 dtype 降精度转换。

这些是独立暖态诊断，释放返回值位于计时区间之外；缓存和分配寿命与完整调用不同，因此不能相加或从正式总量中相减。正式门槛始终使用包含全部复制／转换开销的第一次配对测量。补充诊断前后核对了正式 JSON 和九个源码／二进制摘要，均未变化。

### 内存

每个场景／后端另起独立进程测量完整服务 RSS，共 98 次探针，未计入正式耗时：

| 集合 | Python 峰值 RSS 中位数 / 最大值 | Native 峰值 RSS 中位数 / 最大值 |
| --- | ---: | ---: |
| 13 个轮廓场景 | 271.03 / 436.56 MiB | 271.09 / 440.72 MiB |
| 36 个快速测径场景 | 149.22 / 218.59 MiB | 148.63 / 229.59 MiB |

这是包含解释器、依赖和准备后输入的进程高水位，不是单个函数的分配量。原生内核减少了特定中间数组，但本组 RSS 数据不支持宣称整体更省内存；高水位零增量也不等于零分配。

## 构建与发行

原生源码指纹：`6ec36eb4c2aa5ff467c9c4efbde9b61e6339338ada993f2435ddc0211e0a0e80`。这是按源码相对路径、分隔符和文件内容汇总的指纹，不是单个 `kernels.cpp` 的文件 SHA256。ABI 为 `cpython-313-darwin`，C++ 标准为 201703。

- [原生自检](native-self-check-macos.json)：三个内核实际运行，输入未修改。
- [包安装检查](package-smoke-macos.json)：默认原生 sdist→wheel；显式 `FDM_BUILD_NATIVE=0` 的纯 Python wheel；缺扩展时 auto 回退、强制 native 报错；损坏扩展在 auto 模式报 ImportError，未静默回退。
- [冻结探针](frozen-result-macos.json)：PyInstaller onedir 成功，`stdin/stdout/stderr=None`；即使请求 `FDM_NATIVE_BACKEND=python` 仍实际使用原生。

构建使用 `build/native-kernel-migration/package-source/` 的隔离源码副本，排除原 egg-info 和已有二进制。`python -m build` 使用隔离构建环境，先产生 sdist，再由 sdist 构建 wheel；构建日志保存在本机 `native-build.log`、`python-build.log`。安装使用独立 `--target installed-native/installed-python`，验证进程使用 `python -I`、`cwd=/tmp`，断言模块来源为对应安装目录；复用现有 `.venv` 的运行依赖，**没有声称是全新运行依赖环境验收**。

已直接检查归档内容：sdist 含 C++ 和许可证且不含二进制；原生和纯 Python wheel 都含许可证，原生 wheel 含 `.so`，纯 Python wheel 不含二进制，见 [归档检查](archive-content-checks.json)。pybind11 声明已同步至第三方说明及资源清单的大小／SHA256。构建前真实导入、指纹检查和微型计算，core/full 的扩展收集、`functional_checks.native_kernels` 及安装器复用 onedir 的门槛均已接入。

macOS 探针只验证最小冻结环境中的内核。尚未产出或验收 Windows `.pyd`、完整 onedir／安装器，也未执行 Windows 安装后 GUI 或真实显微样本计量验收。[构建与 Windows 验收步骤](../../native-kernels.md) 可在目标 Python／MSVC 环境继续执行。

## 恢复参考后端与复现

源码运行时设置 `FDM_NATIVE_BACKEND=python`；清除该变量恢复默认 auto。二进制损坏、ABI／源码不匹配必须修复并重启，不会自动切换算法。冻结发行始终要求原生；恢复旧发行行为应使用已验证的旧安装包。

```sh
FDM_NATIVE_BACKEND=python python -m fdm
FDM_NATIVE_BACKEND=native python -m fdm
python scripts/benchmark_native_kernels.py --output build/native-kernel-migration/paired-performance-new.json --measure-rss
python scripts/benchmark_native_overhead.py --formal-result build/native-kernel-migration/paired-performance-new.json --output build/native-kernel-migration/native-overhead-new.json
```

新测量写入不同文件，保留本次证据。合成性能验证通过不代表所有真实输入都按同倍率加速，计量验收和 Windows 安装验收仍须独立进行。
