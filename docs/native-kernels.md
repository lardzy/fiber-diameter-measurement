# 原生计算内核

FDM 使用 C++17 / pybind11 加速三个内部计算函数：轮廓像素边界提取、轮廓截线扫描、快速测径的骨架短分支剪枝。Python 继续管理标定、业务、取消、任务和界面。骨架化继续使用 `skimage_zhang`，测量修订号仍为 3；本次不改变项目文件格式。

## 构建

使用运行应用的同一 Python 构建，不跨 CPython 小版本、操作系统或 CPU 架构复制 `.so` / `.pyd`。构建系统采用 pybind11 的[官方 setuptools 接入方式](https://pybind11.readthedocs.io/en/stable/compiling.html#modules-with-setuptools)，pybind11 精确固定为 3.0.4。内核不链接 Qt、OpenCV 或 OpenMP。

开发环境需要 C++17 编译器：macOS 使用 Xcode Command Line Tools，Windows x64 使用带 Windows SDK 的 MSVC Build Tools。

通常可在虚拟环境安装项目并完成构建：

```sh
python -m pip install -e ".[dev,packaging]"
```

已有工作区的 egg-info 包含需保留的改动时，使用以下方式只构建扩展；wheel/sdist 检查放在独立源码副本中进行：

```sh
python -m pip install pybind11==3.0.4 wheel
python setup.py build_ext --inplace --force
```

默认必须成功编译；不忽略编译错误。`FDM_BUILD_NATIVE=0` 用于在干净环境安装纯 Python 开发版本，不删除之前存在的扩展，也不覆盖运行时的后端选择。其他取值报错。

```sh
FDM_BUILD_NATIVE=0 python -m pip install --no-cache-dir .
```

PowerShell 使用 `$env:FDM_BUILD_NATIVE="0"`，需要恢复编译时删除该环境变量。构建禁止 fast-math、浮点融合和 `-march=native`，以保持数值行为与通用 CPU 兼容。

## 后端与故障处理

| `FDM_NATIVE_BACKEND` | 源码运行行为 |
| --- | --- |
| `auto`（默认） | 优先原生，仅扩展本身不存在时使用 Python |
| `python` | 使用保留的 Python 参考实现，供对照与问题隔离 |
| `native` | 必须成功载入原生扩展 |

后端选择在服务调用时读取，便于同一进程进行配对测试。非法值、损坏的二进制、缺失的二进制依赖、ABI／接口版本错误、源码与构建指纹不一致均明确报错。计算过程中的错误或取消不回退重算。

升级 Python 或切换平台后，旧 ABI 后缀的 `_kernels.*.so` / `_kernels.*.pyd` 即使被 Python 导入器忽略，也会阻止 auto 静默回退并提示重新构建。显式 `python` 模式仍可用于源码对照。

冻结应用总是要求原生内核，源码调试环境变量不能绕过此要求；`--self-check --json` 会独立执行真实 C++ 内核。需要在已发布应用恢复旧行为时，应恢复此前经过验证的安装包。

```sh
FDM_NATIVE_BACKEND=python python -m fdm
FDM_NATIVE_BACKEND=native python -m fdm
python -c 'from fdm._native import build_info; print(build_info())'
```

`build_info()` 包含接口版本、编译器、Python ABI、pybind11 版本及 C++ 源文件／头文件的 SHA256。源码工作区首次加载时核对指纹；修改 C++ 后需重新编译并重启进程。wheel 安装后的运行不依赖仓库中存在 C++ 文件。

## 数值与任务契约

- `boundary_segments(mask)` 输出 `float64[E,2,2]`，按上、下、左、右分组，每组内行优先。保留孔洞、分量、像素单元边界和 200 万条边上限；两遍扫描避免多组中间坐标数组。
- `contour_sections(segments, heights, step)` 输出行偏移 `int64[R+1]` 和连续区间 `float64[K,2]`；包装层恢复现有 tuple 返回值。半开区间及 `1e-10 / 1e-12 / 1e-8` 判定与 Python 参考一致。
- `prune_short_spurs(skeleton, distance_map)` 输出独立布尔掩膜，保留八轮上限、排除斜向捷径的邻接关系、批量删除和短交叉整体保护；float32／float64 距离图不降精度。
- 内部扩展接口要求 C 连续且对齐的数组；服务包装层接受切片／翻转输入并最多整理一次。可写数组会复制为独立快照；已不可变且连续的任务快照可以借用，调用方不得从另一别名并发修改它。
- 原生循环释放 GIL；分配 Python 对象、操作回调时持有 GIL。检查点原样传播既有取消、超时异常；Python 的绝对截止时间仍由 Python 检查，工作线程仍负责丢弃过期结果。

Python 参考函数为 `boundary_segments_python`、`_sections_python`、`_prune_short_spurs_python`，供回归和性能对照。不可将原生输出的区间顺序、近零判断或掩膜简化为另一种算法。

## 验证与性能复现

```sh
QT_QPA_PLATFORM=offscreen FDM_NATIVE_BACKEND=native python -m pytest -q tests/test_native_kernels.py tests/test_native_backend.py tests/test_contour_comparison.py tests/test_fiber_quick_geometry.py
QT_QPA_PLATFORM=offscreen FDM_NATIVE_BACKEND=python python -m pytest -q tests/test_contour_comparison.py tests/test_fiber_quick_geometry.py
python scripts/benchmark_native_kernels.py --output build/native-kernel-migration/paired-performance.json --measure-rss
```

Windows GUI 测试使用本机 Qt 平台插件；不要照抄 macOS 的 `QT_QPA_PLATFORM=offscreen` 设置。

基准固定 13 个轮廓场景和 36 个测径场景，默认每个阶段预热五次、交替采样三十对，比较语义后才接受计时。涵盖 2K／4K／6K、旋转、孔洞、多分量、普通纤维及全部 30 个交叉压力输入；五个历史拒绝案例也必须保持拒绝原因。

计时包含 Python 入口、快照转换、原生计算和结果重建。分割、任务排队、Qt 结果安装及屏幕呈现不在这个基准内；合成结果不能代表真实显微样本计量或 Windows 安装验收。内核数倍提速不等于整个应用同倍率提速。

报告保存每对原始计时、P50/P95、输入与源码摘要、构建信息、线程配置和峰值 RSS。集合指标对同一采样序号求和后取分位数，并明确标记为构造的配对集合指标；不是逐场景倍率的平均值。`--measure-rss` 在另外的独立进程测量两种后端的完整服务峰值内存；这些操作不计入正式耗时。

## 发布检查

core/full 均包含扩展及许可证，PyInstaller 显式收集扩展。Windows 构建前执行真实导入、源码指纹核对和微型计算，避免仅凭 `find_spec` 接受坏 DLL。新增 `functional_checks.native_kernels` 独立验证三个函数及输入未变，原来的 scikit-image 检查继续保留。安装器复用 onedir 时也必须通过该检查。

小型冻结探针用于验证无控制台环境、二进制装载和强制原生选择：

```sh
python -m PyInstaller --noconfirm --clean --onedir --name fdm_native_probe --paths src --hidden-import fdm._native._kernels --collect-data fdm._native --workpath build/native-probe/work --distpath build/native-probe/dist --specpath build/native-probe scripts/probe_native_kernels_frozen.py
build/native-probe/dist/fdm_native_probe/fdm_native_probe build/native-probe/result.json
```

Windows 使用相应 `.exe` 路径。该探针不是完整应用安装包。Windows 正式验收还需在目标机器依次执行原生构建、core/full/public onedir、自检、安装器制作／安装后自检及真实交互确认；开发机的 macOS 探针不能替代这些步骤。

在具备目标 Python、MSVC、Inno Setup 和相应发行资源的 Windows 环境，可按以下顺序验收。每次 onedir 构建会更新同一输出目录，应分别保留报告后再进行下一配置；`public` 是 full 配置的发行选项。

```powershell
python -m pip install --no-cache-dir -e ".[dev,packaging]"
python -c "from fdm.services.native_kernels_self_check import run_native_kernels_self_check; print(run_native_kernels_self_check(validate_source=True))"
python -m pytest -q tests/test_native_backend.py tests/test_native_kernels.py tests/test_build_windows_onedir.py tests/test_release_self_check.py
python scripts/build_windows_onedir.py --profile core --strict-asset-hashes
python scripts/build_windows_onedir.py --profile full --strict-asset-hashes
python scripts/build_windows_onedir.py --profile full --public-release --strict-asset-hashes
python scripts/build_windows_installer.py --reuse-onedir --public-release --strict-asset-hashes
```

这里必须使用可编辑安装（`-e`），使原生扩展位于发行脚本实际加载的 `src/fdm/_native` 中。普通 wheel 安装只会把扩展放进安装目录，不能替代源码目录的打包前构建。

在安装后的目录运行应用的 `--self-check --json`，确认 `functional_checks.native_kernels` 四个 case 均为 true，另一个 `fiber_quick_geometry` 检查仍为 `skimage_zhang`、修订号 3。还需实际操作轮廓对比、普通／分叉测径、计算中取消和关闭窗口；将结果与目标 Windows 环境中的 Python 参考路径配对。正式发布沿用项目原有干净工作区、签名和资源要求。

首轮 macOS 实际结果、已知基线失败和原始证据见 [2026-10-05 验证报告](validation/native-kernel-migration-2026-10-05/README.md)。Windows 修复、MSVC 构建及设备验证见 [2026-10-13 验证报告](validation/native-kernel-windows-2026-10-13/README.md)；该设备的完整性能验收仍有两项 P95 退化未通过。
