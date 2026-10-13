# Windows 原生内核修复与设备验证（2026-10-13）

本轮修复了审阅发现的两项问题，并在用户指定的 Parallels Windows 11 虚拟机中同步迁移分支、构建 MSVC 扩展和运行双后端回归。最终源码的 298 项专项测试、261 个子测试通过，49 个正式性能场景的计算结果全部一致。

**Windows 性能验收仍有两项 P95 退化未通过。** 轮廓、剪枝和复杂分叉集合的提速目标已通过，但不能将本轮标记为全部验收完成。Python 参考路径继续保留。本文的数据来自 Windows ARM64 上的 x64 Python，不代表物理 x64 设备或 Windows ARM64 原生 Python 的结果。

## 分支与现场保护

- Windows 项目：`C:\Users\lishuyang\Downloads\fiber-diameter-measurement`。
- 原分支为 `main`，提交 `12eb1f108e1e0e8bee675eab0940fd43ee7935bd`；已切换到 `lard/native-kernel-migration`，基于 `1b343c7c19b90adec162e36e03f20192e02865de` 加本次修复。
- 迁移提交尚未推送，因此采用已校验 Git bundle 同步本地提交，再同步本次修复，没有依赖远端旧分支，也没有推送或合并。
- 同步前备份了 Windows 的六个 egg-info 修改文件、四个私有模板，以及 Git 状态、暂存区和工作区 patch。可编辑安装重新生成的 egg-info 已恢复原字节。
- macOS 原有 UI、测试、IDE、egg-info、研究文件和暂存内容均保留；本次没有提交这些文件。
- Windows 原有 10 个文件和暂存区的最终摘要核对见 [现场保护记录](workspace-preservation-windows.json)。两端修复代码按 LF 规范化后相同，C++ 源码的原始字节相同，见 [源码摘要](final-source-hashes.json)。
- Windows 备份位于 `build\native-windows-validation\before-sync\`；旧 onedir、安装器和 PyInstaller 工作目录分别保留为 `dist\windows-before-native-20261013`、`dist\installer-before-native-20261013`、`build\pyinstaller-before-native-20261013`。

## 修复内容

1. **旧 ABI 扩展不能伪装成“未安装”。** Python 导入器会忽略其他 Python 版本或平台后缀的二进制。`fdm._native` 现在在精确的扩展缺失异常后检查 `_kernels.*.so/.pyd`，发现遗留二进制便明确报错并提示重建。真正缺失仍允许 auto 回退；显式 python 仍用于源码对照。测试使用真实导入器，覆盖 macOS、Linux、Windows 的旧后缀。
2. **修正 Windows 构建说明。** 使用 `pip install -e ".[dev,packaging]"`，确保 `.pyd` 位于发行脚本实际优先导入的 `src/fdm/_native`。普通 wheel 安装不能替代源码树的原生构建。
3. **修复 MSVC 下的剪枝验证开销。** 首次正式 Windows 运行只得到约 1.10 倍剪枝提速。将每像素的浮点转换／`std::isfinite` 改为用 `memcpy` 读取 IEEE 浮点存储的指数位。仍检查整幅距离图并拒绝所有 NaN/Inf，不降低精度；新增 float32/float64 的 NaN、正负无穷、负零、次正规数和最大有限值测试。继续使用 `/fp:strict`，未改剪枝规则、轮数、阈值、取消或截止时间。
4. **修正跨平台测试夹具。** 路径期望值改用 `Path` 构建；完整测径对照在计时前执行生产工作线程已有的 scikit-image 预热，避免 Windows 首次依赖导入触发 3 秒计算超时。生产超时规则未放宽。

该 Windows 检出还存在既有 Git CRLF 转换，导致 26 个资源与清单中固定的大小／SHA256 不符。仅在换回 LF 后同时精确满足清单大小和摘要时恢复字节，并保存原件；没有修改清单或关闭哈希检查。恢复记录见 `asset-line-ending-fixes.json`，原件位于 VM 的 `build\native-windows-validation\asset-line-endings-before\`。后续 Windows 检出仍需保持这些资源的规范字节；本次没有改变 Git 的全局换行配置。

## 环境与正确性

| 项目 | 本轮环境／结果 |
| --- | --- |
| 客体系统 | Windows 11 Pro 10.0.26200，ARM64，4 vCPU、16 GB 内存 |
| 目标 Python | CPython 3.12.13 x64，`win-amd64`；扩展 `_kernels.cp312-win_amd64.pyd` |
| 编译器 | Visual Studio Build Tools 2022 17.14.41；MSVC 1944，工具 14.44.35207，SDK 10.0.26100.0 |
| 依赖 | NumPy 2.4.3、OpenCV 4.13.0.92、scikit-image 0.26.0、PySide6 6.10.2、pybind11 3.0.4、PyInstaller 6.19.0 |
| Windows 最终原生专项 | **298 tests、261 subtests 通过**，80.33 秒 |
| Windows Python 参考专项 | **83 tests、138 subtests 通过** |
| macOS 最终内核／服务回归 | **139 tests、138 subtests 通过** |
| 正式性能输入语义检查 | **49 场景、111 个服务／内核阶段全部一致**，保留五个既有拒绝样本 |

最终原生源码指纹为 `91c68f11c1e9afd98a1ff0cd3053475d7c7e9481fc9c44078c4fa98996e7d415`。这是按源码路径和内容汇总的指纹，不是单个 C++ 文件的 SHA256。接口版本为 1；Windows ABI 标识为 `cpython-312`。[源码自检](native-self-check.json)、[冻结探针](frozen-result-windows.json)及 [wheel 检查](wheel-check.json)均核对此版本并实际运行三个内核。

初次 Windows 回归出现冷启动超时、路径斜杠和资源 CRLF 夹具问题，修正后全部专项通过。没有修改算法使合理拒绝的输入通过。测试证据见 [Windows 最终日志](windows-final-tests.log)、[Windows JUnit](windows-native-optimized-final.xml)、[初次／参考后端日志](windows-initial-and-reference-tests.log)和 [macOS 最终回归](macos-final-tests.log)。此次未重跑整个项目全量测试；此前 macOS 全量测试的五项基线 UI 布局失败仍按 [首轮报告](../native-kernel-migration-2026-10-05/README.md) 记录。

## Windows 正式性能

固定原迁移的 49 个输入与验收集合，每后端／每阶段预热 5 次，再交替执行 30 对采样。采样期间没有并行测试或构建。完整服务计时包括分派、输入快照、内核和返回值恢复；输入摘要、源码／二进制摘要、依赖版本、线程配置和每对原始耗时都在 JSON 中。OpenCV 使用 4 线程，新增内核不建立线程池。

| 固定集合 | Python P50 / P95（ms） | Native P50 / P95（ms） | 结果 |
| --- | ---: | ---: | --- |
| 2K／4K／6K 正常轮廓完整对比 | 1116.263 / 1141.100 | 313.259 / 334.741 | **3.563 / 3.409 倍**，通过 ≥1.5 倍 |
| 全部 36 场景剪枝入口 | 51.111 / 51.565 | 5.439 / 5.580 | **9.396 / 9.242 倍**，通过 ≥3 倍 |
| 固定复杂分叉 9 场景完整测径 | 88.374 / 90.110 | 69.209 / 70.212 | **耗时降低 21.69% / 22.08%**，通过 ≥10% |

集合统计按相同采样编号对各固定场景耗时求和，再取 P50/P95；不是单次完整批处理计时，也不是各场景提速比的平均值。

完整服务退化门槛仍为 `max(5%, 0.5ms)`，未放宽：

| 未通过项 | Python P95（ms） | Native P95（ms） | 允许的 Native 上限（ms） |
| --- | ---: | ---: | ---: |
| `quick_horizontal_60px` | 23.467 | 24.777 | 24.640 |
| `quick_diagonal_40px` | 38.785 | 43.711 | 40.724 |

49 个场景的 P50 均通过，47 个场景的 P95 通过。上述两项内核独立计时仍明显加速，但完整服务出现尾部慢样本；本次证据不能确定是内核外开销还是虚拟化／调度波动，不能据此豁免 P95 门槛。

保留全部测量顺序与失败结果：

- [第一次完整正式运行](paired-performance-windows.json)：旧指纹 `6ec36e…`，剪枝约 1.10 倍，未通过。
- [修复后的 36 场景诊断](quick-performance-optimized.json)：剪枝约 9.62 倍，测径场景退化检查通过；因未运行轮廓而整体为 `incomplete`，不能代替完整验收。
- [最终 49 场景正式运行](paired-performance-windows-final.json)：三个集合提速门槛通过，但上表两项 P95 失败，整体保留 `failed`。

没有为筛选好成绩覆盖或反复重跑正式结果。后续需独立复核这两项完整服务尾延迟，Windows 性能验收才可关闭。此处是合成几何输入，未包含分割、模型推理、队列等待、画面呈现，不构成真实显微样本计量验收。

Windows 运行没有额外拆分复制／转换成本，完整入口计时已包含它们；现有 RSS 探针依赖 Unix `resource`，在 Windows 返回 null，因此不报告 Windows 内存收益。首轮 macOS 的独立开销和内存数据仍只适用于其原测量环境。

## 构建与安装验证

默认原生的隔离 sdist→wheel 构建成功，wheel 含一个 `win_amd64` 扩展和 pybind11 许可证，sdist 含 C++ 源码且不含二进制。独立 `--target` 安装后用 `python -I` 断言模块来源并实际运行内核，四个 case 均通过。该安装复用现有虚拟环境的运行依赖，没有宣称全新依赖环境验收。

最小 PyInstaller onedir 探针通过；即使设置 `FDM_NATIVE_BACKEND=python`，冻结程序仍实际执行原生内核。探针将标准流设为 None，保持窗口应用条件。

| 构建配置 | 严格资源哈希 | 实际程序自检 | 原生四项检查 |
| --- | --- | --- | --- |
| core | 通过 | [通过](core-packaged-runtime.json) | 全部 true |
| full | 通过 | [通过](full-packaged-runtime.json) | 全部 true |
| full + public | 通过 | [通过](public-packaged-runtime.json) | 全部 true |

三个报告均为 `ok=true`、`errors=[]`、`warnings=[]`。原有 `fiber_quick_geometry` 自检仍为 `skimage_zhang`、修订号 3。public 按现有政策排除 `area-models` 和 `content-templates`；core/full 非 public 产物保留四个原有私有模板，仅留在用户的本地虚拟机中。

安装器通过了 [复用 onedir 自检](installer-reuse-runtime.json)，随后实际安装到独立目录。安装返回 0，无需重启；日志未出现创建快捷方式的操作，任务选择为空，没有启用文件关联。[安装结果](installation-result.json)、[安装日志摘录](installation-log-excerpt.log)、[安装登记](installed-registration.json)记录了本次安装。

安装后由当前 Windows 用户运行 `.exe --self-check --json`，结果为 **`ok=true`、`errors=[]`、`warnings=[]`**，见 [安装后完整自检](installed-runtime.json)。即使父进程设置 `FDM_NATIVE_BACKEND=python`，报告仍为 native，四个 case 全部 true。安装后的 build ID 与 public onedir、安装器复用自检一致：`b7a876415af244f1b6823b149ef6437d`。

| 产物 | Windows 位置 |
| --- | --- |
| 安装器 | `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\dist\installer\fiber-diameter-measurement-setup-0.4.7-public.exe` |
| 本次安装目录 | `C:\Users\lishuyang\Downloads\fdm-native-validation-installed-20261013` |
| public onedir | `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\dist\windows\FiberDiameterMeasurement` |
| core 对照包 | `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\dist\windows-core-native-20261013\FiberDiameterMeasurement` |
| full 对照包 | `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\dist\windows-full-native-20261013\FiberDiameterMeasurement` |

安装器大小 **397,297,072 字节**，SHA256 为 `2e0c3c3d9882c30271e5532154845176f290b78d6773562a85bf5d4a8782a013`。四份程序目录中的 `.pyd` 均为 194,560 字节，SHA256 相同：`63f910f02cf6849413eb320ad1a4c050ee142fe1185a4921d126ec2088096368`。全部 27 项产物摘要见 [artifact-hashes.json](artifact-hashes.json)。

安装参数按 [Inno Setup 官方说明](https://jrsoftware.org/ishelp/topic_setupcmdline.htm) 设置：独立 `/DIR`、`/TASKS=""`、`/NOICONS`、`/NORESTART` 和 `/NOCLOSEAPPLICATIONS`。测试安装保留在虚拟机，需移除时运行该目录的 `unins000.exe`。此前的 onedir／安装器备份仍保留，未被覆盖。

本轮是含未提交修复的设备验证构建，未启用要求干净工作区的 `--strict-release`，没有降低资源哈希或功能自检门槛。未签名、未发布；没有在不含开发工具的干净 Windows 镜像、物理 x64 Windows、ARM64 Python 上验收，也未执行安装后人工逐项 GUI 操作、显微相机或真实样本计量验收。Windows 自动化轮廓 UI／工作线程回归和安装后功能自检已经通过，不能据此替代上述边界。

归档 JSON/XML 保留原始数据；`.log` 统一编码为 UTF-8 并清理行尾空白，安装日志文件明确为摘录。完整构建／安装日志仍在 Mac 和 VM 的 `build/native-windows-validation/`。

## 复现与参考路径

在 Windows 项目根目录，以其目标解释器执行：

```powershell
.venv\Scripts\python.exe -m pip install --no-cache-dir -e ".[dev,packaging]"
.venv\Scripts\python.exe -m pytest -q tests/test_native_backend.py tests/test_native_kernels.py tests/test_native_benchmark.py tests/test_contour_comparison.py tests/test_contour_comparison_ui.py tests/test_contour_comparison_wand.py tests/test_fiber_quick_geometry.py tests/test_cancellation.py tests/test_build_windows_onedir.py tests/test_build_windows_installer.py tests/test_build_support.py tests/test_release_self_check.py
.venv\Scripts\python.exe scripts\benchmark_native_kernels.py --output build\native-windows-validation\new-independent-run.json
```

本轮未提交或推送修复，所有源码改动保留在迁移分支供审阅。未作签名发布，也未将测试安装包认定为正式发行。

新性能运行必须写入新文件，并在没有并行测试／构建的条件下进行。源码对照可设置 `$env:FDM_NATIVE_BACKEND='python'`；用 `Remove-Item Env:FDM_NATIVE_BACKEND` 恢复默认 auto。冻结应用始终要求原生，不能靠该变量绕过发行门槛；需要回到旧发行行为时使用先前经过验证的安装包。
