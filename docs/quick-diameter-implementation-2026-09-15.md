# 快速测径：首轮提速与打包验证

日期：2026-09-15。改动基于 `80b8c3bb3ecf1f8f92401e160f981454c8799b63`。

后续更新：短纤维、图像边缘、失败提示和修订号 2 打包自检见[稳定性调整记录](quick-diameter-robustness-2026-09-15.md)。本文保留首轮实现与当时的测量结果。

对应问题：纤维已经高亮，但直径线迟迟不出现。此前的诊断和算法资料保存在 [调研报告](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/quick-diameter-research-2026-09-15.md)。

## 1. 本轮结果

已将几何计算固定到编译版 Zhang 骨架，并修正换用正确骨架后暴露的交叉、端帽和候选方向问题。6 个合成掩膜全部成功；本机重复测量的几何总耗时中位数约为 **0.9–33.1 ms**。

| 合成目标 | 改动前，ms | 当前，ms | 速度比 |
| --- | ---: | ---: | ---: |
| 小矩形 | 71.7 | 0.9 | 78.4× |
| 横向 60 px | 698.9 | 9.6 | 73.1× |
| 斜向 40 px | 803.9 | 18.0 | 44.8× |
| 斜向 120 px，2048×1536 | 1236.2 | 33.1 | 37.3× |
| 弯曲 48 px | 817.1 | 17.2 | 47.5× |
| 十字交叉 | 347.2 | 4.6 | 75.5× |

基线每例 3 次、当前每例 7 次，均报告中位数。两次使用同一组掩膜及预生成轮廓，排除分割模型、首次导入和界面绘制；运行环境为 macOS ARM64、Python 3.13.12、OpenCV 4.13.0.92、NumPy 2.4.3。数据见 [baseline.json](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/quick-diameter-2026-09-15/baseline.json) 和 [implemented.json](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/quick-diameter-2026-09-15/implemented.json)。这些结果不能代表真实显微图像的端到端 p95 或首次点击延时。

十字样本当前返回约 **40 px**。直接替换骨架而未修正后处理时曾返回 180 px，这个错误已纳入回归检查。

## 2. 实现内容

### 编译后端与线程

- 在 `pyproject.toml` 声明 `scikit-image>=0.26,<0.27`，由 `uv.lock` 锁定 scikit-image 0.26.0 及传递依赖。
- 明确调用 `skeletonize(..., method="zhang")`；缺少或损坏组件时报告错误，删除原先慢且连接关系错误的 Python 细化后备实现。
- 进入快速测径工具时启动后台预热；预览与已确认任务的 worker 都使用缓存后的同一实现。预热使用 Qt 队列连接，不在界面线程同步导入。
- 保留预览任务的过期结果抑制和已确认任务的完成机制。

### 测量位置与方向

- 骨架图保留八连通关系，去掉已有正交路径上的冗余对角连接，减少阶梯线被误判成分叉。
- 修剪短于分叉处局部直径的末端分支，避免倾斜端帽产生的短叉干扰测量；不修剪没有分叉的整条纤维。
- 按局部宽度排除交叉邻域，采用分散的空间候选点；这不是严格按弧长等距采样。
- 局部 PCA 只拟合包含候选点的骨架连通分支，降低其它分支改变方向的影响。
- 为靠近端帽的位置降低评分；候选宽度和评分在排序时消除浮点舍入噪声，保留确定性选择。
- 缩放后按实际横纵比例及像素中心关系还原位置和方向，最终仍在原掩膜上寻找端点。
- 边界搜索改为分块向量化，搜索上限覆盖图像对角线；旧版每侧 480 步耗尽时误把内部点当作边界的问题已修复。

几何实现：[fiber_quick_geometry.py](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/services/fiber_quick_geometry.py)。

### 可观察性

结果的 `debug_payload` 增加或保留以下字段：

| 字段 | 含义 |
| --- | --- |
| `skeleton_backend` | 实际几何实现，本轮为 `skimage_zhang` |
| `geometry_queue_ms` | 请求创建到 worker 开始处理的等待时间 |
| `geometry_prepare_ms` | 掩膜处理、区域选择与缩放时间 |
| `geometry_skeleton_ms` | 骨架生成时间；未完成预热时也会包含后端加载 |
| `geometry_candidates_ms` | 骨架修剪、分支处理、候选测量和选择时间 |
| `geometry_ms` | 服务的几何总时间 |
| `roi_size`、`roi_scale_xy` | 工作区域尺寸及映射回原图的比例 |

这些计时不包含结果投递到界面后的等待和实际绘制。

## 3. 打包链处理与真实冻结检查

本轮更新了 `runtime_assets.toml` 的 core/full 依赖要求、依赖版本清单和第三方声明文件的大小/SHA-256。应用和 Inno Setup 的版本均核对为 `0.4.5`，本轮未调整版本号。

新增 [测径服务 PyInstaller 钩子](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/packaging/pyinstaller/hooks/hook-fdm.services.fiber_quick_geometry.py)，正式的三个可执行文件和测径冻结探针共用它：

1. 收集 scikit-image 的 `.pyi` 懒加载文件及 scikit-image / SciPy / lazy-loader 元数据、许可证。
2. 由上游钩子追踪 Cython 扩展及 SciPy 本地库。
3. 补齐 SciPy 的动态 `numpy.fft` / `numpy.linalg` 适配模块，并按已安装目录兼容 `_lib` 和 `_external` 两种布局。

第三项来自实际冻结运行发现的问题：PyInstaller 6.19.0 的 SciPy 钩子仍补充旧路径，而 SciPy 1.18.1 已迁移到 `_external`；未补齐时，冻结程序报 `ModuleNotFoundError: scipy._external.array_api_compat.numpy.fft`。修正后已运行通过。探针自身也补齐了几何数据类型需要的 Qt 依赖。

### 发行包自检

`--self-check --json` 新增 `functional_checks.fiber_quick_geometry`：

- 检查骨架后端名称、scikit-image 版本及是否为本地编译扩展。
- 检查长条骨架的连通性。
- 用真实服务测量直线、斜线和交叉掩膜，验证宽度范围。

[onedir 构建脚本](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/scripts/build_windows_onedir.py) 要求此项通过，否则终止出包。[安装器脚本](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/scripts/build_windows_installer.py) 的复用路径也调用同一检查，旧包缺少这项结果时需要重建。

### 已完成的冻结验证范围

已使用 **macOS ARM64 + PyInstaller 6.19.0 / hooks-contrib 2026.3** 实际构建并运行测径探针，工作目录切换到 `/tmp`，移除 `PYTHONPATH`、`PYTHONHOME`、`VIRTUAL_ENV` 环境变量。检查确认：

- `frozen=true`，编译扩展位于包内部。
- 标准输入、输出、错误流设为 `None` 时，几何探针仍通过。
- 直线、斜线、十字测得约 39 / 24 / 40 px。

原始结果：[frozen_result.json](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/quick-diameter-2026-09-15/frozen_result.json)。探针仅用于验证测径依赖，未包含分割模型与完整产品验收；268 MB 是该 macOS 探针的目录大小，不能推算 Windows 安装器的增量大小。

另外尝试了 SciPy 1.17.1 的独立冻结检查，但 PyPI 元数据下载超时，未完成构建。当前实际运行通过的是 SciPy 1.18.1；锁文件在 Python 3.11 下选择的 1.17.1 仍需随目标环境验证。

**尚未在 Windows 真机生成并安装完整 EXE 安装包。** Windows 本地 DLL、完整窗口渲染、驱动和用户实际慢样本仍需在目标环境验证；不能将本机冻结探针当作这项验收。

## 4. 回归与复现

新增覆盖骨架连通性、依赖缺失、不同尺寸/角度/宽度、交叉方向、宽区域边界搜索及后台预热线程的测试。已有连续点击、提前确认、工具切换与线程关闭测试也纳入检查。

整套测试：**`1 failed, 2845 passed, 1497 subtests passed`**，耗时 388.79 秒。本轮测径、线程、打包与发行自检相关用例通过；不是整套全绿。

唯一失败为 `tests/test_atomic_io.py::test_production_json_serialization_explicitly_rejects_non_finite_values`：扫描到轮廓比较功能中 6 处 JSON 序列化没有显式传入 `allow_nan=False`。已逐文件与改动前 HEAD 比较，下列三个文件内容完全一致，失败与本轮测径改动无关：

- `src/fdm/ui/contour_comparison_dialog.py:957`
- `src/fdm/services/contour_comparison.py:480`（该行包含 3 次调用）
- `scripts/benchmark_contour_comparison.py:66`、`:68`

没有修改这些无关文件。独立关键用例与训练导出集成检查为 `54 passed, 81 subtests passed`；快速测径 UI 场景为 `8 passed`。锁文件一致性检查 `uv lock --check` 和 `git diff --check` 通过。

从仓库根目录复现：

```bash
uv sync --frozen --extra dev --extra packaging --extra area-infer
QT_QPA_PLATFORM=offscreen uv run --no-sync python -B -m pytest tests/test_fiber_quick_geometry.py tests/test_background_task_controllers.py tests/test_build_support.py tests/test_build_windows_onedir.py tests/test_build_windows_installer.py tests/test_release_self_check.py -q -p no:cacheprovider
uv run --no-sync python -B docs/research/quick-diameter-2026-09-15/benchmark.py --output /tmp/fdm-quick-implemented.json --repeats 7
uv run --no-sync python -B -m PyInstaller --noconfirm --clean --distpath /tmp/fdm-quick-frozen/dist --workpath /tmp/fdm-quick-frozen/build docs/research/quick-diameter-2026-09-15/frozen_probe.spec
```

Windows 目标机上，在已有完整运行时资源的 checkout 中执行：

```powershell
uv sync --frozen --extra dev --extra area-infer --extra packaging
uv run --no-sync python scripts/build_windows_installer.py
& .\dist\windows\FiberDiameterMeasurement\FiberDiameterMeasurement.exe --self-check --json
```

公开 checkout 按项目现有说明使用 `--public-release`。安装后还应执行安装目录里的 `--self-check --json`，并以实际慢样本检查首次点击、重复测量、连续补点和提前确认。

## 5. 后续边界

- 当前仍是掩膜决定端点的代表直径，保留像素中心距离口径。例如 50 列前景通常对应 49 px 的中心间距离；本轮没有把它宣称为灰度亚像素精度。
- 原图灰度剖面与外边缘配对属于后续第二阶段，可复用 `SnapService`，尚未接入本轮测径。
- 全尺寸掩膜前处理、完整界面投递/绘制计时，以及提前确认时进一步复用同一计算结果，仍可继续优化。本轮优先解决已经实测占主导的 Python 细化，以及更换骨架后的几何和打包问题。
- 当前验证数据是合成掩膜。真实粘连、多纤维交叉、强弯曲、低对比度和真实标定后的精度，需要用户样本及目标机器补充验证。
