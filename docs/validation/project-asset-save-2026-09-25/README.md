# 项目图像资源增量保存验证

日期：2026-09-25。旧版基线：`12eb1f1`。开发环境：macOS 27 / arm64，Python、PySide6 版本见 JSON 记录。

## 结果

DSX / POIR 的已导入图像在重复保存时复用无损资源。下面每个场景均先保存一次，再连续保存 30 次；计时覆盖真实 `MainWindow.save_project()` 同步调用，包括保存计划、资源阶段、项目 JSON、保存状态和界面状态更新，不包含之后的屏幕重绘。

| 场景 | 尺寸与位深 | 旧版 P50 / P95 | 新版 P50 / P95 | P50 降幅 |
| --- | --- | --- | --- | --- |
| DSX 拼接彩图 | 6651 × 2265，RGB8 | 810.64 / 827.49 ms | 2.90 / 3.51 ms | 99.64% |
| POIR 拼接激光强度 | 1016 × 4709，GRAY16 | 180.93 / 187.22 ms | 3.76 / 4.33 ms | 97.92% |
| POIR 拼接彩图 | 1016 × 4709，RGB8 | 278.16 / 287.49 ms | 3.12 / 3.52 ms | 98.88% |

新版 90 次重复保存中，图像编码、回读解码、像素哈希、资源文件全文哈希和资源复制调用均为 **0**。资源文件的内容、修改时间及原始设备文件内容保持不变。仍执行轻量文件属性检查、目录清理检查和项目 JSON 写入。

首次保存与另存为各测一次（不是 P50/P95）：

| 场景 | 首次保存：旧版 → 新版 | 另存为：旧版 → 新版 | 重开后保存：旧版 → 新版 |
| --- | --- | --- | --- |
| DSX 拼接彩图 | 798.82 → 16.12 ms | 811.82 → 14.98 ms | 815.81 → 3.88 ms |
| POIR 拼接激光强度 | 178.24 → 10.51 ms | 177.32 → 10.55 ms | 181.33 → 4.38 ms |
| POIR 拼接彩图 | 273.06 → 10.96 ms | 274.65 → 10.79 ms | 276.45 → 4.02 ms |

首次保存和另存为复制已验证的文件，并在临时文件上校验复制结果；不再压缩、解码这些图像。导入仍会进行一次无损编码及往返校验，重开项目仍需读取并解码资源。复用记录建立时增加一次文件哈希；它不放到每次保存中执行。

原始记录：[旧版](before-macos.json)、[新版](after-macos.json)。临时项目和设置使用独立临时目录，测试结束后删除。样张原件只读，前后 SHA-256 相同。

## 实现边界

- `RasterAssetReceipt` 记录已验证文件、不可变 `RasterPlane` 的对象身份、编码元数据、文件 SHA-256 和文件属性。它共享已有像素对象，不复制像素或放入项目 JSON。
- 导入、派生图片生成及单图/批量重开时建立记录。读取前后的文件属性必须一致，避免把新文件与旧内存像素关联起来。
- 常规保存先校验文档声明的尺寸、位深，再检查像素对象、元数据和文件属性。测量、标定、标注、水印、比例尺及显示映射均与原始像素分开。
- 文件属性包括设备号、文件标识、字节数、修改时间和变化时间。属性改变时重新读文件校验 SHA-256；内容未变则更新记录。这里没有每次保存都进行全文磁盘完整性扫描。
- 像素或编码元数据变化时执行原有无损编码/回读校验。资源缺失时可从内存重建；既有修订资源损坏时拒绝保存，保留旧项目及错误现场。
- 另存为先校验临时副本，再原子发布资源，最后原子提交项目 JSON。新复用记录只在 JSON 成功后发布；失败删除本次新增文件，旧记录、旧项目、路径和脏状态保持不变。
- 每张打开的图片最多保留一条复用记录，关闭图片、重置工作区时清理。不更改项目 schema、用户设置格式或版本号。
- 普通项目内栅格图像共用这一实现，包括设备导入、派生图及采集图。数字切片继续走原有 SQLite 快照保存流程；本次回归涵盖数字切片保存和显示功能。

## 回归与打包边界

最终结果：**496 passed，35 subtests passed**；静态检查及 `git diff --check` 通过。

[回归日志](regression.log)与[静态检查](lint.log)覆盖：

- GRAY8、GRAY16、GRAY32_FLOAT、RGB8、RGBA8，无损保存与后续复用。
- 连续保存不编码、不扫描像素、不读图像文件；测量、标注、标定、水印及显示修改仍写入项目。
- 改变像素、DPI/ICC 编码元数据后生成新资源；尺寸声明不一致时仍拒绝保存。
- 文件删除后重建、文件属性变更后复核、同字节数损坏检测、读取期间文件被替换。
- 首次保存、另存为、单图/批量重开，删除设备原文件后的项目独立性。
- 复制失败、复制校验失败、既有目标损坏、JSON 失败、多图中途失败及失败后的重试。
- 资源记录随关闭/重置释放；已有项目分析资产、数字切片、图像处理、水印及 Windows 打包脚本回归。

扩大回归发现两处基线已有问题，并在 `12eb1f1` 的隔离源码中复现：[基线日志](baseline-existing-failures.log)。本轮同步修正测试中的旧比例尺颜色期望（与已确认的红色默认值一致），并给旧比例尺基准脚本补上仓库要求的 `allow_nan=False`。没有改变比例尺功能。

新增模块使用静态导入，仅依赖现有栅格服务及 Python 标准库，未增加模型、第三方包或 PyInstaller 配置。上述结果是 macOS 源码开发验证；Windows 安装包、安装后运行、较慢硬盘及网络盘的耗时和外部文件变更检测仍需实机验收。

## 复现

```bash
# 当前实现；样张目录需包含已有 DSX1000 / OLS5000 测试样张
uv run --no-sync python scripts/benchmark_project_asset_save.py \
  --iterations 30 --label incremental-assets --output after-macos.json

# 基线源码放在独立目录，复用同一个 uv 环境与同一份基准脚本
mkdir -p .tmp/asset-save-baseline-12eb1f1
git archive 12eb1f1 src/fdm | tar -x -C .tmp/asset-save-baseline-12eb1f1
uv run --no-sync python scripts/benchmark_project_asset_save.py \
  --source-root .tmp/asset-save-baseline-12eb1f1 \
  --iterations 30 --label 12eb1f1 --output before-macos.json

uv run --no-sync pytest -q \
  tests/test_project_raster_asset_reuse.py tests/test_project_export_controllers.py \
  tests/test_models_project_io.py tests/test_project_io_copy_on_write.py \
  tests/test_project_analysis_asset_save.py tests/test_project_analysis_persistence.py \
  tests/test_raster_io.py tests/test_raster_derivation.py \
  tests/test_main_window_image_processing.py tests/test_device_image_import.py \
  tests/test_digital_slide.py tests/test_digital_slide_annotation_display.py \
  tests/test_watermark.py tests/test_watermark_preferences.py \
  tests/test_p0_project_interaction.py tests/test_atomic_io.py \
  tests/test_build_windows_onedir.py
```
