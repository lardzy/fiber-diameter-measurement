# 后台保存开发验证（2026-09-26）

## 结论与边界

本次为 macOS 开发验证。已完成后台保存、保存期间编辑、关闭等待、状态显示，以及打包门禁的回归；**未在 Windows 打包程序或安装后的目标机器上运行**。

- 回归：**594 passed，58 subtests passed**，见 [regression.log](regression.log)。
- 真实运行时探针：后台保存、水印渲染、画布叠加渲染全部通过，见 [runtime-self-check.json](runtime-self-check.json)。画布探针包含实际子进程渲染及比例尺检查。
- 新增模块 Ruff 检查和相关文件严重错误检查通过，见 [lint.log](lint.log)；`git diff --check` 通过。主窗口原有两处 `Mapping` 注解未导入的问题不在本轮修改范围。
- 项目 schema、设置格式、版本号不变；未增加第三方依赖。

## 同机前后对照

基线为 `0f774d4` 的归档源码；优化后通过真实 `MainWindow.request_save_project()` 发起，计时直至工作线程完成并在主线程确认。每个稳定场景执行 30 次。

| 真实样本 | 像素类型／尺寸 | 同步基线 P50 / P95 | 后台保存完成 P50 / P95 |
|---|---|---:|---:|
| DSX 拼接彩色图 | RGB8，6651 × 2265 | 2.79 / 3.13 ms | 1.63 / 1.90 ms |
| POIR 拼接强度图 | GRAY16，1016 × 4709 | 3.59 / 4.07 ms | 2.26 / 2.50 ms |
| POIR 拼接彩色图 | RGB8，1016 × 4709 | 2.90 / 3.22 ms | 1.62 / 1.74 ms |

| 场景 | 准备快照 P50 / P95 | 主线程确认 P50 / P95 |
|---|---:|---:|
| DSX 彩色 | 0.062 / 0.077 ms | 0.042 / 0.059 ms |
| POIR 强度 | 0.703 / 0.830 ms | 0.042 / 0.050 ms |
| POIR 彩色 | 0.065 / 0.080 ms | 0.042 / 0.050 ms |

90 次重复保存均为 **0 次原生图像编码、0 次完整资源哈希**。资源内容和修改时间不变，原始设备样本哈希不变。首次保存、另存为、重新打开后保存也记录在原始 JSON 中。

这些小型项目的完成时间改善还包含更轻量的 GUI 更新，以及后台保存不清理可能仍被当前会话引用的旧修订资源；不能将差异全部归因于线程本身。

事件监测使用 5 ms QTimer，并包含请求起止边界。请求短于采样周期时，`max_event_gap_ms` 是请求区间，而非一次真实的定时器超时。真实慢盘的响应仍需目标机验证。

原始数据：[before-macos.json](before-macos.json)、[after-macos.json](after-macos.json)。源码样本只读，项目、设置和导入暂存资源均位于独立临时目录。

### 复现性能检查

```sh
mkdir -p .tmp/save-feedback-baseline-0f774d4
git archive 0f774d4 src/fdm | tar -x -C .tmp/save-feedback-baseline-0f774d4
uv run --no-sync python scripts/benchmark_project_asset_save.py --source-root .tmp/save-feedback-baseline-0f774d4 --label 0f774d4 --output docs/validation/background-save-2026-09-26/before-macos.json
uv run --no-sync python scripts/benchmark_project_asset_save.py --background --label background-save --output docs/validation/background-save-2026-09-26/after-macos.json
```

## 行为与一致性

新增异步测试覆盖：

- 人为阻塞写入约 2 秒，20 ms UI 定时器持续响应，事件间隔小于 250 ms；旋转动画持续运行，测量、标定、标注和缩放可继续。
- 保存点击时的快照与磁盘一致；期间新增修改保持 dirty，撤销回保存点时恢复 clean。
- 连按保存去重、仅保留最后一个待处理快照；之后未再次请求的修改不会自动保存。
- 首次保存取消、写入失败、取消后续队列、另存为失败后仍向原目标重试；磁盘成功但界面确认异常时明确显示“已写入，状态异常”，不假报文件已回滚。
- 新增图片、分析结果更新、Logo 改文字及删除外部 Logo 原文件；新状态不会被旧快照覆盖。
- GRAY8、GRAY16、GRAY32_FLOAT 的后台保存和资源复用；DSX／POIR、RGBA8、复制失败、损坏资源、多个资源的回滚另由现有保存回归覆盖。
- 数字切片备份、保留新的 metadata、关闭图片／软件等待及取消、关闭前保存选择、切换项目、关联文件打开回归。
- 迟到请求和旧工作区结果丢弃；重复保存后 Qt 线程、活动快照和待处理快照均被释放。

运行命令：

```sh
uv run --no-sync pytest -q tests/test_project_background_save.py tests/test_project_raster_asset_reuse.py tests/test_project_export_controllers.py tests/test_models_project_io.py tests/test_project_io_copy_on_write.py tests/test_project_analysis_asset_save.py tests/test_project_analysis_persistence.py tests/test_raster_io.py tests/test_raster_derivation.py tests/test_main_window_image_processing.py tests/test_device_image_import.py tests/test_digital_slide.py tests/test_digital_slide_annotation_display.py tests/test_digital_slide_overlay_lifecycle.py tests/test_watermark.py tests/test_watermark_preferences.py tests/test_p0_project_interaction.py tests/test_atomic_io.py tests/test_build_windows_onedir.py tests/test_release_self_check.py tests/test_associated_file_open.py tests/test_history_and_sidecar.py tests/test_history_state_stamps.py tests/test_canvas_lifecycle.py tests/test_qt_callback_lifecycle.py
```

## 界面与打包

人工检查 Qt 渲染结果：深／浅主题，1600 px 与 1024 px 工具栏，状态组件 DPR 1／1.5／2；文字、图标无重叠，宽窗口显示状态文字，窄窗口保留保存位置的状态图标。

- [深色工具栏](toolbar-dark-1600.png)、[浅色工具栏](toolbar-light-1600.png)
- [深色窄窗口](toolbar-dark-1024.png)、[浅色窄窗口](toolbar-light-1024.png)
- [深色状态](states-dark-dpr1.5.png)、[浅色状态](states-light-dpr1.5.png)

`functional_checks.project_save` 的 7 项必须全部通过：`background_completion`、`event_dispatch`、`snapshot_isolation`、`preserves_newer_edits`、`repeat_save`、`failure_rollback`、`thread_idle`。构建门禁逐项检查，包括字段缺失、假成功和版本不匹配。

Windows 待验收：打包后与安装后分别检查首次保存、普通 Ctrl+S、慢盘写入期间测量／切图、关闭等待及取消、失败重试、中文路径、数字切片与另存为。此次 macOS 结果不能替代这些检查。
