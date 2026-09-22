# 比例尺画布编辑验证记录

## 环境与结论

- 开发环境：macOS 27.0 / arm64，Python 3.13.12，Qt 6.10.2。
- 主要回归：**671 passed，51 subtests passed**；另补充 **2 passed**。合计覆盖 673 项测试，以及 51 项子测试。
- 真实子进程渲染自检通过：原有叠加渲染 21 个场景，新增比例尺 10 个场景；spawn 工作进程使用关闭标准输入输出的窗口程序启动条件。
- Windows 构建门禁的通过、缺失及失败报告均已测试。**没有生成或运行 Windows 安装包，Windows 实机验收待完成。**

主要记录：

- [主要回归输出](regression.log)
- [补充兼容测试](additional-regression.log)
- [真实渲染自检 JSON](renderer-self-check-macos.json)
- [性能原始数据](benchmark-macos.json)
- [编辑界面截图](editor-macos.png)
- [完成编辑后保留显示](preview-after-edit-macos.png)

## 功能覆盖

- 自动长度／字号、五种公制单位的 25 种换算组合、未标定 px、超长或过大文字拒绝。
- 四角与相对位置、文字宽于尺条时的边界约束、整条拖动及长度手柄、局部撤销／重做。
- 全部／当前显示、后续新增图片、完成编辑后继续使用测量工具、隐藏／再次开启、启动默认隐藏。
- 不修改项目脏状态、测量历史或旧比例尺锚点；旧设置初值与设置文件往返。
- 导出取消、写入失败、成功后保存偏好、重启恢复；综合导出与画布编辑往返保留内容选择。
- 导出计划冻结标定、缩放、平移、输出尺寸和布局；批量自动长度分别计算，失败图片统一列出。
- 大坐标下主画布与原始分辨率导出像素对照，DPR 1／1.5／2，原图与视窗平移。
- 数字切片非零原点、焦层冻结、缩放变化、旧焦层等待遮罩、原始视窗范围、预览布局不逐帧读取存储。
- 既有快速测径、魔棒、水印、测量对象、数字切片文字、画布缓存、释放资源及导出服务回归。

## 性能

`scripts/benchmark_scale_overlay.py` 使用 4K／8K 合成图像，在 1280×800 的 QImage 上调用生产布局和绘制函数。覆盖整图／当前视窗 × 拖动／平移／缩放／切图，合计 16 组；每组预热 30 次后统计 120 次。基线为相同图像绘制但不含比例尺。

| 图像 | 含比例尺 P50 范围 | 含比例尺 P95 范围 | 各场景增加的 P50 最大值 |
|---|---:|---:|---:|
| 3840×2160 | 0.276–0.488 ms | 0.329–0.589 ms | 0.071 ms |
| 7680×4320 | 0.341–0.580 ms | 0.398–0.691 ms | 0.108 ms |

另执行 2,048 个不同拖动位置检查淘汰：缓存稳定在 **256 项**，被跟踪的 Python 内存约 **285 KiB**；512／1024／1536／2048 次检查分别为 291296／291392／291456／291520 字节，检查记录本身也包含在计数中。稳定场景的预热后进程峰值增量为 0–48 KiB。所有源 QImage 的内容标识保持不变。

这些数字衡量合成图像的 Qt 离屏 CPU 重绘，不包含完整桌面输入延迟、编辑面板刷新、窗口合成、数字切片磁盘读取或 Windows 安装环境。布局缓存有固定上限；生产控制器仅请求比例尺新旧边界的局部重绘，数字切片等待状态使用现有异步原始视窗请求。

## 打包自检

`functional_checks.overlay_renderer.scale_overlay` 报告包含 `ok`、`revision: 1` 和以下十项：

- `unit_nm`、`unit_um`、`unit_mm`、`unit_cm`、`unit_m`
- `preview_export_dpr_1`、`preview_export_dpr_1.5`、`preview_export_dpr_2`
- `uncalibrated_px`、`oversized_rejected`

比例尺探针检查 40 px 字号没有被截断、实际颜色像素存在，以及普通图片／大坐标数字切片／平移视窗使用同一局部渲染结果。Windows onedir 脚本要求所有子项通过，同时继续检查原叠加渲染、快速测径编译后端、魔棒和水印自检。

本目录 JSON 来自在 macOS 上直接调用生产 `run_overlay_render_self_check()`，不是 Windows 安装包 `--self-check --json` 的替代报告。

## 复现

```bash
uv run --no-sync pytest -q \
  tests/test_scale_overlay_editor.py tests/test_ui_canvas_and_export.py \
  tests/test_settings_dialog_navigation.py tests/test_export_service.py \
  tests/test_project_export_controllers.py tests/test_digital_slide_annotation_display.py \
  tests/test_canvas_overlay_handoff.py tests/test_watermark.py \
  tests/test_watermark_preferences.py tests/test_overlay_render_self_check.py \
  tests/test_build_windows_onedir.py tests/test_release_self_check.py

uv run --no-sync python scripts/benchmark_scale_overlay.py \
  --output docs/validation/scale-overlay-2026-09-22/benchmark-macos.json
```

Windows 安装后仍需核对不同显示缩放、中文字体回退、当前／全部图片、数字切片切焦与平移、保存路径失败重试，以及重启后的偏好和默认隐藏状态。
