# 比例尺交互与像素精度改进验证

## 结论

环境为 macOS 27.0 / arm64，Python 3.13.12，Qt 6.10.2。

- 综合回归：**856 passed，51 subtests passed**，见 [完整输出](regression.log)。
- 最后一次轮廓绘制修正后重跑相关模块：**344 passed，17 subtests passed**，见 [最终渲染专项输出](renderer-final-regression.log)。包含 **225 项像素测试**；两轮测试有重叠，不应直接相加。
- 生产渲染子进程自检：原有叠加渲染 **21 个场景**及比例尺 **15 个检查项**均通过，见 [实际 JSON](renderer-self-check-macos.json)。
- 新增模块和所列改动文件关键 Ruff 检查通过，见 [输出](lint.log)；`git diff --check` 通过。整份历史 `main_window.py` 另有 HEAD 已存在的两处 `Mapping` 类型名称未导入，本轮未修改它们。
- **没有运行 Windows 安装程序**。本地结果不替代 Windows 显示缩放、字体与安装后导出的验收。

## 界面与操作

- [编辑界面](editor-macos.png)：右下角入口位于对象捕捉左侧；主按钮直接打开／返回编辑，下拉菜单显示／隐藏／定位。
- [完成编辑后](preview-after-edit-macos.png)：比例尺保留，编辑柄消失，可继续测量。
- [六种样式](styles-macos.png)：端点、上端点、下端点、四等分、纯线、实心条。
- 样式、线宽／实际条高、字号和配色常驻；顶部强调导出与隐藏。移除面板中的测量、标注和水印选框，统一到导出窗口。
- 完成、隐藏、发起导出记忆最近确认设置；取消草稿恢复；重启隐藏；保存失败可重试。
- 取消导出选项、取消路径、渲染失败与成功后均保留编辑面板；直接点击状态按钮也能重开已关闭的属性区。
- 综合导出往返保留选择，新编辑会话恢复默认当前图片；多图时不再强行选全部；明确取消的水印选择不被重新勾选。

## 几何与数字切片

1. 原先居中描边越过长度边界。现统一以左右外边缘跨度表示标签长度，端点向内，线宽不增加长度。
2. 截图检查发现 Qt 路径布尔简化在真实小数坐标下可能删除上端点／四等分的水平条。最终实现直接生成连续闭合轮廓；新增 72 项真实布局用例，不人为替换基线坐标。
3. 整数对齐 100 px 跨度在奇偶线宽下恰好占 100 列；小数跨度按覆盖量验证，并禁止出现区间外的完整像素列。覆盖 DPR 1／1.5／2、不同缩放、字号、位置、原点和六种样式。
4. 数字切片六样式在非零原点（8192, 4096）导出逐列检查完整水平线；改变实时缩放、标定和焦层不改变冻结的输出。等待原生像素时保留遮罩和等待提示，布局不逐帧读磁盘。
5. 固定物理长度切到未标定图后，仅改变线宽／位置／粗体不能静默改为同数值 px；必须明确恢复自动或标定。五种公制单位换算保持一致。

详细规范来源及精度定义见 [调研说明](../../research/scale-bar-controls-and-pixels-2026-09-23.md)。整数像素边界与亚像素覆盖是不同验证条件，不能将抗锯齿后的非零像素个数直接视为物理长度。

## 性能

使用生产布局／绘制函数，在 1280×800 离屏画布上绘制 4K／8K 合成图像。整图／当前视窗 × 拖动／平移／缩放／切图共 16 场景；每组预热 30 次、统计 120 次。最终轮廓修正后的 [原始数据](benchmark-macos.json)：

| 图像 | 含比例尺 P50 | 含比例尺 P95 | 相比仅绘制图像，P50 最大增量 |
|---|---:|---:|---:|
| 3840×2160 | 0.311–0.546 ms | 0.370–0.665 ms | 0.120 ms |
| 7680×4320 | 0.370–0.646 ms | 0.421–0.759 ms | 0.124 ms |

2,048 个不同拖动位置后布局缓存保持 256 项；检查点的 Python 内存约 284 KiB（290832、290928、290992、291120 字节）。源 QImage 内容标识未改变。这里衡量离屏 CPU 绘制，不包含桌面输入延迟、窗口合成、磁盘读取或 Windows 安装环境。

## 打包门禁

`functional_checks.overlay_renderer.scale_overlay` 保留 `revision: 1`。除初版十项外，Windows onedir 脚本要求以下五项为 true，缺失／失败均阻止通过：

- `endpoint_pixels`
- `fractional_span_coverage`
- `style_variants`
- `division_geometry`
- `actual_layout_geometry`

已测试每项缺失和失败的拒绝行为，并保留既有魔棒、水印、快速测径编译后端及画布渲染检查。本目录 JSON 来自 macOS 生产 spawn 渲染探针，不冒充 Windows 包的 `--self-check --json`。

Windows 安装后待验收：100%／150%／200% 显示缩放，中文与回退字体，六种样式，取消导出后继续编辑，偏好重启恢复，普通图片与数字切片焦层／视窗导出。

## 复现最终专项

```bash
uv run --no-sync pytest -q \
  tests/test_scale_overlay_pixels.py tests/test_scale_overlay_editor.py \
  tests/test_overlay_render_self_check.py tests/test_build_windows_onedir.py \
  tests/test_scale_panel_refinements.py tests/test_scale_overlay_control.py

uv run --no-sync python scripts/benchmark_scale_overlay.py \
  --output docs/validation/scale-overlay-2026-09-23/benchmark-macos.json
```
