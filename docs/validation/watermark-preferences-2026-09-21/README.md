# 水印记忆与初始参数验证

日期：2026-09-21。macOS、Python 3.13、Qt / PySide6 6.10.2；基于当前工作区，包含此前尚未提交的 Windows 字体自检修复。

## 用户确认的行为

- 首次打开使用截图参数：`GTTC`、系统默认字体、粗体、非斜体、`#666666`、启用日期时间、重复平铺、75% 不透明度、25% 宽度、45° 旋转、两向 2% 偏移、两向 25% 间距；单个模式位置初值为右下。
- 应用成功后记住本次参数与 Logo。尚无水印的图片载入上次参数；已有水印读取文档配置。新图片仍需确认应用。
- 用户明确选择：新图片继承日期时间开关，具体值使用本次当前时间。已有图片的固定时间不变。
- 取消、移除和撤销不覆盖最后确认的参数；批量范围每次默认当前图片。

## 验证结果

| 范围 | 结果 |
|---|---|
| 水印、项目 / 设置保存、通用设置窗口 | 161 passed，22 subtests passed，4.71 秒 |
| 最终专项：水印记忆、原水印功能、Windows 构建门禁、发布自检、Qt 初始化、数字切片标注与叠加层生命周期 | 209 passed，37 subtests passed，7.08 秒 |
| 真实水印探针 | 24 项全部通过，包含新增 `preferences_roundtrip` |

两组测试有重叠，不能相加为独立测试总数。记忆专项包含真实窗口关闭 / 重新创建、原 Logo 删除后的应用、设置保存失败回滚、旧 Logo 清理与文档资源隔离、无效设置隔离、缺失 Logo 提示、批量应用及撤销、取消和移除不覆盖记忆、数字切片入口隔离。

- [初始参数快照](initial-values.json)：日期值是截图生成时的本机时间，产品使用当前时间。
- [初始窗口截图](initial-dialog.png)：已检查文字、字形、布局、透明度、尺寸和旋转；两向偏移与间距通过参数快照及 UI 测试核对。
- [真实水印自检 JSON](watermark-self-check-macos.json)：仅使用临时目录保存测试参数与 Logo，不改动实际用户配置。

复现最终专项：

```sh
uv run --no-sync pytest -q \
  tests/test_watermark_preferences.py tests/test_watermark.py \
  tests/test_build_windows_onedir.py tests/test_build_windows_installer.py \
  tests/test_release_self_check.py tests/test_qt_raster_runtime.py \
  tests/test_digital_slide_annotation_display.py tests/test_digital_slide_overlay_lifecycle.py
```

Windows 安装包仍需重新构建后实机复测。构建门禁现要求 `functional_checks.watermark_renderer.cases.preferences_roundtrip` 为 `true`。安装后应检查应用 Logo → 关闭程序 → 删除外部 Logo 原文件 → 重启 → 新图片再次应用，以及新图片时间刷新和旧项目时间保持固定。
