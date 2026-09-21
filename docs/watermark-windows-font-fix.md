# Windows 水印文字自检失败修复

日期：2026-09-21；基于 `4aa12bb43785cffd16b5a1c06fb802cf61446060`。

## 故障证据

用户提供的 Windows 打包报告包含 `functional_checks` → `watermark_renderer`。这是嵌套字段，不是带点号的单个键。

- PNG / JPEG / WebP、Logo 透明度、平铺、缓存、数字切片排除通过。
- `text`、`datetime_text`、`datetime_logo` 在 DPR 1 / 1.5 / 2 下均失败，共 9 项；缓存占用 1,279,648 字节。
- [原始报告相关部分](validation/watermark-windows-fonts-2026-09-21/windows-before.json)保留用户实测结果和版本，省去机器路径与其他模块输出。

原初始化器在所有系统上强制使用 `offscreen`。Qt 6.10.2 的该插件在 Windows 使用 `QFreeTypeFontDatabase`，macOS 使用 CoreText；FreeType 实现从 `QT_QPA_FONTDIR` 或 Qt 库的 `fonts` 目录加载字体，不会自动使用 Windows 原生字体数据库。PySide 的标准发行包没有在该目录附带字体。[平台选择源码](https://github.com/qt/qtbase/blob/v6.10.2/src/plugins/platforms/offscreen/qoffscreenintegration.cpp#L56-L66)、[字体加载源码](https://github.com/qt/qtbase/blob/v6.10.2/src/gui/text/freetype/qfreetypefontdatabase.cpp#L28-L50)、[字体目录源码](https://github.com/qt/qtbase/blob/v6.10.2/src/gui/text/qplatformfontdatabase.cpp#L331-L337)。

在 macOS 独立进程中用 Qt `minimal` 平台构造空字体数据库，重现了原报告全部 22 项检查的通过 / 失败组合，缓存占用也完全相同。新增的字体数据库检查明确返回失败；这将问题定位到离屏运行环境的字体加载。[复现报告](validation/watermark-windows-fonts-2026-09-21/fontless-reproduction-macos.json)。

## 修复

- 统一 Qt 图像绘制初始化：Windows 使用 `windows` 平台，macOS / Linux 沿用 `offscreen`；复用已存在的 GUI application，只创建 QImage，不创建窗口。正常 Windows 字体后端负责系统字体与回退。
- 水印自检、画布自检和生产后台绘图进程共用初始化器，避免主进程与工作进程使用不同的字体后端。
- 水印继续保留全部像素判定，并增加字体数据库检查和诊断信息。后台画布额外验证透明底图上的真实字形，防止把标签背景误当作可见文字。
- 打包失败会列出具体水印检查名。onedir 构建和安装器复用检查都保存格式化 JSON 与 stderr；超时或非 JSON 输出也保留已捕获的内容。报告写入失败会警告，不改变原自检结论。

## 本机验证

macOS、Python 3.13、Qt / PySide6 6.10.2；测试使用合成图片。

| 验证组 | 结果 |
|---|---|
| Qt 初始化、水印、真实后台绘图与故障注入、构建脚本、发布自检 | 155 passed，37 subtests passed，10.97 秒 |
| 数字切片、标注模式、预览等待与图块交接、画布渲染 | 380 passed，10 subtests passed，9.12 秒 |
| 独立进程水印探针 | 23 项通过；顶层窗口数为 0 |
| 无字体环境 | 正确失败；原 9 项文字检查和新增字体数据库检查均阻止通过 |

独立进程结果见 [修复后 macOS 报告](validation/watermark-windows-fonts-2026-09-21/watermark-after-macos.json)。测试还覆盖 Windows 平台选择、覆盖不适合的继承平台配置、复用现有 application、缺失字体、缺失文字、报告带 BOM、旧报告失败明细、超时输出保留与安装器阻止编译。

```sh
uv run --no-sync pytest -q \
  tests/test_qt_raster_runtime.py tests/test_watermark.py \
  tests/test_overlay_render_self_check.py tests/test_overlay_process_renderer.py \
  tests/test_build_windows_onedir.py tests/test_build_windows_installer.py \
  tests/test_release_self_check.py

uv run --no-sync pytest -q \
  tests/test_digital_slide.py tests/test_digital_slide_annotation_display.py \
  tests/test_digital_slide_overlay_lifecycle.py tests/test_digital_slide_count_preview.py \
  tests/test_canvas_overlay_handoff.py tests/test_canvas_render_pipeline.py \
  tests/test_canvas_progressive_overlay.py
```

## Windows 复测

用户提供的报告是修复前的 Windows 实测；以上通过结果是 macOS 开发验证。修复后的 Windows onedir、安装包和安装后行为尚待实机确认。

将本次代码同步到 Windows，使用原有资源选项重新完整构建。例如，与用户报告一致的“排除面积模型、保留内容模板”安装包：

```powershell
uv run --no-sync python scripts/build_windows_installer.py --exclude-area-models
```

默认先重建干净的 onedir，再运行自检；成功后才启动 Inno Setup。检查：

- `build/self-check/packaged-runtime.json` 的 `functional_checks.watermark_renderer.ok` 为 `true`、`failed_cases` 为空、字体数量大于 0、`qt_platform` 为 `windows`。
- `functional_checks.overlay_renderer` 的 `worker_platform` 为 `windows`、`worker_text_visible` 为 `true`。
- 安装后再次执行 `--self-check --json`，检查中文水印与时间附注导出、数字切片预览和标注文字。

如果失败，将上述 JSON 与相邻 `packaged-runtime.stderr.log` 一并用于诊断。报告保存在构建目录，不加入发布清单或安装包。
