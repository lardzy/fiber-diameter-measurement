# UI 截图审查

专业测量工作台提供确定性截图入口：

```bash
./.venv/bin/python -m fdm.ui_snapshot \
  --scenario measurement \
  --theme dark \
  --width 1093 \
  --height 576 \
  --scale 1.25
```

截图默认写入已忽略的 `build/ui-review/`。`width` 和 `height` 是 Qt 逻辑尺寸，`scale` 是设备像素比；例如 Windows `1366×768 @ 125%` 去除系统占用后，可用 `1093×576 --scale 1.25` 复核。

可用场景：

- `empty`：空工作区。
- `measurement`：混合长度、面积和计数结果。
- `measurement-results`：展开底部结果中心。
- `acquisition`：实时采集工作区。
- `digital-slide`：数字切片采集工作区。
- `settings`：首选项窗口。
- `screenshot-annotation`：常规选区的内联标注、对象控制柄、上下文属性栏和完整工具栏。
- `screenshot-annotation-small`：窄选区的“更多”溢出菜单、完成按钮可达性和工具栏翻转。

`settings` 场景可再传 `--settings-page general|measurement|annotation|analysis|area|acquisition|screenshot|export`，用于检查长页面、样式预览和底部固定按钮。

`measurement-results` 可传 `--results-tab records|statistics|distribution`，分别检查记录模型、描述统计和无额外依赖的分布图。

可用主题为 `dark`、`light`、`system`。截图生成后必须实际查看，至少检查：命令可达性、文字裁切、横向滚动条、画布尺寸、侧栏折叠、底部按钮和菜单/分裂按钮基线。

固定人工门禁建议覆盖：

- `1093×576 @ 1.25x`
- `1920×1080 @ 1.0x`
- `1280×720 @ 1.5x`（对应 1920×1080 @ 150% 的逻辑区域）
- `1707×960 @ 1.5x`（对应 2560×1440 @ 150% 的逻辑区域）

非 Windows 平台只作为几何和可达性检查；固定像素差异基线应仅在字体与 DPI 稳定的 Windows CI 环境维护。
