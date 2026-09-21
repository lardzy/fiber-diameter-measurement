# 普通图片水印

## 使用

打开普通图片后，选择 **图像 → 水印…**。窗口右侧实时预览效果，点击“应用”后写入当前图片的项目状态；“取消”保留原配置。

- 每张图片可选择文字或图片 Logo 一种内容。
- 文字支持多行、字体、粗体、斜体和颜色；Logo 支持 PNG、JPEG、WebP，推荐透明 PNG。
- “单个”提供九宫格定位；“重复平铺”提供水平和垂直间距。两种布局都能设置不透明度、尺寸和旋转。
- 默认右下角、不透明度 25%、旋转 0°，水印宽度为原图短边的 25%，边距为对应图像宽高的 2%。
- 水印宽度按图像短边计算。边距/偏移按原图宽、高计算；靠边对齐时正数向内，居中和平铺时正数向右、向下。平铺间距按旋转后水印宽、高计算，100% 表示间隔一个水印宽度/高度。
- 勾选“应用到全部已打开的普通图片”可批量应用。批量修改可一次撤销、重做，数字切片自动跳过。
- 取消“启用水印”可保留配置但停止显示；“移除水印”删除配置。新打开和处理生成的派生图片默认没有水印。

水印固定在图像坐标中，随图像等比例缩放。已有标注的“图像像素/屏幕自适应”设置不改变水印。测量、标注和比例尺绘制在水印上方。

## 保存与迁移

水印配置随 `.fdmproj` 保存。Logo 导入后转为 PNG，按内容哈希去重，保存到项目对应的 `.assets/watermarks/`。移动项目时一并移动其资源目录；删除最初导入的外部 Logo 文件不会影响已经保存的项目。

另存为会复制所需 Logo 资源。保存失败保留旧项目和已有资产；当前会话保留历史 Logo，撤销后仍可恢复。缺失或损坏的 Logo 会提示重新选择，包含水印的导出会明确报错。

没有水印的旧项目保持原有格式内容。含水印配置的项目使用 `image-watermark/v1` 必需功能标记，旧版读取器沿用其只读兼容流程。

## 导出

| 场景 | 行为 |
|---|---|
| 测量图、比例尺图、组合叠加图 | 在导出选项中选择“包含水印”；目标普通图片有已启用水印时默认选择 |
| 导出当前图像 → 当前显示效果 | 可选择“包含水印”，用于单独导出水印图片 |
| 导出当前图像 → 原始像素 | 不包含水印，保留原有位深和像素值 |
| Excel、CSV、比例尺 JSON、训练数据集 | 不加入水印 |
| 数字切片及其视窗导出 | 不加入水印 |

完整分辨率、整图屏显比例和当前视窗导出采用相同布局规则；视窗只裁剪图像，不重新定位水印。结果图导出计划会冻结水印配置及 Logo 内容标识。

水印属于显示叠加层。原始像素、图像处理输入、快速测径、魔棒来源图像和测量坐标保持不变。

## 开发与验证

统一渲染入口位于 `fdm.ui.watermark_rendering`。普通画布底图绘制、局部背景恢复和结果图导出复用它；渲染入口再次排除数字切片。绘制使用 SourceOver，PNG 自带 Alpha 与配置不透明度各生效一次。

水印块和解码 Logo 使用总计 32 MiB 的 LRU。平移复用已解码 Logo 和水印块，平铺使用重复纹理；超过缓存预算的稀疏纹理改为绘制可见水印块，大倍率水印则直接裁剪绘制文字或 Logo。画布不生成原图大小的水印蒙层。

```sh
uv run --no-sync pytest -q tests/test_watermark.py
uv run --no-sync python -m fdm.ui.watermark_self_check
uv run --no-sync python scripts/benchmark_watermark.py --output .tmp/watermark/benchmark.json
```

打包程序的 `--self-check --json` 新增 `functional_checks.watermark_renderer`，检查 PNG/JPEG/WebP 解码、中文文字、透明合成、平铺、缓存复用和数字切片排除，覆盖 DPR 1、1.5、2。Windows 构建脚本拒绝缺失或失败的探针结果。

基准测试将关闭水印和启用水印的绘制作为同机对照，测量 4K/8K 合成图片在 1280×800 视窗中的 CPU 绘制时间。它不代表 Windows 安装包、真实显微图像或桌面合成器的交互延迟。实测记录见 [watermark-validation.md](watermark-validation.md)。

## 设计参考

- [digiKam 水印工具](https://docs.digikam.org/en/batch_queue/watermark_tool.html)：文字/图片、相对尺寸、重复排列和批量应用。
- [darktable 水印模块](https://docs.darktable.org/usermanual/development/en/module-reference/processing-modules/watermark/)：相对于图像的缩放、对齐和偏移。
- [ImageMagick 水印示例](https://usage.imagemagick.org/annotating/)：透明合成与平铺。
- [Qt QPainter](https://doc.qt.io/qt-6/qpainter.html#composition-modes)：SourceOver 和预乘 Alpha 渲染。
