# FDM 操作教程 · V2 引导增强版

可编辑的 Remotion 工程，统一制作五支视频：

| 视频                   | 内容                                                   | V2 时长 |
| ---------------------- | ------------------------------------------------------ | ------: |
| 01                     | 重做入门：打开、标定、测量、复核、保存与导出           |    2:27 |
| 02                     | 多图片与项目管理                                       |    1:38 |
| 03                     | 标定与预设复用                                         |    1:55 |
| 04 → 05 → 08 → 06 → 07 | 分类复核、连续测量、快速测径、面积计数、魔棒与同类扩选 |    4:22 |
| 09                     | 比例尺、标注与导出                                     |    1:44 |

另生成约 12:05 的全套连播 MP4，内嵌 42 个章节。V2 的 100 句旁白配置了 176 处引导：目标框、局部放大、连线标记、鼠标光圈、拖动轨迹、逐点编号，以及明确区分点击、输入和观察的动作卡。每句配音之后留有停顿。

## 仓库范围

Git 仅保留动画源码、分镜与文稿、采集/配音/渲染/验证/打包脚本、配置和锁文件，以及 `assets/` 中不能自动生成的原始光学测试图片与来源说明。

以下均为本地生成物，由 `.gitignore` 排除：

| 路径                    | 内容                                                          |
| ----------------------- | ------------------------------------------------------------- |
| `out/`、`dist/`         | MP4、预览、播放目录、交付 ZIP、验证报告、日志和浏览器打包结果 |
| `public/`               | Qt 截图、配音、字体子集、控件坐标、实际导出图片               |
| `narration/`            | 逐句文本及临时音频                                            |
| `demo/`、`demo-series/` | 练习图、项目、标定与导出文件                                  |
| `src/series/data.json`  | 根据配音时长生成的统一时间轴                                  |
| `node_modules/`         | npm 依赖                                                      |

视频、音频和 ZIP 移出 `out/` 后仍会被扩展名规则忽略。V2 输出至 `out/v2/`；旧版 MP4、ZIP 与 `v1-source-snapshot.zip` 保留在 `out/`，不会被 V2 流程覆盖。生成文件不是 Git 中的重建输入。

## 从源码准备素材

完整采集与配音使用 macOS、Node.js、npm、uv、FFmpeg/ffprobe，以及可运行的 FDM Python 环境。配音使用系统 Tingting，语速 200。Remotion 与字体依赖由 `package-lock.json` 锁定。

智能章节使用 FDM 的 Edge SAM 3X 实际推理，需要 `runtime/segment-anything/edge_sam_3x/` 中的 encoder/decoder ONNX。

```sh
# FDM 仓库根目录；已有 Python 环境可以跳过
uv sync

cd tutorials/fdm-basics
npm ci
npm run prepare:assets
npm run lint
npm run dev
```

`prepare:assets` 依次采集真实 Qt 操作、生成中文配音和字幕、解析每个引导目标、计算时间轴、生成字体子集。采集隔离用户设置，仅更新教程生成目录。首次克隆后需先生成素材，再运行 Studio 或类型检查。

Studio 的 `Guided-V2` 包含五个完整 Composition；`Guided-Chapters` 包含 42 个可单独预览的章节。原入门篇已采用统一组件和时间轴。

## 预览、渲染与打包

```sh
# 每个引导目标导出一张预览图
npm run preview:series
npm run validate:guidance

# 生成五集，并检查最终 MP4
npm run render:series
npm run validate:media

# 合成连播版、验证、打包跟练素材与完整交付包
npm run package:series
```

也可仅渲染指定视频：

```sh
npm run render                             # 01
node tools/render_series.mjs FDM-02 FDM-03 # 指定 ID
node tools/render_series.mjs --previews --scenes=01-calibrate,09-annotations
```

支持的 ID：`FDM-01`、`FDM-02`、`FDM-03`、`FDM-Measurements`、`FDM-09`。局部预览会更新预览清单；运行完整覆盖检查前，应重新运行 `npm run preview:series`。渲染脚本默认使用本机 Chrome，可通过 `REMOTION_BROWSER_EXECUTABLE` 指定浏览器路径。

打包脚本核对五个跟练项目的记录数量与相对图片路径，合成连播视频并内嵌章节。视频帧不重新压缩；音轨按各集精确时长连接，避免 AAC 编码填充累积。输出包含六个 MP4、独立 SRT、Caption JSON、五张封面、章节索引、说明、验证报告、跟练 ZIP 与离线播放页。

解压 `out/v2/FDM-全套教程-完整教程包-v2.zip`，打开 `教程目录.html` 即可离线观看，支持章节跳转、播放速度和自动下一集。本机 HTTP 播放：

```sh
uv run --no-sync python tools/serve_tutorials.py
# http://127.0.0.1:3235/v2/教程目录.html
```

服务支持视频跳转所需的字节范围请求。

## 编辑入口

- `tools/basics_storyboard.mjs`：01 入门篇文稿与章节。
- `tools/series_storyboard.mjs`：02、03、合并测量篇、09 的文稿。
- `tools/guidance.mjs`：逐句目标、鼠标动作、前后状态截图切换。优先使用实际控件名与测量对象坐标；必要时使用明确的窗口或原图坐标。
- `src/series/GuidedScreen.tsx`：镜头、目标框、鼠标与轨迹。
- `src/series/Tutorial.tsx`、`GuidedCaptions.tsx`：统一版式、动作卡、章节与字幕。
- `tools/capture_fdm.py`、`capture_series.py`：真实操作、截图、控件坐标与跟练文件。
- `tools/player.html`：交付用离线播放页模板。

修改文稿或目标后，运行 `npm run narration` 和 `npm run font`，再检查类型与预览。配音按文本、语速和音频参数缓存。缺失控件、越界目标会直接阻止时间轴生成，不会静默省略指引。修改布局无需重新生成配音，但仍应重新渲染受影响画面。

界面变化后可单独重采集：

```sh
# 从 FDM 仓库根目录运行
uv run --no-sync python tutorials/fdm-basics/tools/capture_fdm.py
uv run --no-sync python tutorials/fdm-basics/tools/capture_series.py
```

采集元数据记录实际控件、图像坐标转换、测量对象和操作结果。重采集后需核对标记与界面是否仍准确对齐。

## 验证与素材来源

`validate_guidance.py` 核对逐句覆盖、目标和轨迹范围、每步停留时间、字幕对齐、引用资产，以及每个目标的预览帧。`validate_series.py` 对最终成片完整解码，检查帧数、时长、分辨率、帧率、编码和音轨峰值。打包流程另外验证连播版与 ZIP CRC。实际结果写入 `out/v2/*validation.json`；重新渲染后应重新验证。

画面来自 FDM 0.4.7 的真实 Qt 控件，以两倍像素密度采集后编排为分步引导动画。本次使用 macOS 源码界面，未验证 Windows 安装包。

练习图由脚本合成并在画面标明。快速测径和标准魔棒使用 `assets/optical-test.jpg`，保留水印且不添加虚构物理标定；同类扩选使用合成六候选练习图，结果由实际模型和搜索流程产生。教程验证操作流程，不代表显微计量精度验收。

字体子集来自锁定的 `@fontsource-variable/noto-sans-sc`，SIL Open Font License 复制至 `public/fonts/OFL.txt`；渲染无需访问字体网站。Remotion 许可见其[官方说明](https://www.remotion.dev/docs/license/pricing)。
