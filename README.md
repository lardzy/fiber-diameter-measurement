# Fiber Diameter Measurement / 纤维显微测量工作台

Fiber Diameter Measurement 是一款面向显微图像的离线桌面软件，用来完成纤维直径测量、面积分割、类别管理、标尺标定、实时采集和结果导出。它不是通用图片编辑器，而是把“打开图片、标定、测量、复核、保存、导出”这条实验室工作流收在一个本地工作台里。

![工作台总览](docs/readme-assets/workspace-overview.jpg)

## 适合场景

- 测量显微图片中的纤维直径，支持手动画线、连续测量和边缘吸附。
- 统计纤维、孔洞或其它目标区域的面积，支持多边形、自由形状和魔棒分割。
- 按纤维类别整理结果，统一类别颜色，后续继续编辑或复核。
- 从 USB 相机或 Microview 设备实时预览，抓拍当前帧后直接进入项目测量。
- 导出叠加图、比例尺文件、CSV 和 Excel，方便进入报告或后续统计。

## 核心能力

### 图片、项目与标定

- 支持打开单张图片、多张图片、整个文件夹，也可以直接拖入图片、文件夹或 `.fdmproj` 项目文件。
- 多图片以工作区形式管理，每张图片保留自己的标定、类别、测量记录、画布视图和撤回 / 重做历史。
- 支持图内比例尺标定、标定预设、项目统一比例尺和 CU 标尺导入。
- 普通图片会在同目录读写 `*.fdm.json` 标尺侧车文件；完整会话可保存为 `.fdmproj` 项目。

### 测量与标注

- 直径测量：手动线段、连续测量、边缘吸附和快速测径。
- 计数：在图上逐点计数，并随测量记录一起保存。
- 面积测量：多边形面积、自由形状面积、标准魔棒和同类扩选。
- 形状编辑：可移动线段端点、编辑面积外轮廓和内部孔洞。
- 叠加标注：支持文字、矩形、圆形、直线和箭头。

![标定与测量流程](docs/readme-assets/measurement-workflow.jpg)

### 分割与辅助识别

- 标准魔棒基于本地 EdgeSAM / EdgeSAM-3x 模型，支持正负采样点、ROI、孔洞填充和剔除区域。
- 同类扩选可以以一个参考实例为入口，在当前图片中查找相似候选并加入当前类别。
- 快速测径会先分割目标纤维，再异步计算代表直径线；可配置 ROI、边缘剔除和线段修正偏移。
- 面积自动识别可调用独立 worker 和已配置的模型权重，批量生成面积实例与类别结果。

![面积识别与交互分割](docs/readme-assets/area-segmentation.jpg)

### 类别与结果管理

- 可以新增、编辑、合并和删除纤维类别，类别名称与颜色会同步到当前图片和项目全局模板。
- 左侧类别列表显示当前图片数量与项目总数，方便检查每类结果是否完整。
- 右侧测量表展示类别、类型、结果、单位、模式、置信度、状态和 ID。
- 支持删除选中测量、删除指定类别下的测量，或清空当前图片全部测量。

### 实时预览

- 支持通用 USB 相机和 Microview 采集链路。
- 可在预览中抓拍单帧导入项目。
- 支持景深合成，将多帧预览结果融合后作为新图片进入测量流程。
- Microview 预览支持地图构建：移动样品台到相邻视野并保持 20%-40% 重叠，系统会对稳定 tile 做景深合成并拼接可靠地图。

![实时预览与景深合成](docs/readme-assets/live-preview.jpg)

### 导出

- 可导出测量叠加图、比例尺图、测量 + 比例尺叠加图。
- 可导出比例尺 JSON、图片汇总 CSV、纤维种类汇总 CSV、测量明细 CSV。
- 可导出 `纤维测量结果.xlsx`，包含测量明细、图片汇总、纤维种类汇总和导出信息。
- 面积结果会记录内部孔洞面积，便于后续按净面积或孔洞情况复核。

![项目保存与导出](docs/readme-assets/export-output.jpg)

## 推荐工作流

1. 打开图片、文件夹或项目；也可以从实时预览抓拍一帧。
2. 先完成比例尺标定，或应用已有标定预设。
3. 建立需要的纤维类别，并切换到当前要记录的类别。
4. 根据图片质量选择测量方式：手动线段、连续测量、边缘吸附、面积绘制、魔棒分割、同类扩选或快速测径。
5. 在测量表中复核结果、模式、置信度和类别。
6. 保存项目，或导出叠加图、CSV、Excel 和比例尺文件。

## 数据文件

| 文件 | 作用 |
| --- | --- |
| `*.fdm.json` | 图片标尺侧车文件，只保存标定信息。 |
| `*.fdmproj` | 项目文件，保存图片列表、测量记录、类别、标注、视图状态等。 |
| `<project>.assets/` | 项目资产目录，用于保存抓拍导入等项目内图片。 |
| `图片汇总.csv` | 按图片聚合的导出统计。 |
| `纤维种类汇总.csv` | 按类别聚合的导出统计。 |
| `测量明细.csv` | 每条测量记录的明细导出。 |
| `纤维测量结果.xlsx` | 多 sheet Excel 综合导出。 |

## 安装与运行

### 环境要求

- Python `3.11+`
- 推荐 Windows 10 / Windows 11；基础图片测量能力也可在常规桌面 Python 环境运行
- 基础依赖见 `pyproject.toml`

### 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

如果需要面积自动识别，请额外安装推理依赖：

```bash
pip install -e ".[area-infer]"
```

### 启动

```bash
python -m fdm
```

或：

```bash
fdm
```

## 运行时资源

- 魔棒分割模型位于 `runtime/segment-anything/edge_sam/` 和 `runtime/segment-anything/edge_sam_3x/`。
- 面积自动识别使用 `runtime/area-infer/` 中的 worker 与参考引擎，模型权重通常放在 `runtime/area-models/` 并在设置中绑定名称。
- Microview 相关 DLL、驱动和控件位于 `runtime/camera/microview/`，主要面向 Windows 环境。
- 项目打包脚本会从这些目录收集运行时资源，源码运行时也会按相同约定查找。

## 开发与测试

运行测试：

```bash
python -m unittest discover -s tests
```

Windows onedir 打包：

```bash
python scripts/build_windows_onedir.py
```

同步安装器版本号：

```bash
python scripts/build_windows_installer.py --sync-only
```

版本号以 `src/fdm/version.py` 为准，安装器的 `packaging/inno-setup/version.auto.iss` 会由脚本同步生成。

## 注意事项

- 面积自动识别需要额外的 `torch / torchvision` 依赖和可用模型权重；源码仓库不一定包含完整业务权重。
- 如果缺少高精度 EdgeSAM-3x 文件，魔棒分割会回退到标准 EdgeSAM。
- 如果运行环境缺少 Qt Multimedia、相机驱动或 Microview 运行时，实时预览能力会降级或不可用。
- 地图构建首版仅支持 Microview 实时预览；低纹理、重复纹理或重叠不足时会拒绝生成地图，避免导出错误拼接结果。

## 开源协议

本项目采用 GNU General Public License v3.0（GPLv3）发布。使用、复制、修改和分发本项目代码时，请遵守 GPLv3 的相关要求，并在再发布时保留协议与版权信息。
