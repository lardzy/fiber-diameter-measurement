# Fiber Diameter Measurement

离线纤维直径测量桌面软件，面向 Windows 10 无独显电脑。V1 聚焦完整测量闭环：多图同时打开、图内/预设标定、手动与半自动吸附测量、项目文件保存、叠加图导出、CSV/Excel 导出。

## 当前能力

- 多标签页同时打开多张图片，支持 `PNG/JPG/BMP/TIF/TIFF`
- 三栏工作区：图片列表、测量画布、标定与测量结果面板
- 测量模式
  - `浏览`：选择已有测量线并拖动端点微调
  - `手动测量`：直接绘制直径线
  - `半自动吸附`：先拉近似直径线，再用局部 ROI + 模型/传统算法寻找纤维边界
  - `比例尺标定`：在图上拉线，输入真实长度与单位
- 标尺来源
  - 图内标定
  - 设备预设标定
- 结果导出
  - 测量线叠加图
  - 比例尺叠加图
  - `image_summary.csv`
  - `fiber_details.csv`
  - `measurement_details.csv`
  - `measurement_export.xlsx`
- 项目文件 `*.fdmproj`：保存图片路径、标定、分组、测量记录与视图状态
- ONNX 模型接口：优先用本地 ONNX 模型做分割推理；模型不可用时自动回退到传统图像算法

## 技术栈

- `Python 3.11+`
- `PySide6`：桌面界面
- `OpenCV + NumPy + ONNX Runtime CPU`：模型推理和图像处理扩展点
- `pandas + openpyxl`：建议的表格处理依赖
- 当前仓库中核心业务逻辑同时保留了纯 Python 可测试实现，因此即使没有装 GUI/推理依赖，也能跑单元测试

## 快速开始

### 1. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```bash
pip install -e .
```

### 3. 启动桌面应用

```bash
python -m fdm
```

或

```bash
fdm
```

## 使用流程

1. 打开一张或多张显微图片。
2. 选择 `比例尺标定`，在图中拉出已知长度的比例尺线并输入真实长度。
3. 如果你有固定倍率，也可以先在右侧新增并应用标定预设。
4. 选择 `手动测量` 或 `半自动吸附`。
5. 半自动模式下先拉一条近似垂直纤维长度方向的线，系统会在局部 ROI 中自动吸附到边界。
6. 切回 `浏览`，拖动已有测量线端点做人工修正。
7. 需要分根统计时，先创建纤维分组，再把测量记录归属到对应分组。
8. 通过 `保存项目` 保存现场，通过 `导出结果` 生成叠加图和表格。

## ONNX 模型约定

- 目标是二分类分割：背景 / 纤维
- 输入建议为单通道灰度图或可转换为单通道的图像
- 输出支持以下形态之一
  - `[1, 1, H, W]`
  - `[1, H, W]`
  - `[H, W]`
- 预测值大于等于 `0.5` 会被视为纤维掩码
- 未加载模型时，软件会退回到局部阈值 + 形态学清理 + 连通域筛选 + 法线求交的纯算法流程

## 测试

当前测试全部基于标准库 `unittest`，不依赖 PySide6。

```bash
python -m unittest discover -s tests
```

覆盖范围：

- 标定换算与项目文件读写
- 旋转 ROI 提取
- 半自动吸附算法在合成纤维图上的边界定位
- CSV / XLSX 导出结构

## 目录结构

```text
src/fdm/
  app.py
  geometry.py
  models.py
  project_io.py
  raster.py
  services/
    export_service.py
    model_provider.py
    snap_service.py
  ui/
    canvas.py
    dialogs.py
    main_window.py
tests/
```

## 后续建议

- 在已有标注数据基础上训练轻量分割模型并导出量化 ONNX
- 为半自动测量增加批量复核面板和快捷键
- 增加单根纤维的多次测量统计视图
- 补 Windows 打包脚本和图标资源
