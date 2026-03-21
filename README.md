# Fiber Diameter Measurement

离线纤维直径测量桌面软件，面向 Windows 10 无独显电脑。V1 聚焦完整测量闭环：多图同时打开、图内/预设标定、手动与半自动吸附测量、图片侧车保存标定、项目文件保存、叠加图导出、CSV/Excel 导出。

## 当前能力

- 多标签页同时打开多张图片，支持 `PNG/JPG/BMP/TIF/TIFF`
- 支持打开文件夹中的全部图片，忽略子文件夹
- 三栏工作区：图片列表、测量画布、标定与测量结果面板
- 测量模式
  - `浏览`：选择已有测量线并拖动端点微调
  - `手动测量`：直接绘制直径线
  - `半自动吸附`：先拉近似直径线，再用局部 ROI + 模型/传统算法寻找纤维边界
  - `比例尺标定`：在图上拉线，输入真实长度与单位
- 绘图快捷操作
  - `Shift`：绘制或拖拽时限制为水平/垂直
  - `Ctrl`：吸附到像素中心
  - `Delete/Backspace`：删除当前选中测量
  - `Ctrl+Z / Ctrl+Shift+Z`：撤回 / 重做当前图片中的编辑
- 标尺来源
  - 图内标定
  - 设备预设标定
- 标定会自动保存到图片同目录下的侧车文件 `<图片名>.fdm.json`
- 纤维类别按编号管理，支持 `1-9` 快捷切换当前激活类别；类别可带可选名称和独立颜色
- 结果导出
  - 测量线叠加图
  - 比例尺叠加图
  - 比例尺 `JSON`
  - `image_summary.csv`
  - `fiber_details.csv`
  - `measurement_details.csv`
  - `measurement_export.xlsx`
- 导出入口为下拉菜单，可按类型分别导出，也可直接导出“叠加图 + Excel”
- 项目文件 `*.fdmproj`：保存图片路径、标定快照、分类、测量记录与视图状态
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
7. 在右侧“纤维类别”区域创建类别，类别按数字编号显示；按 `1-9` 或直接点击类别块切换当前激活类别。
8. 新建测量会自动归入当前激活类别；旧测量不会因切换类别而改变。
9. 在测量记录表中可以直接修改任意测量所属的类别，画布选中和表格选中会自动同步。
10. 通过 `保存项目` 保存完整会话，通过导出下拉菜单选择导出叠加图、比例尺文件、Excel 或 CSV。

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
- 标定侧车保存与自动恢复
- 文档级撤回 / 重做
- 旋转 ROI 提取
- 半自动吸附算法在合成纤维图上的边界定位
- CSV / XLSX / 比例尺 JSON 导出结构

## Windows 打包

当前仓库已包含一套适合 `PyInstaller` 非单文件目录打包的脚本，产物可以直接交给 `Inno Setup` 继续制作安装包。

### 1. 安装打包依赖

```bash
pip install -e .[packaging]
```

### 2. 生成 `onedir` 打包目录

Windows PowerShell:

```powershell
python .\scripts\build_windows_onedir.py
```

或直接双击 / 命令行执行：

```bat
scripts\build_windows_onedir.bat
```

### 3. 打包输出位置

脚本会生成：

```text
dist/windows/FiberDiameterMeasurement/
```

这个目录就是后续 `Inno Setup` 的安装源目录。默认会清理旧的 `dist/windows` 和 `build/pyinstaller`；如果你想保留旧构建缓存，可以使用：

```bash
python scripts/build_windows_onedir.py --no-clean
```

如果 Windows 上的已打包程序双击后没有任何反应，可以先构建一个带控制台的诊断版本：

```powershell
python .\scripts\build_windows_onedir.py --console --bootloader-debug
```

然后在终端中启动：

```powershell
.\dist\windows\FiberDiameterMeasurement\FiberDiameterMeasurement.exe
```

若应用在启动早期抛出异常，程序还会把日志写到：

```text
%LOCALAPPDATA%\FiberDiameterMeasurement\logs\startup.log
```

### 4. 相关文件

- `packaging/pyinstaller/fdm_onedir.spec`
- `scripts/build_windows_onedir.py`
- `scripts/build_windows_onedir.bat`
- `packaging/inno-setup/fdm_installer.iss`

这套配置默认使用窗口模式、非单文件目录输出，并显式收集 `onnxruntime` 的运行时二进制与必要模块，避免 `collect_submodules()` 把可选依赖一起扫入后产生无关警告。

### 5. 使用 Inno Setup 生成安装包

仓库中已包含一份安装脚本模板：

```text
packaging/inno-setup/fdm_installer.iss
```

推荐流程：

1. 先运行 `python .\scripts\build_windows_onedir.py`
2. 用 Inno Setup 打开 `packaging\inno-setup\fdm_installer.iss`
3. 按需修改顶部宏，例如 `MyAppVersion`、`MyAppPublisher`
4. 编译后会在 `dist\installer\` 下生成安装包

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
