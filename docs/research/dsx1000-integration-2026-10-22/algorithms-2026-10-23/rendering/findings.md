# DSX 的 3D 显示：高度场网格与视觉增强

2026-10-23 静态分析。此模块不负责从焦层恢复高度；它接收已有高度图，生成用于观察和交互的 3D 网格。这个分工对复现十分关键：好看的 3D 表面并不能反证每个高度值都可靠。

## 证据身份

- APP `OlycImageControl.dll` SHA-256 `82180eddd2ab0ef88cd744b7e9af26162356d932c2037ed2dd3d7f8018ab07fa`，完整成员证据 `ev_3d4eafea998ffcddb9c26de4668733e0390c5d1267223ecec9dc200575a7ffd7`。
- APP `ImageProcessingWrapper.dll` SHA-256 `0bac697fb85805b24d20362a0bc03f56989c03c3cebef74a2c54caf1a056ea63`，完整成员证据 `ev_95a59d69a896d27afe8d1e8c4ba34dfe1f7c23ae235ddafbb2a8d2d5aa4689f2`。
- UI设置程序集 `OlycThreeDSettings.dll` 的成员证据 `ev_71cca03e4f4d8348b4504fbfe76405918dca9acef87b066fa9d7241c4b265599`；不能仅凭此UI把设置视为真实渲染行为，以下以 ImageControl 实际方法体为准。

完整证据见 [managed-evidence.json.gz](managed-evidence.json.gz)；调用和算式分别在 [mesh-selected-cil.txt](mesh-selected-cil.txt)、[view-selected-cil.txt](view-selected-cil.txt)、[resize-selected-cil.txt](resize-selected-cil.txt)。配置文件中 `ThreeDAutoDepthFactor=4.0` 的文件身份与行锚点见 [config-anchors.json](config-anchors.json)。

## 已闭合的显示流水线

```text
ThreeDImageControlModel._Display3DImage
  → BL3DManager / Display3D
  → _CreateImageArrays：高度/颜色/强度/无效标志重采样
  → _MakeHeight2DArray：物理比例、显示范围裁剪
  → _MakeIndicesFor3DObject / _PrepareVertex
  → _CreateVertexfor3DObject + 法线
  → world × rotation × translation / view / projection
  → SlimDX.Direct3D9 的顶点与索引缓冲区
  → DrawIndexedPrimitives
```

### 1. 显示网格与采集分辨率分离

`_CreateImageArrays` (`0x06000269`) 将长边设置为 **1024**，另一边按宽高比截断为整数，最小10；不是“仅大图缩小”，小图也会放大。`_SetImageWidtAndHeight` (`0x06000204`) 再把尺寸向下调整为当前网格间距的整数倍。

高度、彩色、强度分别经 Wrapper `ResizeArray`，InvalidData 经 `ResizeArray1`。已核查三类 wrapper 的实际 `cvResize` 参数都是1，对应线性插值调用；原型采用当前 OpenCV 的 `INTER_LINEAR`，未证明旧版与新版每个像素舍入完全相同。显示网格加密不会增加实际高度采样精度。

### 2. 高度比例、截断与翻转

`_MakeHeight2DArray` (`0x060001fb`) 使用：

\[
r=1024/\max(W_0,H_0),\quad
s=0.2r/p_x,\quad
h_d=\mathrm{clamp}[(u\,p_z\,c_{maker}\,c_{user}-h_{min})s,\ 0,\ (h_{max}-h_{min})s].
\]

其中 u 是16位高度码，p_x/p_z 来自图像实体分辨率；maker/user 是Z校正值。具体常量在 CIL 是浮点展开值 `0.20000000298023224`。这是显示坐标换算；物理测量不能使用已经裁剪和夸张的 h_d。

`_CreateVertexfor3DObject` (`0x06000276`) 的高度显示路径生成 `(row×0.2, col×0.2, -h_d×depthFactor)`，Reverse 通过切换 depthFactor 正负实现。显示XY与通常图像 x=col/y=row 的约定不同，导出给其他渲染器时必须明确映射。

### 3. 自动高度夸张确有实现

`_GetAutoDepthFactor` (`0x060001f3`) 读取配置 `ThreeDAutoDepthFactor`，缺省/解析失败用4。其主要算式为：

\[
f_{auto}=\frac{\max(W_0,H_0)p_x}{4(h_{max}-h_{min})}.
\]

属性写入前四舍五入到一位小数并经过float转换，函数本身返回未舍入值。实际还受自动模式选择和后续高度系数消费路径影响。该因子让高度范围相对横向尺寸更明显，不能解释为真实比例。

方法中图像边长还有 `conv.i2`，超大图/零高度范围的表现不能按正常输入推断；原型明确拒绝超出已支持域的数据。

### 4. 三角形、法线和颜色

`_InitializeIndicesFor3DObjectDisplay` (`0x06000205`) 为网格构造三角形索引；实际缓冲区长度为 `floor(6*(W-1)*(H-1)/(strideX*strideY))`，包含边界补写逻辑，不能简单认定长度恰等于理想采样单元数乘6。主要面次序为 `(右下,右上,左上)` 与 `(右下,左上,左下)`，法线函数使用反向叉积。`_DrawObject` (`0x0600021a`) 用索引长度/3作为图元数提交。它是规则高度场表面，每个XY只有一个高度，不能表示多重遮挡层、纤维内部体积或悬垂结构。

`_CalculateSurfaceNormal` (`0x0600026d`) 实际计算三角形边向量的叉积并归一化；`_CalculateAndApplyVertexNormal` (`0x0600026b`) 按角点、边界、内部顶点分别累加邻接面的**单位法线**，再归一化。高度夸张改变后有重新计算法线的路径。这会影响光照观感，不能用亮暗推断原始曲率。

顶点颜色可来自图像颜色通道或高度LUT，并有最高/最低高度着色分支。这里不是根据纹理生成新的几何。

### 5. 无效高度不是自动挖孔

本版本 `_IsPixelValid` (`0x0600026a`) 的完整方法只有 `ldc.i4.1; ret`。法线计算确实调用它，因此不能仅凭名称声称该计算已经排除无效像素。顶点构造另外读取 InvalidData，并在相关显示/粗糙度状态下设置颜色；这与删除无效三角面不同。

该结论限定于本版本这些方法。没有声称 DSX 所有高度处理都不使用有效性信息。复现科学可视化时可另加“无效点留孔”，但应标为我们自己的改进，与原厂完整网格显示模式区分。

## 独立实现与覆盖

[reference.py](reference.py) 提供显示尺寸、自动夸张、显示高度换算、轴向映射、物理高度场网格和 PLY 导出。[validate_reference.py](validate_reference.py) 的 [15项数学检查](reference-validation.json) 已通过，生成 [reference-plane.ply](reference-plane.ply)。不加载原厂或Direct3D。

当前原型正常网格几何与单位换算有明确实现，顶点法线采用可移植的相邻三角面单位法线求和。[实现规格](implementation-spec.md) 明确了紧凑网格与原厂边界缓冲区的差别；[交互演示](height-field-lab.html) 已通过浏览器检查。相机/灯光初值与实际消费路径已在下面补查。未做原厂截图或逐像素着色对拍，也不复刻全部拾取/UI状态。

## 补查：灯光、网格默认值与相机

依据 [settings-selected-cil.txt](settings-selected-cil.txt)、[transform-selected-cil.txt](transform-selected-cil.txt)：

- 常规非 contour 分支的网格X/Y间隔为2；contour分支使用 `LineInterval`，还经过 contour专用调整。Display3D构造器的LineInterval初值4，不等于所有图像都用4。
- `Display3D` 的构造初值含高度系数1、允许范围0.1–50、zoom范围0.2–5、初始旋转X=35°/Y=60°、灯光强度4。保存设置和UI可覆写，现场值不能直接从构造初值认定。
- `_LightSetup` 将强度等级映射成 `d=clamp((level-1)/9,0,1)`；ambient=`0.2+0.8*(1-d)`，diffuse=`d`。方向取灯光旋转矩阵与物体旋转矩阵乘积的第三行。已观察到实际 SlimDX Light/RenderState 调用。
- 世界变换先把物体中心移到原点附近，再转 Z=π、X=π/2；后续乘用户Euler旋转与平移矩阵。View使用 `LookAtLH`，Projection使用 `PerspectiveLH`，根据窗口宽高比保持模型比例。
- `CreateGrayScaleLUTArray` 确实被顶点颜色分支调用；同类中虽然有 `CreateRainbowLUTArray`，在本程序集完整 call_edges 中没有找到对该方法的调用。当前高度彩色分支从 `ILUT.GetCurrentLUT` 取得LUT，因此不把“存在彩虹方法”当作实际配色路径。
