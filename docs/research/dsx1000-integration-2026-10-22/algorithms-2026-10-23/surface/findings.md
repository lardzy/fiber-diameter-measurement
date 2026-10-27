# DSX 高度图的倾斜、曲面与球面校正

分析日期：2026-10-23。仅静态读取副本；没有加载 CLR、运行原厂程序或连接设备。这里研究的是 APP 对已有高度图的处理，与 CTRL 从焦层重建高度的 EFI 算法分开。下面公式是对实际指令的数学转述，不是原厂源代码。

## 文件与证据

| 代号 | 文件身份 | 完整证据 ID |
|---|---|---|
| S1 | APP/OlycSurfaceCorrectionFilter.dll；1.2.4.44；SHA-256 `af324fc21fdf0999526e36169e38ec31c04a959609ee5f70e190b64c2d25b817`；MVID `0921d07e-4bf1-4822-9070-57dfec5b9446` | `ev_951d2e8f1c32aff2a9b533e814e57a9386c0d3ba51a8c4e67a59a326b9c843bf` |
| S2 | APP/ImageProcessingWrapper.dll；1.5.0.1；SHA-256 `0bac697fb85805b24d20362a0bc03f56989c03c3cebef74a2c54caf1a056ea63`；MVID `cea2d21b-bb71-4eda-a437-c22ecb39acb1` | `ev_95a59d69a896d27afe8d1e8c4ba34dfe1f7c23ae235ddafbb2a8d2d5aa4689f2` |

APP 与 CTRL 的同名 ImageProcessingWrapper.dll 不同，不能互换证据。S2 是 C++/CLI 混合程序集，整体覆盖标记 partial；本文用到的 CIL 方法均有完整方法体、IL 哈希且解码无 issue。底层 `cvAdd/cvSub/cvResize/cvInvert/cvGEMM` 的调用已定位，未另行反编译 OpenCV 的通用运算实现。

原始记录见 [managed-evidence.json.gz](managed-evidence.json.gz)。可读指令见 [surface-selected-cil.txt](surface-selected-cil.txt)、[curve-selected-cil.txt](curve-selected-cil.txt)、[wrapper-selected-cil.txt](wrapper-selected-cil.txt)、[sphere-selected-cil.txt](sphere-selected-cil.txt)。

## 1. 自动倾斜校正：剖面拟合与整幅高度减趋势

**已由方法体证明。** S1 `ExecuteAutoInclinationFilter` (`0x06000086`) 根据 X/Y/双向选择进入 `_CalculateInclinationInXDirection` / Y。X 路径调用 `_CalculateLeastSquare` (`0x0600008b`)，累加 `Σx、Σx²、Σh、Σxh`，求直线 `p(x)=a+bx`：

\[
b=\frac{n\sum xh-\sum x\sum h}{n\sum x^2-(\sum x)^2},\quad
 a=\frac{\sum h\sum x^2-\sum x\sum xh}{n\sum x^2-(\sum x)^2}.
\]

随后生成 `correction[x] = mean(p)-p(x)`。S2 `WImageWrapper.InclineX` (`0x060000e4/e5`) 逐行把该向量加到浮点高度矩阵；Y 路径逐列处理。双向模式是两次一维处理，不能描述成已证明的全图鲁棒二维平面拟合。

调用还根据斜率及像素/高度比例计算 `abs(cos(atan(...)))`，获得投影尺寸，随后经过 `RescaleInclinedHeightData` 和 `ResizeInclinedImageData`。颜色和已有 InvalidData 也进入尺寸处理。这不是只改 3D 视图的相机角度。

**边界：** 此处是传入剖面的普通最小二乘处理；没有在该拟合方法内看到离群值剔除、RANSAC 或置信度加权。后续已补查宽剖面路径：直接读取高度矩阵并平均，该接口没有 InvalidData 参数，不能认为宽剖面已屏蔽无效高度。

## 2. 三点平面校正：邻域平均 → 叉积 → 去斜率

**已由方法体证明。** S1 `0x060000a2/a3/a4/a5` 的调用链是：

1. 在三个选中位置周围取方形邻域平均，越界坐标夹到边界；该辅助方法返回前执行 `conv.u2`。这属于当前高度编码中的取值，不等于物理微米取整。
2. 构造两条三维向量，`Vector.CrossProduct` (`0x0600013c`) 得到法向量 `(A,B,C)`，再计算 `D=-Ax₀-By₀-Cz₀`。
3. 用 `-A/C` 和 `-B/C` 生成以图像中心为参考的 X/Y 趋势向量，调用 S2 的逐行/逐列减法，再统一偏移高度。

`C==0` 时该函数以 1 作分母兜底，不能把这个分支解读成对共线/退化点的几何有效性保证。现场应使用分散、非共线且处在可信背景上的点。

**值得借鉴：** 取邻域平均比单像素点抗噪，但若选区跨越纤维边缘、孔洞或台阶，就会把物体形状混进参考面。FDM 若做此功能，应保存选点、邻域大小、拟合残差和有效掩膜；这些是本次建议，不是本段已发现的原厂功能。

## 3. 自动曲面校正：二次剖面，不是任意三维曲面

**已由方法体证明。** S1 `0x06000097/98` 将多项式阶数固定为 2，传入 3 个系数的拟合请求；`buildpolynomialleastsquares` (`0x060000ad`) 调用 ALGLIB 的 `polynomialfit`，成功后 `polynomialbar2pow`。随后指令明确求值：

\[
p(x)=c_2x^2+c_1x+c_0,\qquad h'(x,y)=h(x,y)+\overline{p}-p(x).
\]

Y 方向同理，可按方向分别执行。手动 `_CalcSpline` (`0x0600009e`) 已进一步归类：它用三个控制点解一个**二次多项式**，没有分段三次样条。若三个Y相同直接输出常量；若X坐标两两间距小于1，按 `x1=x0+1`、`x2=x0+1`、`x2=x1+1` 的顺序修正后求系数。默认控制线 `_CalcSplineX/Y` (`0x0600009a/9b`) 同样是二次最小二乘，再将输出限制到0–65535并转UInt16。它与 EFI 中自然三次样条是两种不同算法。

手动补偿 `_ApplyOffsetOfCurvedSurfaceCompensationManualMode` (`0x060000a0`) 使用曲线均值减曲线，保持趋势均值。垂直曲线在应用前 `Array.Reverse`，因为图中剖面方向与原始行方向相反。不能直接将UI纵剖面控制点当图像从上到下的行坐标。

**工程含义：** 二次趋势扣除能方便观察弯曲基底上的局部起伏，但也会扣掉真实的长尺度形状。对于纤维截面或曲率测量，不能默认先“压平”再当作原始形貌使用。

## 4. 球面校正：四点解球心，逐像素扣除球面

**已由实际算式与调用链证明。** S1 `ExecuteSurfaceOfSphereCompensation` (`0x060000ab`) → S2 `SphereCorrection.CorrectSphereImage` (`0x060001f6/f7`) → `CalculateSphereCenter` (`0x060001f9`) / `CalculateSphereRadius` (`0x060001fa`) → `CreateSphereCorrectedImage` (`0x060001f8`)。

球心计算把四点约束消去半径，构造 3×3 线性系统，实际调用 `cvInvert` 与 `cvGEMM`。半径为选点到球心的欧氏距离。后续像素循环可直接读到：

\[
z_s(x,y)=c_z\pm\sqrt{r^2-(x-c_x)^2-(y-c_y)^2},\qquad h'=h\cdot s_z-z_s.
\]

凹凸标志控制根号项正负；存在球面有效域判定和临时掩膜。有效域外、显示背景和 Z 编码转换还受模式参数控制，本文不把所有分支概括成“无效点统一为零”。这段是几何基底扣除，不能修复采集时遮挡或缺失的真实表面。

## 5. 高度编码和可追溯性

**已证实的具体处理：** S2 `WImageWrapper.RescaleInclinedHeightData` (`0x060000e9`) 把浮点工作矩阵变回 16 位高度数据；若最小值≤0，则加偏移使最小值移至 1；若调整后最大值>65535，则使用 `scale=65535/max` 缩放。返回该 scale，上层保存至 `_dblPreviewImageZPitchFactor`。必要时还裁取 ROI、`cvResize` 回目标尺寸。

因此不能仅拿校正后的 16 位灰度值当作物理高度，也不能直接与校正前逐像素相减。需要一起理解 Z 分辨率、比例因子、零点变化、空间重采样和 InvalidData。`SaveToHistory` (`0x0600011a`) 已证实把 `SCExecutionStates.ZResolutionFactor` 和导出的FilterExecutionState写入 Operation 对象；这闭合了处理历史对象的写入，没有声称完整追到每种文件格式的序列化。

## 补查：剖面与空间重采样

新增可读证据：[profile-selected-cil.txt](profile-selected-cil.txt)、[profile-wrapper-cil.txt](profile-wrapper-cil.txt)，仍使用S1/S2完整成员证据ID。

- S1 `_GetHorzProfileData` (`0x06000105`) / `_GetvertProfileData` (`0x06000106`) 的宽带路径：`start=max(0,int(position)-int(width/2))`，`end=min(last,start+int(width)-1)`，再确保end≥start。顶部被夹边后仍尽量保留整条宽带，底部可能缩窄；不是重复边缘像素的对称窗。
- S2 `ProfileWidthWrapper.CalculateHorizontal/VerticalProfileWidthAverage` (`0x060001b2/1b3`) 逐条累加double，除以实际包含的行/列数，再 `conv.u4 & 65535` 写回UInt16。正常UInt16输入即截断均值；没有有效性图或异常点权重进入这个接口。
- width=0走 `IFilterCommon.GetProfileLineData`。该处为水平/垂直整数线段；任意斜线插值器尚未列入本原型。
- S2 `RescaleInclinedHeightData` 的空间路径是**左上角裁剪**。若投影宽度W′>W，裁剪宽 `max(1,int(W²/W′))`；高度同理。先在ROI上求min/max和编码，再线性resize回W×H，不能描述成自动居中裁剪。
- [独立参考实现](reference.py) 已包含宽剖面、三点二次曲线、垂直反向与该裁剪/重编码路径；[21项检查](reference-validation.json) 已通过。球面外围各UI背景模式、任意斜线剖面、文件格式序列化仍与核心数学复现分开。

## 对 FDM 的建议与验证

以下为设计建议，不是已经实现的功能：

- 把原始焦层、原始重建高度、校正后的高度作为不同数据产品，保留处理历史、物理单位和有效掩膜。
- 自动倾斜拟合优先使用操作者指定的可信背景 ROI，并报告残差；对纤维本体不要自动做曲面扣除。
- 高度工作数据保留浮点物理单位；需要 16 位输出时显式存储 offset/scale，避免用显示编码替代测量坐标。
- 现场同一高度数据分别做“不校正、倾斜校正、曲面校正”，导出前后高度/标定，检查平面斜率、局部台阶高度、横向尺寸和无效区边界。该实验用于确认实际保存语义，不能用外观变平作为精度验证。

