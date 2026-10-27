# DSX 自动拼接：静态算法分析

更新时间：2026-10-23。分析对象是用户提供的软件副本；没有执行原厂 EXE/DLL、连接设备或修改产品代码。文件身份见 [identity-manifest.json](identity-manifest.json)，函数证据见 [evidence-map.md](evidence-map.md)，重建规格见 [implementation-spec.md](implementation-spec.md)。

## 主要结论

这套软件存在三条不同的拼接实现，不能把某一 DLL 的算法概括为全部 DSX 功能。

| 路径 | 真实入口和选择条件 | 配准 | 位置整合 | 状态 |
|---|---|---|---|---|
| APP 默认 DSX 离线拼接 | `OlycImageStitching.dll` → APP `OlyImageStitching.exe` → `CompositeDLL64::CompositeMatch` | 有重叠先验的灰度相关搜索；2 px 候选步进；2×2 抽样代价；默认 ProcedureType=6 | 从高纹理/亮度区域开始的邻接顺序匹配、父位置累加、低纹理回退 | 已闭合主干；完整边缘 ROI/顺序细节见 gap |
| CTRL 拼接工具 | 托管 `OlyImageStitching.exe` → `OlyiImageStitchingAPI.dll` → matching/stitching/correction DLL | `OlyNmv` Hanning + FFT 相位相关 + 3×3 亚像素拟合 | 指定 relation 父链累计，最终整数坐标 | 已闭合配准和融合；不把实时防抖的 0.35 门限误套到此处 |
| APP 的 OLS 分支 | `IsDSXOutprocessStitching=false` → `OLSStitching/OlyaImageStitching.exe` | 内置相位相关，非 LPF 为 5×5 加权质心；LPF 为 3×3 正弦比模型及振幅拟合 | mask 邻接路由、多边加权全局优化、可选硬件坐标变换/回退 | 已闭合内部主要路径；不是 APP 默认路径；另有未激活 ALMIS 旧实现 |

APP 托管构造器 `0x06000117`、弹窗操作 `0x06000155`、控制器 `0x060001ae` 都把 `IsDSXOutprocessStitching` 设为 true；导出接口 `0x060001ad` 接受调用者参数，`ExecuteOutprocess/0x06000062` 根据字段选择两个 EXE。它证明默认路径及可选分支，不能证明本次现场操作必然使用某条路径。实际仍应记录进程名、配置和日志。

### APP 默认配准不是通用特征点拼接

`CompositeMatch@1800082a0 → InitMemoryComposite@180007340 → ImageMatch@180008bf0` 读取 8/16 位 RGB，转换为灰度并按所有 tile 的共同极值归一到 16 位范围。ProcedureType 4 用相邻像素差平方的边带纹理量；5 用平均亮度；6 用边带最大值减最小值。它们用于选择种子及邻接处理顺序。

`GetShift@18000a420 → GetMatching@1800122a0 → cost@180014140` 以重叠宽度、容差、相机最大角度约束搜索区域，Gaussian 平滑，可选 Laplace 后再平滑。候选位移按 2 px 遍历，误差为：

\[
E=\left|1-\frac{\sum A B}{\sqrt{\sum A^2\sum B^2}}\right|
\]

求和也每隔 2 行、2 列取样，没有减局部均值。全零能量返回 1。这里未见亚像素求解、SIFT/SURF、透视模型或全局 bundle adjustment。

**MinVote 的返回值不是上述相关误差或相关峰置信度。** Ghidra 最初把 `GetMatching` 错判为 void；汇编 `180012d09…180012d51` 明确计算两幅匹配 ROI 的 `(max-min)/65535` 并取较小者，返回 XMM0。`OrderMatch@18000d330` 选择该值最大的方向，严格大于 MinVote 才接受，否则 `LowVoteMatch@18000e450` 按名义重叠回退。默认 MinVote=0.1。证据 `ev_17e3ade3ac47b2f9f202d35fb2ca53c42d05358f18370dce677ae0fe630f1431` 与 `ev_effd0326ec4482e73382c75d5ba20cbc180a15318ec90a6b179bdb7956e4ab08`。

独立合成测试显示这种门限的局限：周期图有 44 个等价最优点，MinVote≈0.763；不相关高对比度图的 MinVote≈0.999，也会通过；正确低对比度图相关误差为零，但 MinVote≈0.010 会被拒绝。它是纹理可用性门限，不能被 FDM 当成“位置正确概率”。

### CTRL 配准的 NMV 与 OLS 相位相关必须分开

CTRL `OlyiImageMatching64@180007f90` 真正调用 NMV 初始化、下采样、Hanning 窗、EstimateMV。其输出第一 double 是可靠性，后二者是取反位移；上层 `180006970` 使用位移累计，未见对该可靠性比较。实时防抖中的 `.35` 和 `15 px` 条件属于另一个调用者，不能沿用。

CTRL `OlyNmv` 的可靠性是归一化峰值，模型详见相邻 [防抖分析](../stabilization/implementation-spec.md)。OLS 的 LPF 分支则拟合三变量 `(dx,dy,amplitude)`，残差确实乘第三变量；返回 amplitude 作为 score。非 LPF 分支用 5×5 质心和窗口总和/(W×H)。**两者虽有相似正弦比公式，但可靠性定义并不一致。**

OLS `132670` 还有真实前置过滤：匹配 ROI 的 99.9 百分位低于 8 位图的 4，或 16 位图的 1024 时，保留先验位置、score=0；峰值门限默认在读到有效 `HWStageXYPosition` 时为 .1，否则 .05，拒绝 `score <= threshold` 和 `score >= 1.2`。这是实现中的启发式门限，不是统计概率。可选 CLAHE 的真实构造也已确认，clip limit=40、8×8 网格，不能认为所有分支必然启用。

### OLS 分支确有全局位置优化

`156060 → 130c30`（普通图）或 `12e080`（平面多层图，`isplanefilm`）收集邻接匹配后，都调用 `124330`。初始化 `138300 → 15f5c0 → StitchRouteSearch` 根据 tile mask 建立遍历、ID 映射和已有邻接边，既有上/下，也有左/右关系。这个 DLL 只处理网格关系，不读取图像。

`124f90/124dc0/124a90` 给出目标、梯度、Hessian，而非仅根据 ALGLIB 字符串猜测：

\[
F(p)=\sum_{(i,j)} w_{ij}\left((p_i-p_j)-d_{ij}\right)^2.
\]

X/Y 分别优化，`130c30` 传入 `w=sqrt(score)`；非正 score 替换为 .05 后再取平方根。初值为硬件或规则网格坐标，原始 ALGLIB LM 停止参数包含 `1e-15` 和最多 10000 次。目标只含相对差，没有绝对锚点项；每个连通分量都有平移自由度，具体 LM 保持何种平移规范不能仅由目标推断。

**零分坏边仍有权重，并没有被删除。** 原型三节点反例中，一条错误距离为 50、score=0 的边，使正确 0/10/20 的两端各偏移约 4.635 px。可借鉴全局约束方法，但接入 FDM 时应显式标记不可靠边、检查环路和连通分量。

另有 `122ec0 → 125310/125190` 的五参数变换：`R(theta) × (diag(sx,sy) p + [tx,ty])`，随机取三个点、最多 `min(C(n,3),1000)` 次，按距离阈值选内点后再拟合。两个点单独走直接拟合；未见包装层先判断共线退化。该实现不能泛称完整仿射/单应变换。

其启用条件也已闭合：硬件坐标文件至少有一个点时，`recalibrationenable` 在优化前用正分边的 `score²` 加权平均夹角旋转零分回退边；`gridmatchingenable` 在优化后启用五参数拟合。两者作用和时机不同，并非默认全开。文件先减最小 X/最大 Y，再除以 1000；具体单位仍需文件生产方或实际数据对照，不能凭除数猜单位。

### 融合主要是边缘距离权重，非已证实的最优接缝

APP 的真实融合是 EXE 内部 `140010a90 → 140013c60 → 140014350 → 140015e60 → 14001cf60/01d320`，不是仅凭 `CompositeImage` 导出认定的另一套混合函数。

其权重 `14001ec10/01ed40` 是到 tile 最近边缘的距离，下限 1，配置类型 0/1/2/3 对应 1/2/3/4 次幂。先取最后覆盖 tile 作为基准；关闭校正、Count=1、单图覆盖或超过内部距离阈值时直接写它；否则向前遍历，按 Count、权重上限与 Adjuster 合成，整数除法截断。

默认 APP 配置 `ColorCorrectEnable=0`，所以不能仅因为存在羽化代码就说当前默认始终平滑接缝。CTRL 配置色彩校正开启、平方距离、Count=4，与 APP 的 Count=100/Type=3 不同。已查主干未见 graph-cut seam 或拉普拉斯金字塔多频段混合。

OLS 普通二维图的真实虚表已解析到 `148530 → 147110 → 14d450 → 146790 → 1459e0 → 017f10/018370`。它也有倒序覆盖和距离权重融合的结构，但单通道融合会排除零值及其权重，三通道零值正常参与。多层类型走另一入口，不能据此宣称全部 OLS 高度/多层融合已复刻。

高度图的拼接先在重叠区域（可排除中央区域）计算父图和当前图平均高度差，再给当前整幅高度图加常数偏移，之后按位置融合；没有把该偏移解释为真实物理表面。它能消除台阶偏置，不能修复倾斜/尺度误差。原型斜坡反例保留 >2 高度单位误差。

## 可以借鉴的部分

1. 把硬件位置当搜索先验，而不是无限制匹配。明确先验、图像修正、最终位置三者。
2. 对平面拼接构建邻接图，X/Y 用可解释的相对位移约束优化；输出每条边残差、连通分量和基准坐标。
3. 把纹理丰富度、峰形质量、位移合理性、环路一致性分成不同证据，避免一个 score 包办所有判断。
4. 色彩和高度分开处理，保留未校正原图/原始高度。显示拼接可以羽化；不能将平滑后的视觉效果作为计量正确性证明。
5. 分块加载和输出限制适合大切片，但 FDM 的完整焦层数据应保留图像—XY/Z—采集身份关联，不能只保留拼接后的单张结果。

## 无设备验证

[reference_prototype.py](reference_prototype.py)、[输入 fixture](fixtures/synthetic-inputs.npz)、[run_reference_tests.py](run_reference_tests.py)、[test-results.json](test-results.json) 同存。固定 seed=20261023/20261024；记录 Python、NumPy、SciPy、OpenCV 版本和源/输入 SHA256；39/39 项通过。包含非相关、周期、低对比度、奇数位移、窗口外位移、断图、零分坏边、倾斜高度、退化标定等反例。

其中两项执行完整 `reconstruct_grid` 路径，而不是只检查模块公式。3×3 场景的 12 条邻接配准全部通过，最大位置误差 0.073707 px，整数原点全部恢复，融合图 MAE=0；输入、输出 NPZ 和 PNG 均在 [fixtures](fixtures/) 中。中心 tile 变黑的反例拒绝其四条边，明确返回独立分量；该孤立图块仍按先验位置、经全画布平移归一化后参与 mosaic。返回 `valid` 只是有图块覆盖的 mask，不是配准可信度；需结合配对接受状态和连通分量使用。主路径采用 OLS 风格配准/全局目标与明确的线性羽化；不模拟原厂所有 cap、分块、文件格式与遍历顺序，详见 [可执行主路径](implementation-spec.md#可直接执行的推荐研究主路径)。

这只验证独立重建的算法性质。原型不是逐位复刻；没有执行原厂程序对拍。整数候选搜索在奇数位移上无法表达真值；5×5 质心在本次亚像素 fixture 上偏差约 0.404 px，LPF 正弦比+振幅模型误差约 0.000045 px。后者不能外推为真实样品精度。

## 剩余工作边界

分类清单见 [gap-ledger.md](gap-ledger.md)：A 为选定核心功能仍缺的算法环节，目前没有；B 为特定旧引擎、其他工作模式或逐位数值兼容；C 为真图/实机对拍。当前已足够实现并运行基本二维自动拼接，仍不足以宣称重建了原厂全部工作模式、文件格式、异常处理和历史数值行为。尤其不能由离线拼接算法推断 DSX 运动完成反馈或新帧保证。

184 条唯一 REA 证据按原始文件身份索引，保存在 [rea-evidence.zip](rea-evidence.zip) 和对应 gzip 中；[decompiled](decompiled/) 提供可读片段，[SHA256SUMS.txt](SHA256SUMS.txt) 覆盖交付文件。证据来自静态分析，不含原厂二进制副本。
