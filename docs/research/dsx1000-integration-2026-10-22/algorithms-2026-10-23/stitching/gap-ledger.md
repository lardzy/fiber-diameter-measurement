# 拼接复现范围与剩余边界

状态基于 2026-10-23 软件副本。目标是独立重建基本二维自动拼接：已知 tile 邻接和位置先验，完成配准、全局坐标、融合、失败诊断。不是复刻所有 DSX 历史模式或逐位输出。原厂代码的实际启用路径与新原型选择分开记录。

## A：选定核心功能的算法缺口

**当前没有阻断所选核心路径的算法缺口。** `reconstruct_grid` 已实际串联重叠 ROI → Hanning/LPF 相位相关及亚像素拟合 → score 判定 → XY 全局优化 → 画布坐标 → 距离羽化和覆盖 mask；其中错误边保留/剔除可显式选择，连通性可检查。`valid` 仅表示有图块覆盖，不表示配准可信；孤立 tile 仍按先验位置参与 mosaic。完整输入输出、算法选择和例外见 [implementation-spec.md](implementation-spec.md#可直接执行的推荐研究主路径)。

这项判断有两项全路径测试支持：9 张 tile 的 12 条边全部正确配准并输出 mosaic；黑色中心 tile 使 4 条边失败并返回孤立分量。其余 37 项用于模块性质和反例，不冒充全部全路径测试。合成主路径为规则 3×3、同尺寸灰度数组、整数像素位置先验；没有宣称生产级任意布局、超大图、全部色彩和文件格式支持。

## B：其他模式、旧引擎兼容与数值一致性

B-1 表示所涉算法已有直接原文，独立原型没有逐项模拟。B-2 表示该可选范围仍有静态语义未闭合。二者均不阻断上面的选定核心，但若以后要求原厂输出兼容，需要单独补齐。

| 编号 | 类别 | 范围及已有证据 | 当前边界 / 复现影响 |
|---|---|---|---|
| ST-01 | B-1 | APP Composite GetShift 18000a420、GetMatching 1800122a0 完整反编译 | 部分交集、四向 ROI 符号、初始候选相位未全转译；当前 cosine_search 仅完整包含 ROI。选定主路径使用 OLS 风格相位相关。 |
| ST-02 | B-1 | Composite SetMatchOrder、OrderMatch、LowVoteMatch、OrderMatchPosition | 原厂规则网格访问/并列规则未全模拟；当前主路径显式枚举全部四邻接边，不声称等价于默认 APP 父链。 |
| ST-03 | 已闭合 | OLS 默认门限及 recalibration/grid 条件 | HWStageXYPosition 有有效坐标→.1，否则.05；两开关字段和实际条件已写入规格。默认 APP 选择 Composite，OLS 需明确 false 分支，不把可选优化称为默认功能。 |
| ST-04 | B-2 | OLS 平面多层候选：12e080、13bcb0 → 132670，有效面积乘 score | 多层候选优先级和 invalid-height 语义未全部闭合。普通二维 tile 拼接不需要这个层选择模式；完整焦层可保存后逐层沿既定 XY 布局处理。 |
| ST-05 | B-1 | OLS 全局目标/梯度/Hessian、LM 初始化/终止参数 | 独立原型解同一二次目标，采用明确的每连通分量均值规范。原厂奇异目标的 LM 数值规范和历史舍入没有对拍；不影响数学目标的重建。 |
| ST-06 | B-2 | OLS 五参数模型、随机三点、距离内点、两点分支 | 原厂奇异 Hessian、重复随机采样和欠约束两点拟合的完整失败行为未闭合。研究实现明确拒绝共线/不足三点，属于有意改进。 |
| ST-07 | B-1 | APP 8/16 位倒序 blend、1..4 次距离幂、cap 函数001110 | 原型 blend_pixel 接受外部 cap；全路径选无 cap/Count 限制的线性羽化。APP 默认校正关闭，不能称默认必有羽化。 |
| ST-08 | 部分闭合 / B-2 | OLS 普通二维真实虚表→148530→147110→14d450→146790→1459e0→017f10/018370 | 普通融合与单通道零值排除已闭合；018bf0/018d10 权重类型表尚未逐条转译。另一多层入口151e90及全部高度后处理未闭合。当前核心采用已明确的独立距离羽化。 |
| ST-09 | B-2 | Composite 可选 Shading 与 OLS IntensityInfo/HighVoltageSetValue | 所有遮光/曝光校正模型未闭合。当前主路径假设曝光可比；没有宣称消除真实照明差。 |
| ST-10 | B-2 | 分块加载、独立图类型路径及 APP 高度常数校正 | 原厂文件 codec、invalid-height/NaN 约定、超大图分块的全部边界未闭合。研究用显式数组/有效 mask，不生成原厂私有格式。 |
| ST-11 | B-1 | 历史 OpenCV 2.4.9、ALGLIB、MINPACK 的具体调用和二进制身份 | 原型用当前 NumPy/OpenCV/SciPy，带 sinc 奇点 guard。未作逐位兼容或原厂执行对拍。 |

## C：真图与实机对拍

| 编号 | 未完成的验证 | 必要输入与目的 |
|---|---|---|
| ST-12 | 真样品位置精度、误匹配、接缝、计量保持性 | 原始 tile、原厂硬件坐标文件、实际 EXE/SHA/配置、原厂结果；人工标注地标并测残差和高度偏差。合成图的 0.073707 px 不能当真实精度。 |
| ST-13 | DSX 采集与位置/新帧/停止链 | 由整体接入分析和现场 collector 获取状态与时间线；离线算法不能证明电机已停止或曝光对应哪个位置。 |
| ST-14 | 完整焦层应用与样品变化 | 同一 XY 的多层原图及焦层身份，验证统一 XY 布局在深度变化/视差下是否适用。仅有二维 mosaic 不能替代完整 Z 栈。 |

## 原型覆盖合同

| 独立模型 | 已覆盖 | 未覆盖 / 有意差异 |
|---|---|---|
| Composite cosine_search | 2 px 候选/抽样、zero-energy、texture score、严格门限 | 不是完整 GetShift 方向/交集/遍历；不输出原厂文件 |
| global_normalize_gray | shared range、RGB 灰度、8→16 位 | optional Gaussian、旧 OpenCV 舍入；全平输入显式 guard |
| phase_ols | 非 LPF 5×5 质心、LPF 3×3 正弦比+振幅、中心搜索、corner LPF | 不模拟全部外层 CLAHE/百分位过滤；SciPy 和 sinc guard，不作原厂数值等价声明 |
| global_positions | 原始二次目标、sqrt(score) 权重、非正 score 回退.05、分量诊断 | 线性最小二乘；每分量保持初始均值为明确规范，不冒充原厂 LM |
| fit_hardware_transform | 五参数、随机三点、距离内点、精化 | NumPy 随机/SciPy；拒绝共线、少于3点；未实现原厂两点分支 |
| blend_pixel | APP 倒序覆盖、type0..3 权重、count、adjuster、threshold | 调用者给定 cap；不模拟 uint64 溢出；不包含 I/O 与分块 |
| align_height | ROI 平均差、整图常数偏移 | mask 由调用者提供；NaN 显式拒绝；不修正倾斜 |
| pair_from_prior + reconstruct_grid | 实际串联重叠 ROI、OLS 风格 LPF 相位、score gate、全局 XY、整数画布、距离羽化与覆盖 mask | 固定规则邻接、灰度数组、整数先验；不模拟原厂全部访问顺序和文件 I/O；失败边剔除为显式改进，孤立 tile 仍用先验参与 mosaic，mask 不代表配准可信度 |

测试合同：源码、输入和结果同存，每次运行重新计算 SHA，固定 seed。当前 39/39 包含 2 项全路径测试及失败反例。通过表示独立模型符合所述行为，不表示原厂全部分支等价或真实 DSX 计量验收。
