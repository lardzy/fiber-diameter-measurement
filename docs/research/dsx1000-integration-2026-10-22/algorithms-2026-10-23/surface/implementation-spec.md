# 表面校正可实现规格

适用身份和方法证据见 [findings.md](findings.md)。独立数值核心是 [reference.py](reference.py)，验证入口是 [validate_reference.py](validate_reference.py)，[reference-validation.json](reference-validation.json) 当前记录21项合成检查。没有运行原厂 DLL，因此验证证明算法转述的数学性质，不证明原厂输出逐位一致。

## 数据合同

- 算法工作矩阵：二维 height[row, col]，浮点；x 对应列，y 对应行。本文原型全部显式转为 float64，不复刻每一个 CIL 的 float32 中间截断。
- 原厂 APP 边界：高度实体常为 `UInt16[]`，三点取样辅助方法的均值也转换为 UInt16；工作矩阵/趋势向量是 float。**编码单位**与**物理单位**必须分清。
- 原型 `flatten_profile` / `sphere_from_four` / `subtract_sphere` 允许任意一致单位。`three_point_flatten` 默认启用 encoded UInt16 取样行为；物理单位输入应显式 `quantize_u16=False`。
- 非有限数据不在原型已验证输入域，抛异常。原厂对应方法的 NaN/Inf 处理未通过运行验证；不能将原型保护条件写成原厂的已有保护。
- 原型不会原地改输入；返回新矩阵和系数/掩膜。原厂工作缓冲区原地修改可在将来优化时复现，但需独立管理原始高度副本。

## 算法步骤和参数

| 接口 | 步骤 | 参数及边界 |
|---|---|---|
| `fit_profile` | 一次最小二乘或二次多项式最小二乘；系数递增幂排序 | degree=1/2；样本数>degree；二次求解用 NumPy替代ALGLIB，数学解等价但舍入不保证相同 |
| `flatten_profile` | `H + mean(P)-P`；X逐行广播或Y逐列广播 | profile 必须由调用者提供，不能偷偷把全图平均当作原厂剖面 |
| `width_profile` | 按实际UI规则夹边的宽带平均→UInt16截断 | 全宽水平/全高垂直；width=0按整数行/列采样；没有mask加权 |
| `manual_quadratic` / `flatten_manual` | 三控制点二次插值，减趋势保留均值 | 显式支持原厂近重合X点调整；垂直UI曲线反转后应用 |
| `sample_point_mean` | 以选点为中心，奇数方窗采样，索引越界夹边，求均值 | `quantize_u16=True` 时截断非负且≤65535的均值；不猜测溢出转换行为 |
| `three_point_flatten` | 叉积求法向量→分别扣除中心化X/Y趋势 | 原型默认拒绝 C=0；研究开关可复现原厂 C=0→1 的已观察分支 |
| `offset_three_point_encoded` | 复刻 S2 e8 中动态范围/最小/最大值三分支 | span>65535 时代码为**加** `(span-65535)/2+min`；这不是我们理想化的居中公式，原型保留并单独测试 |
| `encode_rescaled_height` | min≤0 时 offset=1-min；max>65535 时 scale=65535/max；转换UInt16 | 这里不做原厂可选空间裁剪/resize；输出携带 offset/scale。回解误差≤半个编码量化步长 |
| `encode_projected_height` | 左上角ROI→编码→线性resize回原始尺寸 | cropW=max(1,int(W²/W′))（W′>W时）；调用者显式提供投影尺寸 |
| `sphere_from_four` | 四点约束→3×3线性解球心→距离求半径 | 奇异/共面样本原型拒绝，原厂 cvInvert 返回值处理需要注意；不把数值求解成功当成几何可信 |
| `subtract_sphere` | 扣除 `cz ± sqrt(r²-dx²-dy²)` | 凹凸选择明确；原型返回无效域NaN+mask，不模拟原厂所有背景显示模式 |

## 完整 APP 路径与当前原型差异

主几何核已可独立实现。生产复现还必须按来源模式决定下面行为，不能默认省略后声称与原厂完全相同：

1. **空间投影与尺寸。** 倾斜模式通过 `cos(atan(slope*Zpitch/XYpitch))` 推导长度变化，经 S2 `RescaleInclinedHeightData` 裁剪/重采样；颜色与 InvalidData 同步处理。原型现已实现高度ROI编码与resize；调用者还需同步传递颜色/mask和空间映射元数据。
2. **16 位重编码。** `np.rint` 对应原型约定；旧 OpenCV 2.2 `cvConvertScale` 的中间精度、tie rounding 尚未对拍。
3. **剖面生成。** 全宽/全高的线宽平均和边缘规则已闭合并实现；该路径没有mask参数。任意斜线剖面和全部UI控件状态不在原型范围。
4. **球面外围与显示模式。** 已有 CIL 包含 cvNot、cvSet、Z scale 和背景模式；当前原型输出mask，不冒充所有原厂分支。`rendering/resize-selected-cil.txt` 另保存 `ConvertZScale` 方法。
5. **元数据导出。** 已找到 scale 回传至预览Z系数，并核实 SaveToHistory 写入 Operation.ZResolutionFactor 与 FilterOperationData；每种文件格式的最终序列化仍需导出样本对照。参考实现显式返回offset/scale/ROI，避免只有一张16位图。

这些差异限定兼容范围；21项检查通过不代表原厂数据逐位一致或光学测量精度已验证。

## 本机重跑

从仓库根目录运行：

```sh
.venv/bin/python docs/research/dsx1000-integration-2026-10-22/algorithms-2026-10-23/surface/validate_reference.py
```

输入为确定性合成平面、二次曲面、局部台阶、四点球面与退化选点。输出 JSON 记录源码SHA与环境版本，NPZ保存输入和结果。没有设备性能、光学精度或真实表面高度对照。
