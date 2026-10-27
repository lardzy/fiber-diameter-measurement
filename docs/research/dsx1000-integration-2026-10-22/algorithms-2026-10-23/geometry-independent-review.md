# 几何、高度合同与显示模块的独立审查

审查日期：2026-10-23。审查人：拼接分析子任务，未参与这三个参考实现的编写。范围为 `surface/reference.py`、`rendering/reference.py`、`height-contract/reference.py`、对应 `validate_reference.py`、规格/结论和保存的关键 CIL。未改动这三个目录的源码、测试或结果；所有执行均在 `/private/tmp/dsx-geometry-review-x_eir20s/` 副本。

## 审查结论

初始检查 **44/44 通过**（surface 21、rendering 12、height-contract 11），全部修复后独立复验 **49/49 通过**（21+15+13）。未发现所审查正常输入域中核心公式、坐标方向或数量级与所保存 CIL 相反的情况。报告明确区分编码高度、物理高度、显示高度以及原厂兼容/研究改进，这一点成立。

初审发现**跨模块合同问题 R1**、大间隔输入问题R2和非有限标定参数问题R3；作者已修复并由本审查在新临时副本独立验证，**三项全部关闭**。未发现阻断所选正常核心流程的未解决算法问题。下面保留原始复现与修复记录，不将修复前结果混入最终状态。

## 发现

### R1 · 中，已修复并复验：球面 NaN 与网格有效性合同

位置：[surface/reference.py](surface/reference.py#L213) 的 `subtract_sphere`；[rendering/reference.py](rendering/reference.py#L55) 的 `grid_mesh`，尤其原始快照第63行。

前者明确返回无效域 NaN 和 valid mask，后者在读取 valid 之前要求整幅高度全部 finite。因此即便传 `valid=mask, drop_invalid=True`，仍抛 `ValueError: finite height grid and positive pitches required`。目前网格测试中的无效点保留有限高度，没有覆盖两模块组合。

独立复现：

```python
y, x = np.indices((5, 5))
z, valid = surface.subtract_sphere(np.full((5, 5), 10.), x, y, [2, 2, 0], 1.5)
rendering.grid_mesh(z, valid=valid, drop_invalid=True)  # 当前抛 ValueError
```

该例有9个有效点，可以形成8个合法三角面；使用 `np.where(valid,z,0)` 后再调用，所得8面所有顶点都在 valid 中。建议选择并明确一种合同：

- mesh 在 `drop_invalid=True` 时仅要求有效点 finite，对被丢弃顶点采取明确的占位/压缩策略；或者
- 增加显式 adapter，填充无效高度供存储但保留 mask 并删除所有关联三角面。

需要新增跨模块测试：校正输出含 NaN、有效面和法线有限、面不引用无效点。此问题是独立参考 API 的组合缺口，不是发现原厂具备同样缺陷。

**修复复验：** 新版仅在 `valid+drop_invalid=True` 时允许无效位置非有限；有效位置仍必须 finite，无效顶点用0作未被面引用的占位，不改输入。重新执行以上真实 `subtract_sphere→grid_mesh` 组合，9个有效点产生8个面，所有面顶点有效、所有输出顶点和法线有限、原始NaN矩阵未改变。另有“NaN错误标为valid”的拒绝测试。R1关闭。

### R2 · 低，已修复并复验：大间隔产生零尺寸

位置：[rendering/reference.py](rendering/reference.py#L23) 的 `display_dimensions` 与 `prepare_display`。

`display_dimensions(100,2,32,32)` 返回 `(1024,0)`。对2×100图调用 `prepare_display(..., interval=32)` 会进入 OpenCV 后触发 resize 的尺寸断言，而不是报告输入超出支持域。该问题不影响默认间隔2的常规图像，但接口当前仅检查“正数”，规格没有给出“间隔不得超过归一短边”等约束。

建议明确拒绝不能形成有效采样网格的尺寸/间隔组合，或选择至少一个/两个采样点的研究规则并注明与原厂对齐不同；新增狭长图/大间隔的反例。没有要求复刻原厂异常路径。

**修复复验：** `display_dimensions` 现在对对齐后≤0的尺寸明确抛 ValueError。2×100图、interval=32 的原复现已在进入OpenCV前被拒绝；新增大间隔检查通过。R2关闭。

### R3 · 低，已修复并复验：部分标定参数没有有限性校验

位置：[height-contract/reference.py](height-contract/reference.py#L9) 与 [height-contract/reference.py](height-contract/reference.py#L40)。

`lens_cutoff(np.nan,20,1)` 返回 NaN；`map_nominal_to_absolute([1,2],0,np.nan,1,1)` 返回 NaN 数组。深度数组本身有 finite guard，但首点、标尺、光学参数的 guard 不完整。当前测试都是有限正常标定，所以这不是现有断言失败，也不是原厂行为的错误转述。

建议在接入层统一验证标定值、方向、首点及单位标识。只要报告仍限定正常有限标定输入，这不构成核心算法阻断。

**修复复验：** `lens_cutoff` 现对全部4个光学参数检查 finite；绝对坐标映射对 nominal_first、absolute_first、scale_to_absolute 检查 finite。新临时副本原测试13/13通过；另外逐个参数注入 NaN、+Inf、−Inf，共21个独立探针全部被 ValueError 拒绝。原来的两个复现均不再传播NaN，正常值公式未改变。R3关闭。

## 公式与证据核对

| 项目 | 独立核对 | 判断 |
|---|---|---|
| 自动线性/二次剖面校正 | S1 8b/ad/97 的一维最小二乘与 `mean(fit)-fit`；工作矩阵按轴广播 | 符合；没有将其误称全图鲁棒平面拟合 |
| 三点平面 | 邻域夹边、取均值后 UInt16 截断、叉积斜率、中心化趋势、C=0分支 | 正常输入符合；原型拒绝退化默认与原厂兜底明确区分 |
| 三点手动曲线 | S1 9e 顺序调整相近X，二次插值；a0垂直反向 | 结论与原文对应；名称含 Spline 未被误解成自然三次样条 |
| 高度重编码 | S2 e9 偏移到最小1、max限65535的scale；投影ROI为左上角、先编码再resize | 正常范围符合；offset/scale保留，不把校正码当原始微米 |
| 球面 | S2 1f9线性球心系统、1f8凹凸符号/球面域 | 符合；补测凹球半径100，球心误差1.1e-13，半径误差8.6e-14，凹凸标志正确 |
| 显示尺度 | ImageControl 1fb 的 `0.2/(xPitch/r)`、Zpitch×maker×user、min/max裁剪 | 公式和符号符合；物理mesh与显示mesh分离恰当 |
| 显示轴与法线 | 276的row→X、col→Y、负Z；单位面法线累加 | 研究网格手性/紧凑索引与原厂边界布局差异已明确，不把理想网格称原厂缓冲区对拍 |
| 自动夸张 | 1f3 长边×xPitch/(k×高度范围)，属性舍入 | 正常域符合；conv.i2溢出和零范围没有被无根据地解释成支持 |
| SG编码 | DaoHelper bf3：`float(pitch)/1000/(65535//count)`；Filter 1cc再×1000、倒数乘depth、饱和、conv.u2→Int32元素 | Python scalar顺序符合；dtype边界和量化误差断言有效 |
| outgoing Z scalar | ProcessJoin `_ImageResolutionForImageProcess` 对 helper先conv.r8再×1000×1000 | `wire_resolution` 使用double乘法符合原文；和 buffer深度步长相差约1000是事实，不等于证明两者都用微米 |
| 绝对位置映射 | `absolute_first + direction*(depth-nominal_first)*scale` | 明确标为新合同，不是伪装成原厂方法；仍需首点/符号/单位的现场证据 |

高度合同的量化测试证明有明确单位边界，不证明 reconstruction depth 就是电机绝对 Z。这里没有发现静默把 SG 数值单位当微米的实现；保持 `wire_resolution` 这个中性命名是必要的。

## 测试支持到什么程度

- surface 的平面、二次基底和局部台阶以独立解析真值构造，能检测斜率/轴向/局部形状误差；球面并非只有同函数往返。三点退化、宽带边界和空间ROI有针对性检查。
- rendering 的平面法线用物理导数 `[-1,-1,1]` 对照、mask模式和索引范围，足以支持正常规则高度场与显示公式；并未覆盖 NaN有效性合同、所有相机/光照或旧GPU输出。
- height-contract 的 dtype、边界饱和、单位倍率和量化步长检查支持标量转述；它不是完整“EFI→编码→APP实体→校正→mesh”的组合执行。各模块合计49项不可表述为49个完整工作流。此次独立审查另外实际组合了球面校正与带孔网格，修复后通过。
- 两组 surface 投影测试只检查 crop尺寸与resize后shape，没有验证完整空间映射对应和颜色/mask同步；报告已有此边界，建议保持。真实计量精度、保存格式、局部视差等仍须现场数据。
- 复现性方面，现有 JSON 记 `reference.py` SHA；surface未记录OpenCV版本，三个模块未各自记录测试脚本SHA/fixtureSHA。若总体交付 SHA清单覆盖这些文件，已有静态身份；若单独发一个JSON则证据不足。此项是交付追踪建议，不是数学失败。

## 本次执行身份

| 模块 | 原始快照 reference.py SHA-256 | /tmp 重跑 |
|---|---|---|
| surface | `cffa971b81fb5095def4360d6bffabc8184f63d4429b32182df58a0172075dab` | 21/21 |
| rendering | `4b0daabcdb7acff4158a4d8d3be49e9b91586be66e9512334c0b2ef6128c7ed3` | 12/12 |
| height-contract | `cde64e53bf06a79d6cc67cbda4f31fcbd81185309afa701f31ac35c9ca522e3e` | 11/11 |
| rendering 修复后 | `723d83553f566f7f445f59a82b3ea81ee6a4373fc12a8e1a758b1806fddb5fdb` | 15/15，另真实球面→mesh组合通过 |
| height-contract 修复后 | `cd0785fc9a434e0742cda1cff2d1408a9ea7ae6c8b9c8157dd0e50f512954483` | 13/13，另21个非有限参数探针通过 |

解释器为仓库既有 `.venv/bin/python`；Python 3.13.12、NumPy 2.4.3、OpenCV 4.13.0。原有测试 stdout/返回码保存于临时副本 `replay-results.json`，补充探针与输出为 `probes.py` / `probe-results.json`。修复后的副本为 `rendering-fixed/`、`height-contract-fixed/`，结果为 `fixed-results.json`、`fixed-combination-results.json`、`height-fixed-results.json`。未执行原厂 EXE/DLL，也没有向设备发送命令。
