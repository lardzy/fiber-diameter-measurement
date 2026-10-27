# EFI / 焦点高度独立审查

日期：2026-10-23。审查对象为 `efi-3d/reference_efi.py`、`validate_reference.py`、`implementation-spec.md`、`findings.md` 及其已导出的 SG64 静态证据。只在 `/tmp/dsx-efi-review.i4Ou3s` 运行 Python 副本；没有运行原厂 DLL、连接设备或修改作者目录。以下严重性按研究参考程序衡量：P1 阻断主要结果可信度，P2 应修复的具体实现/验收缺陷，P3 文档澄清。

初始结论：29/29 现有测试可以重现，但不能据此认为已验证所有已恢复的数值分支。发现两项 P2 实现/输入行为问题、一项 P2 验收缺口和一项 P3 用词问题；均已将复现发给作者。修复后的独立复验记录见本文末尾。

## 审查基线

| 文件 | 初审 SHA-256 |
|---|---|
| reference_efi.py | `7c644c0580cfaf5711f77def5d393554685671f1369f16bc8e0689d03cf043d8` |
| validate_reference.py | `fbc8118a6bbfb13e89f40396baf6b541b3b555585fc2176067d2925362bb99d5` |
| implementation-spec.md | `c27e4f58ff6bb27cae47c842801332bada1a531c04e57f8a475d9f41e7c11d60` |
| findings.md | `6a91fde456fc4737ffcab9a636478aacf582fd3553bcc465bc5324a549aa8fd5` |

测试环境沿用仓库 `.venv/bin/python`。初审 29/29 通过；两平面内区 raw height MAE=0.0632395819、post height MAE=0.1207310632、EFI pixel MAE=7.62566233、中间层 MAE=26.13313675，与作者报告一致。这些是一个合成场景的结果，不是原厂数值 oracle，也不是光学精度或硬件性能验证。

## R1 · P2：混用 float32 间距与 float64 网格，可能整格漏掉合法端点

位置：初审 `reference_efi.py:209–218`；规格第 4 节。

`dz` 先由 float32 Z 算出，随后以 float64 `arange` 构造候选，再直接与 float32 端点转换成的 double 比较。在以下正常小数 Z 上，最后一个 float64 候选略大于末端，被整个剔除。单调递增响应因此选倒数第二格，不是最高层。

```python
z = np.arange(9, dtype=np.float32) * 1.3 + .4
response = np.arange(9, dtype=np.float32)[:, None, None]
height, _, _ = estimate_height(response, z)
# 初审输出 10.637499809265137
# z[-1] = 10.799999237060547，差约一个 dz=0.1625
```

原生 `0x180014890` 的已导出伪码明确使用 float32 `j*dz+zfirst`、`ceilf/floorf` 和上界向下修正。用该操作顺序独立计算，本例上界索引为 64，候选值为 10.799999237060547，处于有效区间。证据：`ev_bb2e3c3e524897a281f424abbe687b7acd4355942d41c191c3f385a6fefc3501`；float 间距由汇编证据 `ev_4995d6886e1994bac47df223cac38803753e228fbcafe4a2007e423a28ef56e1` 支持。

影响：不是仅最终值相差若干 ULP，而是候选集合和 winner 改变，尤其影响末层焦点和内部近邻窗口边界。原有“边界峰”测试只有低端，未覆盖这个高端分支。建议按已恢复的 float32 操作顺序生成候选/区间，并加高端及内部边界用例；不应只放宽高度误差容差。

## R2 · P2：多层 16 位三通道可在颜色引导求逆阶段直接异常

位置：初审 `reference_efi.py:283–288`；验证 `validate_reference.py:210–213`。

使用作者已有 scene fixture，在合法 uint16 范围内放大到 16 位并复制三个相同颜色通道：

```python
f = np.load('reference-validation/fixtures.npz')
images = np.repeat(
    np.rint(f['scene_stack'] * 257).astype(np.uint16)[..., None], 3, axis=-1)
reconstruct(images, f['scene_z'])
# 初审：numpy.linalg.LinAlgError: Singular matrix
```

输入为 9×96×128×3，强度范围 10169..53649。捕获到的一个协方差矩阵全部元素为 1853824.0，其 float32 ULP 是 0.125；eps=.01 加到对角后已消失，173 个系数像素的 double determinant 为 0。相同 RGB 通道本身是合法图像，不应当作非法输入。

现有 uint16 测试只含一个焦层，在 `reconstruct` 的单层返回处结束，没有经过颜色引导、量化或高度后处理。规格 E06 已承认逆矩阵舍入差异，但应把本例明确升级为可重现的退化/输入行为，不能仅描述为普通容差。

建议：为这种退化定义明确行为；若采用 float64 协方差/正则化作为独立参考修复，要标为相对原生 float32 的有意差异并保留诊断；也可明确限制当前可用输入范围。补多层 16 位灰度、相同/高度相关 RGB 和一般 RGB 的测试。当前反例只证明参考程序失败，**没有证明原厂 DLL 也失败**；尚须分别核对原生 16 位 LoadInputImage 分支及手写逆矩阵行为。

## R3 · P2：奇偶金字塔测试不能验证恢复的 halo 规则

位置：初审 `validate_reference.py:122–128`、`reference_efi.py:76–95`。

四种奇偶尺寸都只输入常量，断言常量保持和零响应。任何保持常量的边界方式均可能通过。在独立 `/tmp` 副本中，将：

```python
following[following >= n] = n - 1 if length % 2 == 0 else max(0, n - 2)
```

故意改为始终 `n - 1`，破坏奇数右边界的排除端点反射，29/29 仍全部通过；端到端 MAE 也完全不变。现有场景尺寸和内区评估没有区分这条边界规则。

影响：静态恢复的规则本身尚未被推翻，但“测试覆盖恢复的 halo”证据不足。建议加入非恒定边缘脉冲/斜坡、四种奇偶尺寸，独立枚举一维期望值或使用明确的手算矩阵，并验证错误边界变体确实被拒绝；不要用被测 `_up_axis` 自己生成期望。

## R4 · P3：高度归一化原点应称排序后的最低 Z

位置：初审规格第 7 节和 `self_guided_depth` 注释中的 “first input Z”。

`0x18001c480` 从 `param_3+0x10` 取 Z，而 `0x180002e50` 已将排序/去重后的 Z 写入该数组；原始输入序列位于另一个 `+0x08` 指针。因此此处是 **sorted unique Z[0] / 最低 Z**。实现传 `z[0]` 是正确的，不应据文字误改成原始输入首层，尤其是降序输入。作者已确认这一澄清。

## 数学、单位和结论范围核对

- 自然三次样条的三对角系数与 SciPy natural boundary oracle 一致；样条峰使用有限候选搜索，不能称解析极值或新增光学信息。R1 是候选集合实现问题，不是样条系数公式错误。
- 输入 Z 保留 float；候选层号、物理 Z 和原输入 ordinal 分开。高度单位随调用者；频带截止量来自 NA、倍率、光学 zoom、波长组合，不能直接叫倍率。原始 Z 数据来源、物理标定和 APP 编码仍须使用旁支证据，不能由本参考推断。
- 响应幅值、peak margin、texture measured/supported 标签不是概率；`masks` 是弱纹理传播阶段的来源信息，不能证明后续中值/自引导输出仍是未修改测量值。默认 `.25` 四频带权重已正确标为研究 profile，不能当作某倍率的原厂实际参数。
- 双份后处理由同一个 raw depth 派生，EFI 取色和输出表面是不同用途。单个 XY 只保留一个 Z，既有多峰反例支持“不能替代完整 Z 栈”。
- 单层直接返回、短栈质量覆盖、弱纹理无支持、两表面、输入方向、外供位移裁边的现有测试确实执行了相应 Python 分支；外供位移测试没有验证 NMV/ALMIS 估计器。
- CPU/CUDA 默认分派、实际 INI/optics、缓存二次供图 ABI 和原厂数值对照仍是主路径使用前必须确认的合同。报告已明确区分这些边界，未把 Python 测试当作原厂或设备验收。审查没有要求扩大到可选算法或全平台逐位复现。

## 修复后复验

作者修复后，审查者复制至 `/tmp/dsx-efi-fixed-review.WYqKMW`，独立重跑 **37/37 通过**。复验源码身份：

| 文件 | 修订快照 SHA-256 |
|---|---|
| reference_efi.py | `98a134a6cd6175a83be72989aff763ee686a18c0bfc7af1b83c0f31161f5fcbd` |
| validate_reference.py | `bf2ab4f0cbde5232fef6f83346d60cb44ab0a3c88011d9ed3dcafef47ea5c1e0` |
| implementation-spec.md | `b6cf12dd7347ddd6f0d70d8c0f7228d4b60ec8cff160fa2c247443fb3b758714` |
| findings.md | `aa995942d957db6384b9a2a8fd79436a0fd403e27f1f52cd5d1e1b1efc3fc9ec` |

- **R1 实现及回归闭合**：`axial_candidates` 按 float32 计算间距、ceil/floor 区间和上界修正。原反例返回 10.799999237060547，与最高 Z 完全一致。新增末层和内部窗口边界断言通过。
- **R2 参考数值退化已修**：协方差、局部矩和求逆使用 float64，输出转回 float32；原 16 位相同 RGB 多层图现在 EFI/height 全为有限值。新增 RGB 与灰度引导 oracle、多层 16 位灰度和相关颜色验证通过。此为独立参考的稳定性改动，仍不能推为 native float32 的等价结果。
- **R3 验收缺口已修**：新增手算非恒定边缘和末端脉冲。审查者重复同一错误 halo 变体，结果 **35/37，2 项失败，退出码 1**；错误值分别为 14 对正确 12、7 对正确 6。新测试实际拒绝了之前能通过的错误分支，没有放宽断言。
- 修订两平面 post height MAE=0.1207310557，raw/EFI 指标与基线相同；很小的差异与明确的精度改动一致，仍不构成原厂数值对照。

**R4 文档已修**：冻结后的规格第 7 节现为 `sorted unique Z[0]（最小唯一 Z）`。R2 的 float64 引导策略已明确标为独立改进；新增 E11 明确原生16位载入核强度缩放未逐指令核对，16位测试通过不能作为原厂数值对齐证明。

复验结论：本次三个具体 P2 事项和一项 P3 澄清均已闭环，没有剩余由这四项产生的修复阻断。**可接受为明确标注范围的 CPU 算法研究参考；不能据此认定完整16位路径、CUDA路径或原厂输出数值等价已完成。** 若下一阶段必须复现16位真实输入，则 E11 是主路径缺口，应先闭合载入强度变换并按像素幅值重新核验阈值/eps；无需为本次8位核心复验扩展其他可选算法。上述原发现与基线保留，不用更新后的全部通过来抹去初审问题。

## 后续 E11 定向闭合

按主任务追加要求，审查者随后读取 CPU vtable 与两个16位载入像素核及汇编，确认 uint16 原强度直接转 float32，BGR 核采用同一灰度系数，无 `/256` 或8位右移。见 [E11 补证](stabilization/efi-loader-e11-findings.md)，新增8条REA证据已导出。因此上段复验时保留的 **E11 输入转换缺口现已关闭**；无需修改参考输入强度。原生 float32/SSE 与独立 float64 引导的数值差异仍须保留，不能把输入转换闭合扩张为原厂全链数值对照完成。
