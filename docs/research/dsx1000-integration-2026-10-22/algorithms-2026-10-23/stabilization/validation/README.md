# 合成验证结果

固定 seed `20261023`，12 对输入，26 项研究实现检查通过。

| 场景 | native X/Y（原图像素） | reliability | gate |
|---|---|---:|---|
| identity | -0.0000, 0.0000 | 0.975000 | accepted |
| integer_shift | 4.7651, -2.6456 | 0.746751 | accepted |
| fractional_shift | 1.3115, 2.2599 | 0.820631 | accepted |
| brightness_gain | 4.7571, -2.6476 | 0.746536 | accepted |
| threshold_large_16 | 15.5358, 0.0098 | 0.907290 | blur_shift_gt15 |
| threshold_large_24 | 23.9484, 3.1928 | 0.741776 | stage_shift_gt20 |
| black_zero | 0.0000, 0.0000 | 0.000000 | reliability_le_threshold |
| flat_nonzero | -0.0000, 0.0000 | 0.975000 | accepted |
| periodic_hidden_32 | -0.0000, 0.0000 | 0.975000 | accepted |
| unrelated_frames | 0.0000, -8.0000 | 0.158654 | reliability_le_threshold |
| rotation_2deg | -0.2512, -0.4802 | 0.714501 | accepted |
| exposure_blur | 4.7022, -2.5205 | 0.779147 | accepted |

非零平坦图与完整周期平移故意保留为高置信反例；它们‘通过’的是反例被复现，并非配准正确。

identity/平坦场景可出现直接 sin/sin 的非有限残差，详见 JSON residual_finite。改用解析极限的结果单独保存。

输入 fixture、源码 SHA、库版本及所有状态均在 results.json。未运行厂商 DLL、设备或 FDM 产品代码。
