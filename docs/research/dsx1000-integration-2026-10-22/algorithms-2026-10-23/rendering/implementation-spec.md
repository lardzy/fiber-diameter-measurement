# 高度场三维显示的复现规格

来源身份、方法 token 与 CIL 证据见 [findings.md](findings.md)。本规格把物理数据与显示数据分开，足以用任意三角形渲染器实现同类功能；不要求依赖旧 SlimDX / Direct3D9。独立数值实现见 [reference.py](reference.py)，可直接打开的交互说明见 [height-field-lab.html](height-field-lab.html)。

## 输入与输出

- 输入高度 `H[row,col]`，原厂边界为 UInt16 高度码；同时必须有 `xPitch,zPitch,makerZ,userZ`，以及有效性矩阵。代码存在 XY 校正，不能随意假定未知的像素纵横比例。
- 测量层先生成浮点物理高度 `Z=H*zPitch*makerZ*userZ`。原图、标定、原始有效性与派生历史保留。显示裁剪、夸张和重采样都只作用于派生视图。
- 可选颜色纹理与高度同尺寸；高度 LUT 是另一种着色模式。高度无效与黑色纹理像素不是同一概念。
- 输出顶点位置、三角形索引、单位顶点法线和颜色/纹理坐标。原型 `grid_mesh` 使用 `(col*pitchX,row*pitchY,Z)`，便于物理解释；`dsx_display_positions` 单独给出原厂显示轴映射。

## 可实现步骤

1. **尺寸归一化。** 令长边为1024，短边按原图比例截断取整数、至少10，再向下对齐网格步长。高度/色图/原厂无效图路径均调用 `cvResize(...,1)`。参考实现用 OpenCV `INTER_LINEAR`；科学有效性图若改用 nearest/conservative mask，属于明确的改进。
2. **显示高度。** `r=1024/max(originalWidth,originalHeight)`；`s=0.20000000298023224*r/xPitch`；`Hd=clip((Z-minHeight)*s,0,(maxHeight-minHeight)*s)`。这一步不可逆地裁剪显示值，不可用作导出测量高度。
3. **网格。** 正常模式步长为2；轮廓模式读取 `LineInterval` 再调整。在采样网格每个单元取四角 `a=(r,c), b=(r,c+1), d=(r+1,c+1), e=(r+1,c)`，形成同一对角线的两个三角形。原厂主要写入次序是 `(d,b,a),(d,a,e)`，叉积函数再反转边顺序。可移植物理坐标原型采用 `(a,b,d),(a,d,e)` 与标准叉积，避免把原厂的显示坐标手性带进物理模型。
4. **法线。** 每个面 `n=normalize(cross(v1-v0,v2-v0))`；每个顶点累加相邻面的单位法线，再归一化，故不是面积加权法线。角点/边界/内部只累加实际邻接面。正常网格的平面法线应与解析导数一致。
5. **视觉比例。** 原厂显示位置是 `(row*0.2,col*0.2,-Hd*depthFactor)`；Reverse 切换最后一项符号。自动系数是 `max(originalWidth,originalHeight)*xPitch/(k*(maxHeight-minHeight))`，配置缺省 `k=4`；属性另舍入到一位小数。原始高度数据不随该系数改变。
6. **光照。** `diffuse=clip((level-1)/9,0,1)`，`ambient=0.2+0.8*(1-diffuse)`。原厂使用方向光与当前旋转矩阵；独立浏览器演示采用自定义固定方向光。显示高度改变后重新计算法线。
7. **相机。** 原厂居中平移、固定 Z=π/X=π/2 旋转、用户 Euler 与平移，之后使用左手 LookAt/Perspective。参考 HTML 使用独立旋转与正交投影，并没有声称复刻原厂相机。

## 退化条件和范围

- 零高度范围：自动夸张分母为0；原型拒绝，产品实现应显示平面并关闭自动夸张。
- 原厂 `conv.i2` 引入的超大原图边长溢出不应复刻为产品需求。原型只接受该辅助算法已验证范围。
- 原厂 `_IsPixelValid` 恒返回 true，法线链不因此排除无效点。原型提供“保留网格着色”和“删无效面”两种明确模式；第二种是我们自己的数据展示策略。
- 原厂索引分配是 `floor(6*(W-1)*(H-1)/(strideX*strideY))`，并有末端补写/修复循环；它不总等于理想采样网格的 `6*(W/strideX-1)*(H/strideY-1)`。参考实现生成紧凑合法索引，不复刻尾部退化三角形或缓冲区布局。所有原厂分支原文已保留，不能把参考索引数称为原厂缓冲区逐项对拍。
- 该表面只能表达每个 XY 一个高度。不能表达悬垂、相互遮挡的多根纤维或体积内部；原始 Z 栈必须保留。

## 验证与演示

从仓库根目录运行 `.venv/bin/python docs/research/dsx1000-integration-2026-10-22/algorithms-2026-10-23/rendering/validate_reference.py`，15项确定性检查包括解析平面法线、索引范围、有效性模式、编码高度换算、翻转与采样数。

浏览器演示已验证：默认6144面；高度倍数1→4后原始跨度仍12.512 μm；扣除已知二次基底后局部脊和台阶保留；删无效面后5944面；网格开关和恢复角度按钮可用。截图见 [prototype-flat.png](prototype-flat.png)，过程记录见 [ui-validation.json](ui-validation.json)。演示数据完全合成，无原厂图像或设备交互。尚未做原厂 GPU 截图、全部相机交互和逐像素着色一致性对拍。
