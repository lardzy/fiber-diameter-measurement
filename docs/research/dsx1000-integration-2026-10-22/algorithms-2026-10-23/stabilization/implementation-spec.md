# DSX 实时防抖重建规格

范围为随本次副本分发的 `OlyNmv.dll`、托管 live filter 与 WPF 显示链。此文给出由静态程序证明的运算和状态；`nmv_reference.py` 是独立研究实现，没有装载或运行原厂程序。OpenCV 4.13 / SciPy 求解器代替原厂 OpenCV 2.4 与静态链接求解器，尚无原厂输入输出数值对照，不能称逐位等价。

## 1. 身份与证据定位

核心 `OlyNmv.dll` SHA-256 为 `3f7556ebfb87e739a2503c66330703c1836c4f2e7871a549c3b7b3e81f31b875`，x64。原生地址是 Ghidra 默认映像基址 `0x180000000` 下的 VA；不是文件偏移。托管 token 只在对应 SHA 程序集中有意义。全部文件及证据身份见 `source-identities.json`、`evidence-index.json`；原始 REA 记录见压缩证据包，可读摘录见 `.il.txt`、`.pseudo.c`。

| 运算/状态 | 方法/地址 | 证据 ID |
|---|---|---|
| 帧格式、下采样尺寸 | `0x180001c10`, `0x180001c80` | `ev_43c990c218361964d5b507513f44a5a8d404d7b940ddbfc07089c24d62071aae`, `ev_d1e522a5d30d9ab43b769a31ac29c91fe302b6edb85b971d127136b2abf8365d` |
| 8/16 bit 转换 | `0x180001da0`, `0x180002010` | `ev_c5aafc918a44b2aef144b5f654d4afc4ecabc26808df297c7b29f2e8474e6dc3`, `ev_3fb4f7ba64e4cbfc31c06a6c9f68c760e06d2a5741fbbd87fda42232a17039ba` |
| Hann、公开估计入口 | `0x180004d70`, `0x180004f30` | `ev_4058c7c46e47e5ea935564cdab5f9c4ba9ce7eec417b8c045e43c7ca10049a5e`, `ev_b846845dfcadd616b18bba0b6b78988a3b70b2fdf25569b3130982b33124cde4` |
| PhaseCorrelate 主链 | `0x180003d10` | `ev_39dc87d9f7f21510430fd95f5ec509e724a6d175b5819c686b3c5c3ea1879fbc` |
| FFT 尺寸选择/240 项表 | `0x180002820` / `0x180016520` | `ev_56f95d71117fcf7a9b354f841e2b97367ccf69b54c3ba5644178bd7cca7d6bc6`, `ev_fc100b8f3fe87aeb9d40847155a423fd986bd57c16f3cfc17a89b109ca0803fa` |
| 归一互功率 | `0x180002870` | `ev_f7e96bf60d0a5643e721cb57e416070e9d6a9ed6d956c4551ad9e3359cae892c` |
| 矩形低通 | `0x180002ad0` | `ev_fd20fcabd92215ad27a65c1ee90e7ab34b1d0694b76d0f383481568736005cec` |
| 频域结果象限交换 | `0x180002cb0` | `ev_2cf0d31cc6e097e18339e06a5f97f035c131dbad129cf09a38f266b7786f5be2` |
| 亚像素入口/残差 | `0x180003640`, `0x1800034f0` | `ev_33a285b3b4a7a1886755bd9fc6d8b491d49ccc9e8fe254e6c624c3c9ab777dc3`, `ev_8bee991df19cff034cc65a0abfbf3565e80a1f50f87f923029d1123f74ba1779` |
| 非线性求解初始化/步进/Jacobian | `0x180005920`, `0x180005ca0`, `0x180006e60` | `ev_998162eaf720507ee33d26c758717af3a583d221aa306f561a9713bbdf6a3d68`, `ev_d9f8b641eb53e6b0ccd6f6537c7a24071f782defb0b019537c3cd00de0bc46c1`, `ev_d2f6dc3bbd6e9396ad512be13b53df9cabc9b1ba680df8e9a6a712e86ccba27f` |

## 2. 输入转换和前处理

原生转换接受 1、3、4 通道以及每通道 1 或 2 字节。4 通道忽略 alpha；前三通道直接相加，没有颜色亮度加权，也不除以 3。live 托管调用实际以 3 通道、`CamByte/3` 字节每通道解释输入。

设分块大小 `D=max(1, DividerValue)`，原图尺寸 `w0×h0`，输出 `w=floor(w0/D), h=floor(h0/D)`。尾部不足整块的行列舍弃。输出单通道 float32，步长 `4w`，每点：

`I_d(y,x) = Σ[j=0..D-1, i=0..D-1, k=0..min(C,3)-1] I(Dy+j,Dx+i,k) / D²`。

原厂调用 `cv::createHanningWindow` 创建 float32 窗，窗尺寸为下采样尺寸。参考实现调用同名 OpenCV 函数，不自作主张替成 Gaussian 或普通外积 Hann。当前 OpenCV 对应的是平方根形式的二维 Hann；原厂 DLL 中此导入函数的内部实现未逐指令复核。

FFT 尺寸各选 `fft-sizes.json` 中不小于当前尺寸的首项；表共 240 项，全部偶数，2 到 65536。输入尺寸 `>=65536` 返回 -1。不能用默认 `cv2.getOptimalDFTSize` 替代，因为它可能选奇数。原图和窗向右、下补零，随后两图分别乘窗。

## 3. 相关面、低通和可靠性

记 `A=DFT(previous×window)`、`B=DFT(current×window)`。全域运算，没有特征点、ROI 搜索、旋转或仿射模型。对每个频率：

```
re = A.re*B.re + A.im*B.im
im = A.im*B.re - A.re*B.im
den = sqrt(re²+im²) + 1.1920928955078125e-7
R = (re+i*im)/den
```

注意是 `A×conj(B)`。默认启用矩形低通。mask 内部 `c_mask=clamp(c,.1,1)`，`fx=int(W*c_mask/2), fy=int(H*c_mask/2)`。未移位频谱中，若 `fx<=x<W-fx` **或** `fy<=y<H-fy` 则清零；仅保留四角低频矩形。不是径向低通，不是 Gaussian。若 cutoff 配置越界，mask 的 clamp **不会**同步改变后续可靠性分母。

实数逆 DFT 不作 `1/(WH)` 缩放。结果交换四象限，再找全局最大值 `p` 与整数峰位置 `(px,py)`。尺寸表全为偶数，因此正常二维路径的交换与 `fftshift` 一致；交换函数自身对奇数边缘不作标准 fftshift，但主路径不会产生奇数尺寸。

默认可靠性严格为：

`reliability = peak_value / (c² * W * H)`。

这里 `peak_value` 是整数峰的高度，亚像素拟合不会重算它。即使关闭低通，这个分母仍含原始配置 cutoff。此值不是概率，不能与 OpenCV `phaseCorrelate` 的 5×5 response 混用；特殊配置下可大于 1。默认 `c=.5`，live gate 要求严格 `reliability > .35`。

## 4. 亚像素拟合

峰必须有完整 3×3 邻域；边缘峰返回 -4，主链输出零位移、零可靠性。整数模式可跳过此步。邻域位置 `xi,yi ∈ {-1,0,1}`，其相关面值为 `zi`。初始值：

```
dx0 = -(left-right)/(2*left-4*center+2*right)
dy0 = -(up-down)/(2*up-4*center+2*down)
p0 = (dx0,dy0,1)
```

优化器接受 3 参数、9 残差，但已反编译的残差只读取前两参数；第三项不是已证明的幅度参数。`c_fit=clamp(c,0,1)`，残差为：

`r_i = z_i - [sin(π*c_fit*(yi+dy))/sin(π*(yi+dy)/H)] * [sin(π*c_fit*(xi+dx))/sin(π*(xi+dx)/W)]`。

这是正弦比峰模型，而非高斯峰或 5×5 加权质心。中心抛物线只是初始估计。拟合后 dx/dy 各限制到 `[-1,1]`，细化峰 `px-dx, py-dy`。原生位移 `((W/2-refined_x)*D, (H/2-refined_y)*D)`；托管层随后整体取反。

非线性优化可见有限差分 Jacobian、QR/阻尼步长和实际/预计下降比；规格上可归为 LM 式求解，但不能仅凭结构断言具体源码来源。默认 factor=100、maxfev=400、ftol=xtol=√DBL_EPS≈1.4901161193847656e-8、gtol=0。Jacobian 用正向差分，`h=sqrt(max(epsfcn,DBL_EPS))*abs(pj)`，零则使用 `sqrt(max(...))`。收敛状态 1/2/3 分别对应函数/步长/两者，4 梯度，5 最大调用，6/7/8 机器精度极限；外层不依据这些状态拒绝峰，仅循环至不为 -1。

**数值边界不能掩盖：**原厂残差直接算 sin/sin，变量恰为零时存在 0/0；初始抛物线平坦时也可能 0/0。原厂没有显式极限值分支。反编译的最终 clamp 在 NaN 比较路径会落到 -1。研究原型默认保留 direct ratio，并记录 `residual_finite`；`stable_ratio=True` 才使用 `c*n*sinc(c*t)/sinc(t/n)` 的解析连续形式，这是建议改进，不是原厂实现。没有原厂数值 oracle，不能保证 SciPy 在非有限残差上的退出与原厂完全相同。

## 5. live filter 与显示状态机

证据 `ev_9608f5546b072cfe45109c88ec2a520edaec01fb7864c17feeaacd091a863d73`：

1. `ImageProcessService._BlurringEnable 0x0600018d` 置 `NmvIsCreateBaseImage=true`，读 ini，启用 live filter。
2. `AUseableOlyNmvFilter.IsCreateBaseImage 0x06000037` 消费 DAO 新建基准标志，或检测尺寸改变；`CerateBaseImage 0x06000038` 创建窗、基准下采样图与双缓冲。
3. `ExecuteShiftEstimation 0x06000039` 用上一参考与当前图估计。原生成功后更新 reference 为当前，并交替 buffer；原生报错提前返回，reference 不前进。不是永久首帧参考。
4. `BlurringCorrectionFilter.DoExecute 0x0600009b` 新基准时请求清累计。其余帧先估计，再执行顺序 gate：任一轴绝对值 `>20` 重置；否则 `>15` 重置；否则 `reliability<=threshold` 重置；余下接受。20 与 15 逻辑有包含关系，但日志分支意义不同，不应擅自合并为单阈值再声称等同实现。
5. 接受时 `Shift=floor(shift*1000)/1000`，负值向负无穷取整。拒绝 gate **不撤销**已更新的帧参考，只清显示累计。原生返回码在这个 filter 调用处没有单独传播成成功/失败 UI；默认输出值与可靠性会使大多数错误进入重置。
6. 末尾复制 src buffer 到 dst buffer，像素值不平移、不插值、不反卷积。输出是附有位移元数据的 live frame。

`OlyuImageDisplay.ImageDisplayControl`（证据 `ev_6a6b0c6918aea1a0912603189ee65b2740f6bdd6b67e8424ec1deb550d67a2b1`）：

`_UpdateBlurAccumulationValue 0x0600002e` 对两轴独立执行：若 reset 或 `abs(acc+shift)>buffer`，该轴累计置 0；否则累计相加。默认 X buffer=20 图像像素；Y buffer=`20*GraphicHeight/GraphicWidth`，可由 setter 改变。一次超出 X buffer 不必重置 Y，除非全局 reset 为真。

显示位移为 `acc_x*display_width/GraphicWidth` 与 `acc_y*display_height/GraphicHeight`。`OlyuMicroscope.Display._ImageUpdateEventHandler 0x060009e3` 将偏移计入 WPF Thickness 的 Left/Top，再更新位图显示。该事件订阅链已闭合；没有发现这里向 XY/Z 发送反馈。图像注释的显示坐标也有边缘留白处理，但不能因此把显示像素位移写入物理台坐标。

## 6. 配置、错误与研究接口

副本 `OlyNmv.ini`：D=8、LPF=true、cutoff=.5、subpixel=true、reliability=.35。DAO 构造默认相同。配置用 `Double.Parse`，不能假定读取与区域设置无关。设定文件本身及日志应在现场比对。

原生/托管错误边界：无效参数 -1、未初始化 -2、内部异常 -3、边缘峰估计失败 -4、托管封装异常 -5。包装器 `OLYNMVUtil` 六个 stdcall P/Invoke 与导出入口对应，见 `ev_570c82c88133e0efb09c38dbcee1c6400f3463055f0ef0557d27a78f1f8d1ed8`。本原型不实现 DLL ABI、缓冲区分配与多线程生命周期，它复现计算合同和明确的 live 状态。

重建可用 API：`block_convert`、`estimate`、`filter_gate`、`LiveDisplayModel.feed/request_base`。返回始终区分 native shift、managed shift、reliability、拟合状态、gate reset、显示累计；不生成“电机稳定”“新曝光”“采集成功”等字段。被拒绝的估计仍保留诊断值。

## 7. 与 Z 栈防抖的边界

原厂还有 fast/fine EFI 的 `ParamIsStabilizer` / `ParamStabilize` 及 X/Y limit。ProcessJoin 会针对其裁边修正输出尺寸与注释偏移。它们不是此 live WPF 偏移器。其 native 算法由 EFI/3D 研究分支追踪，本规格不会把 live 的 15/20 像素阈值或 .35 直接套给 Z 栈。

## 8. 尚缺的复现验证

- 原厂 DLL 对固定输入、输出 float buffer、位移、reliability、极端/非有限参数的对照，需隔离 Windows 离线测试，不需要设备但本轮未执行原厂。
- 原厂 OpenCV 2.4 的 Hann/DFT 浮点舍入及静态 LM 与 SciPy 的迭代差异。
- live 图像全链路前后 filter 顺序和配置对输入像素的影响；目前只证明本 filter 的 src→dst 原样复制。
- 真实相机帧序、掉帧、曝光模糊、台移动时 reset 频率以及显示边界的现场效果。
- 图像纹理不足、重复结构下高可靠性错误匹配不能仅靠现有 gate 排除，合成反例应随报告保留。
