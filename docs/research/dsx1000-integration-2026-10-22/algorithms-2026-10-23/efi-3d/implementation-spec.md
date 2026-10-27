# DSX SG CPU 景深合成与焦点高度：可实施规格

日期：2026-10-23。依据用户提供的 DSX-BSW 副本静态恢复；未加载或执行 Olympus DLL，未连接硬件。本文描述 SGSharp → SGmv → SG64 的 CPU 路径，不把其他 Core64、CUDA 或 ALMIS 分支视为已覆盖。参考实现为 [reference_efi.py](reference_efi.py)，证据索引为 [evidence-index.json](evidence-index.json)。

## 1. 复现范围与输入输出

本路径完成 **已知 Z 的图像序列 → EFI 图 + 每 XY 一个 Z 的高度图**。它不生成三维体素，不保留同一 XY 上的多个表面，不生成显示网格。显微标定、绝对台面坐标、采集完成、新帧身份、APP 16 位高度编码和显示夸张均在本层之外。

原生输入支持灰度或三通道、8 或 16 位无符号像素，图像和深度行跨度由参数指定；内部灰度、响应、深度和引导计算主要用 float32。8/16 位三通道核灰度均为 `0.114 B + 0.587 G + 0.299 R`，不做 gamma 线性化。补查确认16位灰度和彩色均先无符号扩展，再直接转换为 float32，保留0..65535原强度；没有除256或右移8位。参考保持原强度的输入转换与此一致；引导滤波的 float64 数值保障仍是独立改进，不等于原厂数值对齐。每次输入携带 float Z；本层不会从图像或硬件读取实际 Z，Z 的单位等于调用者传入单位。

SG64 最终输出同输入宽高/通道/位深的 EFI，及 float32 单通道高度；EFI 最后通过 `cv::Mat::convertTo` 转回无符号整数。SGmv 若启用稳像可改变有效区域，见第 10 节。参考程序输出 float32 EFI 方便检查，保存成原像素类型时需显式饱和舍入。

`normalize_stack` 在参考实现中拒绝非有限数；默认接受单调升序或降序，任意顺序需显式开启。原生默认也限制方向，重复 Z 的比较阈值接近浮点零（约 `4.93e-32`），并不把小数 Z 取整。原生包含排序、去重、原输入与排序位置索引映射。arbitrary排序对相同Z再按原输入序号升序（0x180011c40、0x180011630），并非无次序随机保留。参考实现默认拒绝重复 Z，只有显式 duplicate_policy=first 才保留首次图像；正常单调输入在原厂确实跳过相邻重复层的响应/平均累加，而 arbitrary 模式全部处理后才去重，且平均分母采用 unique count，可能放大平均引导图；原厂 arbitrary 重复 Z 图像归属及累加平均的每种组合仍列 E04，不用该策略假装已数值对齐。

关键静态证据：SGSharp `ev_379de98b66523eeb7bc419eabf4a809f30e787c4bec9fb63d97cafd76c9bb6b2`；SG64 `0x180002a40` 参数检查、`0x180002e50` Z 归一化、`0x18002bc20` BGR 灰度；各记录的完整 ID 见索引。

16位载入补证链：CPU vtable `0x18007c090` 的第9/10槽分别为 `0x180024440→0x18002c0d0`（灰度）、`0x1800244d0→0x18002c2f0`（BGR）。分组及尾部汇编均可见 `MOVZX word→CVTDQ2PS`；彩色组件按原强度进入累计图，灰度另乘上述三权重。证据 `ev_c6e5c0c4cc790121912c86668d568289a0f6b49b4a0846fc766c0c082d1ec172`、`ev_69af2bc5c4ffa73c29e55d21ca5083e5069699622e602d162cd4da87c7ac4d0a`、`ev_2277be1da764d865d0fd1581ec71dc960d70e1313c74bd5ebbc7a34ec15b1d6e`；[完整闭合说明](../stabilization/efi-loader-e11-findings.md)、[汇编与伪码](../stabilization/efi-loader-e11-evidence.md)、[8条原始REA记录](../stabilization/efi-loader-e11-evidence.json.gz)。本次补证后已同步验证脚本提示，并重新生成37/37结果：16位原始强度转换已证，全流程数值一致性仍未验证。独立审查的历史快照保持原样，其旧提示应按当时证据范围解读。

## 2. 参数、选择与依赖

### 2.1 CPU 分派和公开调用

SGSharp 的 8 个 P/Invoke 指向 `OlyExtendFocusImageSGmv64.dll`：GetMemorySize、CalcCacheImageNum、CreateContext、ReleaseContext、InputImage、Construct、InputImageForConstruct、EndConstruct。SGmv 再转调 SG64。SG64 ComputeMode=1 创建 CPU 实现；0 是另一 reference 路径，2 是 CUDA。CPU/GPU 分派与“快速/精细采集”界面模式是两个维度。

SGSharp 文件头是 PE32/I386，但 CLR flags=IL_ONLY 且不要求/优先32位；它可在64位托管进程运行。两层原生库为 AMD64。因此不能仅凭 SGSharp 的 x86 PE 标签选择32位桥接。当前源文件头与SHA见 `source-identities.json`。

托管声明及阈值 IL 见 [managed-band-and-api.md](managed-band-and-api.md)。原生内部使用 OpenCV 3.1 相关导入、自有金字塔/像素核、SSE 和并行循环。自然样条、频带选择、弱纹理传播、引导加权中值的组合顺序在 SG64 自有代码中可见；高斯模糊、resize、矩阵运算等委托 OpenCV。仅导入库名不能证明某个算法被调用，本文以实际调用点为依据。

### 2.2 托管后备默认值

这是 `EFIDefaultParam.GetDefaultParam` 先写入、然后可被 INI 覆盖的值，**不是设备现场参数**。

| 参数 | 后备值 | 含义 |
|---|---:|---|
| width/height/channel/bits | 1200/1200/3/8 | 之后上层覆盖尺寸等 |
| max/cache images | 100/100 | 内存预算可再改变 cache |
| arbitrary Z / adjacent blend | false / true | 顺序限制/相邻焦层取色混合 |
| Lap levels / depth level | 4 / 3 | 4 个残差频带，高度在第 3 级尺寸，即约 1/4 宽高 |
| fast Gaussian / 3D Gaussian | 3 / 11 | 响应空间平滑核 |
| near samples / interpolation | 5 / 8 | 近峰值窗口/全局候选网格密度 |
| 3D / EFI weak texture threshold | 1.8 / 1.5 | 比较响应幅值，不是概率 |
| texture scale / kernel / nominal max / SAD | .5 / 9 / 500 / 10 | CPU 实际没有使用 max 作循环计数上限 |
| guide channels / scale | 3 / .5 | 图像引导，小图计算系数 |
| 3D WM radius / eps / median | 5 / .01 / 5 | 引导加权中值及中值滤波 |
| EFI WM radius / eps / median | 1 / .01 / 3 | 同上，参数不同 |
| 3D self radius / eps | 9 / .0001 | 归一化高度自引导平滑 |
| EFI self radius / eps | 1 / .01 | 同上 |
| fast/sweep threshold/max | 5 / 1 | 快速 EFI 与平均图混合，构造引导图 |
| fast/fine threshold/max | 30 / 1 | 最终 EFI 混合 |
| ComputeMode | 2 | 托管后备值是 CUDA；本文单独选择 CPU |

参数校验中可见：通道 1/3，位深 8/16，层数 1..32767，cache 1..max；Lap level 1..4；奇数 Gaussian 1..31；近邻数正奇数；插值数≥1；WM/self 半径 0..15；median 3/5；权重至少一项>0。没有证据表明正权重被自动归一化。DepthMapPyrLevel 校验与索引代码对 0 的契约不一致：实现读取 level-1，参考实现只允许 1..levels。

### 2.3 四频带权重不是统一常数

上层 `SGPreProcessTask` 调用 `DaoHelper.GetLensCutoff` 再把返回标量传入 `EFIParam.GetDefaultParam(tempPath, cutoff)`。`GetLensCutoff` 可见公式：

`x = 2 NA / (LensMagnification * OpticalZoom) / (OPTICAL_LENS_WAVE_LENGTH * 1e-6)`。

依据父研究分支已导出证据 `ev_891cb106c231976c46418566a57ec65e2fc1312be125613c325169caa6de500f`、`ev_fc5f7d5ef1644e9899021e77c833b1615a828d67daf31b9c904498e0b9d5286d`，参见 [高度合同证据](../height-contract/sg-preprocess-cil.txt) 和 [截止量来源](../height-contract/lens-cutoff-cil.txt)。因此 x 不能直接称为倍率。NA 选择、波长常量及现场配置以该分支补证为准。

SGSharp `EFIBandBlendParam.GetDefaultParam`（token `0x06000025`）给出六个阈值，3D 为 `[70,35,20,15,10,0]`，FastEFI 为 `[35,20,15,10,5,3]`，可被 `BlendParam/BandBlendParam.ini` 覆盖。`_CalcBandBlend`（`0x06000085`）在相邻两频带间作线性过渡：x 从大到小依次使用 band 0、1、2、3；在 `[t1,t0]` 从 band1 线性过渡到 band0，在 `[t3,t2]` 从 band2 过渡到 band1，在 `[t5,t4]` 从 band3 过渡到 band2，中间区间单独一个 band=1。`band_blend` 给出逐分支实现，测试全范围非负且权重和为 1。INI 阈值顺序退化时的原厂行为未替它修正。

参考验证默认使用明确标注的 `[.25,.25,.25,.25]` 研究参数，避免假装知道现场光学设置；可用上述公式替换 profile 权重。对提供的整个 DSX-BSW 目录按文件名枚举，未发现 EFIParamDefault.ini 或 BandBlendParam.ini；只证明该副本中这次文件名搜索未命中，不证明运行时不会生成或从其他目录加载。

## 3. 每层输入的焦点响应

令图像灰度 `G0=gray(Ik)`。构建 L 个频带，每级：

1. `G[l+1] = decimate2(gaussian5(G[l]))`，核可分离 `[1,4,6,4,1]/16`，输出尺寸 `(ceil(W/2),ceil(H/2))`。
2. `U = upsample(G[l+1], shape(G[l]))`。一维偶数位置 `(left+6*current+right)/8`，奇数位置 `(current+right)/2`，二维分别执行。
3. `A[l] = abs(G[l]-U)`，不是带符号 Laplacian，也不是直接 `cv::Laplacian`。
4. 全部 band 重采样到高度计算尺寸。更细的 band 用面积缩小；本级复制；更粗的 band 用双线性放大。
5. 对同一个 A 独立做 fast 和 3D Gaussian 平滑（核通常 3 与 11，sigma=0，BORDER_REPLICATE）。

边界不能含糊称“默认 OpenCV”：缩小为 reflect101；放大左边排除端点反射；右边若目标尺寸为 2N 则包括末端，否则 2N-1 排除末端。已追到四个边界跳转桩和行/列补边函数。参考 `pyr_up` 显式实现此奇偶差异。极小尺寸兼容性未与原厂执行比对。

`SearchMaxLap` 比较 `old <= new`，相等时选**较晚输入**，不是较高 Z。参考程序排序 Z 后仍用原输入 ordinal 打破平票，避免反向输入被错误改成总选高层。

每 band 的快速 EFI 从该 band 获胜焦层取**全分辨率像素**。低分辨率获胜图映射为 `sx=floor(x*Wd/W), sy=floor(y*Hd/H)`，没有对层号做双线性插值。`F = Σ wf[b] I[winner_b](x,y)`；`Qf = Σ wf[b] max_k Af[k,b]`。所有输入同时累加为平均图，供后续引导混合。

关键地址：`0x180016000` 金字塔、`0x180015100/152a0` 下采样核、`0x1800157e0/15960` 上采样、`0x180016710` band resize、`0x18001c3b0` Gaussian、`0x180016840` max 更新、`0x180025a80/25b60 → 2d520/2d840` 获胜像素拷贝。说明伪码见 [初始证据](sg64-evidence-pseudocode.md) 与 [补查证据](sg64-followup-evidence-pseudocode.md)。

## 4. 从响应到连续数值高度

唯一 Z 数 N=1：直接输出该图及常量 Z。N 小于近邻数 K（通常 5）或 K<2：用离散获胜层的实际 Z，不走样条。该回退的质量图有特殊行为：每个活跃 band 通过 scalar*Mat 后直接赋值，而不是累加，所以最终 Q 等于最后一个活跃 3D band 的权重×离散最大响应。汇编已补证，参考照此实现。

对每个启用的 3D band、每个低分辨率像素：

1. 根据离散获胜层在**排序唯一 Z**中的 rank 选 K 个连续近邻；开始位置 `clamp(rank-floor(K/2),0,N-K)`。
2. 用实际 Z 间距 `h[i]=z[i+1]-z[i]` 解自然三次样条。端点 `c0=c[K-1]=0`；内部三对角方程 `h[i-1] c[i-1] + 2(h[i-1]+h[i]) c[i] + h[i] c[i+1] = 3[(y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1]]`。
3. 每段 `b=(y[i+1]-y[i])/h - h*(c[i+1]+2c[i])/3`，`d=(c[i+1]-c[i])/(3h)`；`S(t)=((d*t+c)*t+b)*t+a`。
4. 全局候选网格 `dz=(zlast-zfirst)/((N-1)*interp)`，`Z[j]=zfirst+j*dz`，j=0..(N-1)*interp。仅枚举近邻 Z 区间内候选，使用 float32 的减法、除法、ceilf/floorf；随后将 `j*dz+zfirst` 分步按 float32 舍入，若候选超过区间上界则递减最高 j。不能先构造 float64 网格再按 float32 端点过滤：这会在已有反例中误删最后一整格。
5. 取样条响应最大候选；严格 `>` 更新，因此相同候选响应保留先出现的候选。不是解析求导极值。
6. 原始高度 `D=Σ w3D[b] Zpeak[b]`，质量幅值 `Q3D=Σ w3D[b] Speak[b]`。

**同一 D 被复制两份进入 3D 和 EFI 后处理。** 两套参数并不意味着两份不同 band 权重的原始高度。FastEFI 的 band 权重影响颜色与引导质量，3D 权重控制原始高度和其质量。

输入允许不均匀小数 Z，而候选网格仍是全局等距，可能偏离某个实际输入 Z；有限插值网格与光学测量精度是两回事。弱纹理、多个等强峰、栈首尾最大值都可能给出有限但不可靠的高度。原生公开输出没有对应的概率置信图；Q 是响应强度。本研究额外输出 boundary_winner、peak_margin、measured/support 等标签，它们不是原厂字段。

关键证据：自然样条 `0x180013fd0` / `ev_cf52a7fd57ecde67a70af79600563ab4fa5e70a8d741ab7f42b9c87f546e952f`；候选 `0x180014890` / `ev_bb2e3c3e524897a281f424abbe687b7acd4355942d41c191c3f385a6fefc3501`；单调升/降/任意顺序核 `0x1800251b0/24c90/24770`。`ev_4995d6886e1994bac47df223cac38803753e228fbcafe4a2007e423a28ef56e1` 汇编证明 dz 是 float，并未转整数。

## 5. 弱纹理传播

3D 和 EFI 各自执行以下步骤，阈值不同：

1. D、Q 面积缩到 texture_scale。`valid=(Q>=threshold)`；`V=valid*(D+1)`，`M=valid?1:0`。
2. 分别 `GV=Gaussian(V)`、`GM=Gaussian(M)`，replicate 边界。
3. 原有 valid 锚点保持原值；其余像素 `Vnext=GM>0 ? GV/GM : 0`，支持图更新为 `GM>0`。
4. 计算非锚点 `SAD=Σ|Vnext-V|`，严格 `SAD < sad_threshold` 才退出。
5. 双线性放大 V，减 1，**只替换原高度网格中原 Q 低于阈值的位置**。

`0x180012b70` 的汇编已确认：nItrMax 只在进入循环前判断正数；循环尾没有计数加减或上限比较，只有 SAD 条件。设置 SAD=0 时，有限非负 SAD 无法满足严格小于；不能把默认 500 当作可靠退出保障。原厂本函数未执行。参考程序强制 cap 并报告 converged=false，这是有意的复现差异 E01。

全部像素都弱纹理时该通路的数值可成为 -1，表示传播没有支持，不能当作真实负高度。高度可能本来是负数，因而不能只靠 -1 sentinel 判断，必须保留独立有效性/来源标签。

证据：`ev_95adec5bb101242d5660b807890989f7587d5f12c237cd164e7fa36be4fc5651`、循环汇编 `ev_f407aba66df2a758d2f8dd0943adfc11f68a1f6010345dd21ee35fc8d739a496`、`0x1800121b0/124d0`。

## 6. 图像引导加权中值

先形成共用引导图：`J = weight(Qf)*FastEFI + (1-weight(Qf))*MeanImage`，其中 `weight=min(maxBlend*Qf/threshold,maxBlend)`，这里使用 fast/sweep 参数。Qf 权重双线性上采样到颜色图。J 缩到高度计算尺寸。引导图实际使用 float 原像素强度（8 位范围通常 0..255），没有观察到先除 255；eps 数值不能套用归一化图的同名默认值。

对任一输入高度 P：

- 量化标签 `label=rintf((P-zfirst)/dz)`，不是把物理 Z 取整。
- 对每个候选高度标签 l 构造 one-hot 图 `p=(label==l)`，以 J 作快速引导滤波。
- 按 l 升序累加滤波后的权重，第一次累计严格 `>0.5` 时选 `zfirst+l*dz`。原生每组处理四个标签；这只是并行分组，不改变顺序定义。
- 最后 `medianBlur`，核 3 或 5。半径=0 时跳过该整段。

快速引导滤波的具体形式（在 scale 缩小后计算，box 边界为 reflect101）：

`mu=box(I)`，`meanp=box(p)`，`covIp=box(I*p)-mu*meanp`。

灰度：`a=covIp/(box(I²)-mu²+eps)`。

三通道：`Sigma[i,j]=box(Ii*Ij)-mui*muj+eps*[i==j]`，`a=inv(Sigma)*covIp`，`b=meanp-a·mu`。

输出 `q=upsample(box(a))·I_full + upsample(box(b))`。指导图系数、one-hot 输入有额外 scale 降采样；此处需要区分高度网格与系数网格。

没有证据显示 q 被限制为非负或累计后归一化；局部线性引导滤波可能有负权重，不能把它描述成普通非负核中值。原生输出预清零，若未越过 .5，保留零；参考另外返回 unassigned。0..255 三通道且 eps=.01 时数值条件可能较差，原厂手写 3×3 逆矩阵与 NumPy 求逆的舍入不保证一致。独立审查证明原型的 float32 矩计算在 16 位相同 RGB 通道时会使 eps=.01 消失并触发奇异矩阵；参考已将全部引导均值、矩和逆矩阵计算提升为 float64，输出再转 float32。这是明确的数值稳定改进，不是原厂实现证据。完整多层 16 位灰度和彩色共线输入已回归；另以 RGB=(g,g,g) 等价灰度 eps/3 的独立标量公式核验。

关键地址：`0x18001aa20` 编排、`0x180019240/19a20` 灰度/彩色系数准备、`0x180019010/19740` 方差/逆矩阵、`0x180017d60/180c0/18360/1a680` 系数与输出、`0x18002eb70` one-hot、`0x1800120e0` box。

## 7. 名称为 FastBilateral 的高度平滑

`0x18001c480` 实际调用前述**灰度自引导滤波**，不是经典指数空间距离×强度差的 bilateral kernel：

1. 找 P 的 max；min 随后被 sorted unique Z[0]（最小唯一 Z）覆盖。
2. `Pn=(P-zfirst)/(max(P)-zfirst)`。
3. 以 Pn 同时作引导和输入，使用本分支 self radius/eps/scale。
4. 反归一化。

若 `max(P)==zfirst`，原生此函数返回 0x18，CPU Construct 会中止后续正常输出；参考选择原图返回并把该行为列 E02，不把它伪装成原厂成功。只有 N=1 的更上层早退不经历这一步。

## 8. 精细 EFI 与最终颜色

用 EFI 分支处理后的高度，双线性升回原图尺寸。对于每个像素查物理 Z 邻居：

- adjacent blend 开启：`alpha=(D-zlo)/(zhi-zlo)`，颜色为 `(1-alpha)*Ilo+alpha*Ihi`。
- 关闭：ratio<.5 选下层，否则选上层，恰好中点选上层。
- 最大端边界明确选末层。**最小端边界有反编译后经汇编确认的未赋值/沿用 alpha 路径**，见 E03；参考显式 clamp 到 0，并不复现寄存器未定义状态。

最后 `Out = w*FastEFI+(1-w)*FineEFI`；w 来自 Qf 和 fast/fine 参数（默认 threshold 30,max1），与构造引导图时的 threshold5 分开。原生 maxBlend 参数若大于1可能形成外插，不能不经证据一概称凸组合。

主要地址：`0x180004050` 层索引/权重准备、`0x180003c30` 最近层、`0x180003960` 邻层混合权重、`0x18000ea90/0f040` 8 位灰度/彩色取色、`0x180003550` 混合。`ev_bc6d07bdf598eb01ed26a4ed05eabcf546f81f56624c125ab96a0eda1626dda0` 汇编记录详列最小 Z 分支。

## 9. 低缓存二次供图合同

本节是静态接口行为，参考 Python 采用内存中全栈，未模拟厂商分配器/ABI。

1. CreateContext 完成后按顺序 InputImage(pointer,floatZ)，第一遍获得响应、快速颜色和高度所需信息。
2. Construct(outputEFI,outputDepth,callback,state) 计算高度、滤波及精细取色的索引。cache 足够则直接完成返回 0。
3. cache 小于累计输入数时，保存中间 Mat/索引/权重，second-pass consumed=0、expected=first-pass input count，**返回 1**。这个 1 是待再次输入图像，不能称成功输出。
4. InputImageForConstruct(count,pointerArray) 没有 Z 参数；它依靠第一遍已保存的原始序号→排序序号映射。调用者须按第一遍原顺序提供相同图像。批次数量超过剩余返回 0x11；未处于 second pass 返回 0x16。
5. EndConstruct 在未处于 second pass 返回 0x16，已供图不足返回 1；完整时做最终混合、输出深度，清 second-pass 和输入计数后返回 0。

CPU Construct 入口另有：无输入返回 0x15；回调非零要求取消返回 0x19；正常完成返回0。外部包装只要返回不等于1，就清输入计数和顺序标志。0x18 是上述退化高度等算法状态，**不能当作“需要二次供图”**。

回调位置确实用于取消检查和进度调用，managed delegate Invoke 返回 int32、参数为(native-int userData,double progress)，参数的精确 C ABI 仍须与实际架构匹配，不把它当硬件停止。缓存不足时 Construct 调用进度 1 并不等于 EFI 输出已经有效。第二遍 SGmv 会复用第一遍位移，不重新估计；影像内容一致性由调用者保证。

证据：`0x18001e660` wrapper 汇编 `ev_84fb3aaf6346047ea5be294fa624b04eef3738bb1aca0394bbaa0db53e23b55f`；`0x18001e6a0` 汇编 `ev_a2aed7eeceb789d780a6b587d118bd723000fd2865234db35bce33a20a222b74`；`0x18001f8f0` `ev_053ea1fefb81aaed0910b37dba2e6bba7f2bae1526e8e79cab05cc2d0dbb18b2`；`0x180004260` `ev_067f9786c9792a47e534bb638157d6714ba6b8b9c4224e4cb218fa59b4332142`；`0x18001fc80` `ev_7b67ede848601e9752368f7180cc7132fc949ffa62475b28c06928c1edb1f548`。

## 10. 栈内稳像几何

SGmv 可选择 `OlyNmv.dll` 或 `ALMIS3ol64.dll`，先把图像转换为估计器需要的格式，取相邻帧位移与质量，再累计。质量达到阈值才接纳新位移，否则沿用此前累计位移；X/Y 累计位移限制在 ±margin。低质量保留旧值不是测得“无移动”。

每层应用 `warpAffine([1,0,-dx-marginX; 0,1,-dy-marginY])`，输出 `Wc=W-2marginX,Hc=H-2marginY`。剩余内容进入 SG64。若输出模式不等于1，EFI 与 float 深度可一起 resize 回 W,H；模式1保留裁后尺寸。

复原源像素坐标（基于调用的 OpenCV resize 中心语义推导）为：

`xsrc=(xout+.5)*Wc/Wout-.5+marginX+dx[k]`；Y 同理。

因此 resize 回输入尺寸后，像素间隔应为 `original_pitch_x*Wc/W`，并保留裁切原点和该层位移。公开 8 个 EFIMV 导入未返回逐层轨迹/有效 mask/变换；调用者可知道 margin/output mode，但是否将元数据传至 APP 另查，不能仅凭等宽等高沿用原始标定。

参考 `stabilize_given_vectors` 实现质量门控、累计、限幅、warp、裁边、resize 及额外几何元数据；**位移估计器不在该函数中**。NMV 的估计器另由 [防抖研究](../stabilization/) 分支覆盖，本分支不把外供向量测试当成 NMV/ALMIS 复现。

证据：SGmv `0x180001740` 动态载入、`0x180002490` pair estimate、`0x1800022d0` 累计门控、`0x1800028b0` warp、`0x180002cc0` 输出尺寸恢复；完整包 [sgmv-evidence.json](sgmv-evidence.json)。

## 11. 可运行的独立参考与验收

本目录源码只有 NumPy/OpenCV 运行依赖，验证额外使用 SciPy 的自然样条作独立 oracle。没有 ctypes、DLL 装载、网络连接、设备路径或写原厂配置逻辑。

从仓库根目录运行：

```sh
.venv/bin/python docs/research/dsx1000-integration-2026-10-22/algorithms-2026-10-23/efi-3d/validate_reference.py
```

当前测试 37/37，通过以下类型：不均匀小数 Z、自然样条对 SciPy、两表面单值高度反例、边界焦层标记、全弱纹理无支持、补值保锚点、SAD=0 退出差异、灰度引导双 box oracle、RGB 常量保持、四组奇偶金字塔尺寸及手算非恒定边缘/脉冲 oracle、末层与内部候选窗口的 float32 边界、完整多层 16 位灰度/共线彩色、物理 Z 邻层颜色、权重和为1、原输入序平票、反向/任意顺序、质量门控稳像裁边、resize 几何。

独立审查暴露的三种错误变体已在临时副本中重注入：旧 float32 矩计算重新触发奇异矩阵；旧 float64 网格被两个边界测试捕获；错误的奇数右 halo 被两个非恒定测试捕获。见 [复查回归结果](reference-validation/independent-review-regressions.json) 和可重跑的 [变体脚本](reference-validation/regression_mutations.py)。

端到端两平面纹理样本：原始高度 MAE=0.06324，后处理 MAE=0.12073；EFI 像素 MAE=7.62566，相同中间单层=26.13314（灰度0..255、合成单位）。后处理在这个场景增大高度误差，图示可见边界处偏移/拖尾；这是应保留的反例，不应通过选择性去掉坏区域掩盖。数值只描述此合成 fixture，不能用来宣称 DSX 精度或提速。

另有离线 NPZ 命令行入口，显式填写 Z 单位，输出数组、有效性标签和含输入/源码/输出 SHA 的 JSON；已对同一 fixture 重放并逐数组相等，拒绝覆盖已有输出，见 `reference-validation/replay-check.json`。例如从本目录运行：

```sh
python reference_efi.py --input reference-validation/fixtures.npz --images-key scene_stack --z-key scene_z --z-unit synthetic --output reference-validation/new-replay.npz
```

输入与输出数组在 [reference-validation/fixtures.npz](reference-validation/fixtures.npz)、[scene-result.npz](reference-validation/scene-result.npz)，完整环境/源码/fixture SHA 在 [results.json](reference-validation/results.json)，对照图为 [comparison-panel.png](reference-validation/comparison-panel.png)。图已经本地打开检查，标签完整，边界异常可见。参考源码、参数、输出、证据一起保存以便现场原图/原厂输出回归。

## 12. Gap ledger 与下一次比较合同

| ID | 精确边界 | 对复现的影响 / 下一步 |
|---|---|---|
| E01 | 原厂纹理填充不执行计数 cap | 参考强制 cap，报告收敛状态；不复制无限循环行为 |
| E02 | maxDepth==sortedUniqueZ[0] 时 native self filter 返回0x18 | 参考保持常量；做原厂对照时核对上层是否捕获/回退 |
| E03 | FineEFI minimum boundary alpha 未定义/沿用 | 参考 alpha=0；不能声称边界逐位相同 |
| E04 | 重复 Z 去重后图像归属、平均图计数和 arbitrary 排序相等项 | 排序已补证为(Z,original ordinal)双键；升序/arbitrary取首个、单调降序逆序后取最后同Z图。参考默认拒绝重复，显式first为独立策略，未复刻三种模式的不同累加/缓存行为；唯一Z不受影响 |
| E05（已闭合） | N<K 离散回退质量的反编译返回寄存器混淆 | 汇编证明 Q 每 band 赋值，最后活跃 band 留存；已实现并测试，证据 ev_709a90a816dabcb85491acf94339e6128eb6f629d9732f6e2e11ee3b4ae7b8ec |
| E06 | OpenCV3.1 vs4.13、float32 vs参考部分 float64、SSE 逆矩阵舍入 | 只有逻辑/公式复现；定义容差后逐阶段比较，不宣称 bit-exact |
| E07 | 真实 optics/INI、CPU/CUDA/UI mode 选择 | cutoff 入口、波长550及NA精确查表来源已由 height-contract 闭合，实际镜头/INI仍须现场核对 |
| E08 | NMV/ALMIS 与 legacy Core64 不是本文 CPU SG64 | 与其他分支按实际入口连接；给定向量不等于估计器复现 |
| E09 | EFI float Z → APP存储/标定/mesh | 交由 height-contract/surface/rendering 分支；不能从此 DLL 推出物理高度已正确编码 |
| E10 | Native 全部 buffer ownership/内存预算/异常处理ABI | 参考全栈内存；接入原DLL前单独按签名与状态测试，不能照搬 Python内存接口 |
| E11（输入转换已闭合） | 16 位灰度/彩色载入强度缩放 | 分派、两载入核及分组/尾部汇编证实uint16直接转float32，无/256或>>8。原型保持原强度正确；float32/SSE与参考float64引导的数值差异保留为E06，不声称全流程逐位等价 |

下一次最有信息量的对照输入：带唯一编号的7～9层原始图、每层实际小数Z、完整参数/INI、原厂EFI与未校正高度、启用/禁用稳像各一组。比较每阶段 band response、winner、raw depth、两份 post depth、EFI、crop transform；允许先比较可导出的末端，再按差异定位。应同时含低纹理、窄纤维、两个表面和栈首尾焦点，避免只在单个易例上验证。

复现状态：**CPU SG 主路径已达到可实现并有独立参考的程度；未达到全版本、全边界、逐位一致或实机验收。** 保留原始 Z 栈、原始高度与派生图，禁止用平滑表面替代未测区域的有效性。
