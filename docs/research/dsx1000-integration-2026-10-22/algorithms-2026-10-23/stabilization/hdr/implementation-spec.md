# Hamal 8 bit 单图反光抑制核心重建

本规格限定 `Common/ImageProcessing/OlyHamal64.dll`，SHA-256 `5223c273499b132712a148f2f03237f5c32102185265563eeb1f301c93d1f27b`，以及 `OlyAntiHalation.exe` / `OlyiAntiHalationWrapper64.dll`。这是软件中名为 AntiHalation/Apical 的**单张图像局部动态范围增强**链；多曝光 HDR merge 是另一组 `AHDRFilter/AHDRFastFilter`，不在本文复现范围。

所有运算从 Ghidra 函数和数据段恢复；未执行厂商程序。`hamal8_reference.py` 是相应 8 bit、AsymMode=1、BlendMode=1 的独立研究实现。其结构、表数据、整数舍入来自静态证据，尚无原厂数值 oracle，因此表述为“可实现的重建”，不宣称原厂输出逐位一致。14 bit、Spica 外部降噪、HDR 多曝光合并明确排除。

## 1. 托管到原生边界

`OlybImageProcessDomain.ApicalAntiHalationFilter._ExecuteApicalAntiHalationFilter 0x06000035` 设置 effectiveBits=8、输入/输出尺寸和指针，再调用 `ImagingExecuteFacade.AntiHalation_Execute`；`_SetApicalParameter 0x06000036` 从 LiveApical DAO 取 Asymmetry、Saturation、Strength、NRLevel 与三组 8 元素 NoiseSuppression 参数，并准备 `OlyAntiHalation.exe`、XML 与临时图路径。此处只证明准备了进程接口，不据此断言每次都会启动新进程。托管证据仍位于上级 `phase-correlation-evidence.json.gz` 的 `ev_9608f5546b072cfe45109c88ec2a520edaec01fb7864c17feeaacd091a863d73`。

`OlyAntiHalation.exe.RunAntiHalation 0x06000001` 的进程执行路径：读 XML→分配/加载单幅图→格式转换→`_ApicalExecute 0x06000005`→可选 `_NoiseReductionExecute 0x06000006`→写输出与结果。`NRLevel=0` 直接跳过外部降噪；1/2/3 交给独立 NoiseReductionWrapper。证据 `ev_09368777195f2aa665adced0801a6bdb86fa91bd98a9cc2b8003b839a08513a8`。

ApicalWrapper 构造调用 `OlyHamalCreateServiceInstance`，通过 `IID_ICustomHalationRejectionCmd` 请求命令；Execute 先调用命令 vtable `+32` SetParam，再调用 `+16` Execute。原生 factory 名在混合程序集元数据中可见，不能仅凭声明验证导出。Hamal 内 `CreateCmdInstance 0x180009ca0` 确有 Custom 命令分支；其 vtable `0x180014f90` 指向 `+16→0x1800065f0`、`+32→0x180006f80`，两者分别是带真实 Execute/SetParam 日志的执行与设参实现。`ApicalWrapper` 证据 `ev_ca952e910b1bd40a71994de244b1f3302d45fa27134a724849e81bc5f6b9adcc`；vtable 证据 `ev_5ed816492b816cd3213ce3b14775b36a937bf6a38faf668c37ce767b3b45d9b0`；factory 分支 `ev_1068a2449982f243613568e4dd689f27f92bfbc503d9fc17e495fca7a5ee3265`。跨 DLL 的实际 thunk/ordinal 装载仍需独立链接核验，算法体身份已确定。

## 2. 参数与图像布局

`SetParam 0x180006f80`（`ev_4f948285a3111b73ac30a5001a7a7084054037ce72aed771c45ba4a9c5990150`）和 `Validate 0x1800033a0`（`ev_39ab63f3fb4b3f69cc6cc1a985288fb25f3ba7c5c92f9e837b102c8287e751c2`）：

- `Execute 0x1800065f0` 的原生初始尺寸检查为宽 121..10000、高 91..10000；**这不是原型对整个区间的可用性承诺**。原型还要求固定点缩图覆盖全部 80×60 格，判据与未闭合边界见第 3 节。本原型仅 uint8 三通道，按原生 C0/C1/C2 顺序。
- Asymmetry 0..100，步长 10；省略/-1 默认 50。它选择三组 11×15 LUT 的一行。
- Strength 0..255，默认 20；Saturation 0..255，默认 64。不是百分比，不静默限制到 100。
- AsymMode=1、BlendMode=1、ExecSwitch=1；LpfCoefSum=2001、LpfCoefSumDiv=131、DivBit=18。
- 三组 8 元素默认：Slp1=`[25,20,15,12,8,5,3,1]`，Slp2=`[163,204,265,356,505,772,1326,2811]`，Bin=`[26,33,42,57,81,123,212,450]`。操作者 DAO 可覆盖，默认值不代表现场实际值。
- 三组数组均恰8个整数，原生 `Validate 0x1800033a0` 分别限定 Slp1=0..32767、Slp2=0..8191、Bin=0..1023。原型显式校验长度和范围，禁止用 NumPy 广播悄然接纳一项数组或负/超界值。

native 8 bit 像素处理阶段会使用 16 bit 中间组件，以容纳矩阵运算。最终矩阵中 C0 权重似 B、C1 似 G、C2 似 R；报告/原型明确用组件顺序命名，未把颜色标签当作已验证全采集链存储格式。

## 3. 固定缩略图、非对称曲线和自适应亮度分段

入口 `0x180004130`→`0x180004910`，相关证据 `ev_4bf95baca06f7ed2580717fd5f7a030b7af21b9f2a5da6309aafb2b00c051371`、`ev_3266da6551cef6164c1b8722f6794f3f5d78ae2ccde398adefb0c6362c0e9763`。

先建 80×60 缩略图。缩放固定点 `sx=round(79*8192/(W-2))`、`sy=round(59*8192/(H-2))`。`0x180001d30` 顺序扫描源图，选各目标格首次映射到的样点，取 `max(C0,C1,C2)`；不是面积平均，不是双线性缩小。映射格 `floor((x*sx+floor(sx/2))/8192)`，Y 同理。证据 `ev_80d438b84b2311c215f041eb32d8336831bd47070a453591d724677dd75f43d5`。

独立审查补出的尺寸限制：在上述范围内缩放步长均小于8192，映射从0开始且不跳格；仅当 `(((W-1)*sx+(sx>>1))>>13)==79` 且 `(((H-1)*sy+(sy>>1))>>13)==59` 时覆盖完整缩图。宽1445时sx=448，只覆盖79列；高1221时只覆盖59行。枚举原生尺寸区间有3701种宽和3861种高不能填满。已有 `Execute`、`0x180004910` 和 `0x180001d30` 中未见更窄的前置尺寸约束或遗漏末行/列的补值；缓冲的初始内容及上层实际允许分辨率未闭合，不能推断原厂输出值。

原型 `thumbnail_geometry` 公开上述支持域；未覆盖时抛出带尺寸、步长与覆盖格数的 `UnsupportedThumbnailGeometry`，在图像核心计算之前拒绝，不复制或猜测原生未覆盖内存。320×240等完整覆盖尺寸仍按原恢复算法处理。该拒绝是研究实现边界，不是已证原厂返回码。验证包含1445宽、1221高反例和全尺寸区间分类计数。

非对称亮度曲线见 `0x18000aed0`（`ev_12f4996b91d57ebc6cd62cc9a154943858df28088d971ef66aaac1ebe6785368`）：选择 LUT 输入分段 `i`，按

`A(v)=min(8191, min(8191, ((v-in[i])*slope[i]+32)>>6)+out[i])`。

完整 LUT 来自 `0x180011bc0`（输入）、`0x180011680`（斜率）、`0x180011920`（输出），保存在 `native-tables.json`，不能用任意 gamma 替代。

缩略图 `A(v)>>10` 得 8 档全局直方图 h0..h7；`0x180002a70/0x18000c2d0` 据最高三档与相邻档的数量比，调整 knot4..6，默认 knots 为 1024、2048、…、8192。对比值达到 2、2.5、10/3、5、10 时，分别减去 `[1,3,5,7,8]×scale`；scale 在 knot4/5/6 为 8/16/32，比较的是 h5/h4、h6/h5、h7/h6。零分母按原生比较顺序处理，不是除法。每段重新用倒数 LUT 求 slope，初始其余 slope=512。参考实现保留这些整数分支。

`0x18000afc0` 将 A 映射回等宽 8 档：每段线性映射到 1024 个整数，按 `+256>>9` 舍入，段内截到 0..1023。低端有单独的 0..8190 截断分支。此后称映射值 N（0..8191）。

## 4. 局部亮度直方图与噪声抑制型正则化

`0x180001f30`（`ev_22054630b65979110a572ba17ae460f540549681abb1fe004820f83266c4d549`）把 80×60 的 `N>>10` 标签图两侧扩展 16 点，形成 112×92 缓冲。边界为包含边缘样点的镜像延拓（`bcb0/b940/bdc0`）。

核心 `0x18000ba60`（`ev_c4c68c2461f4a773ff500404b73aea8c86f1abf8867e9103480047deef5e66a6`）在每个缩略位置，以 33×33 核统计 8 个亮度类别的加权频数。核系数取 `0x180010600` 共 1089 个 int16，值 1..5，中心对称，总和 2001。每档输出 `H_b=(min(weighted_count,2001)*131)>>6`，范围至 4095。**这不是对 RGB 做普通高斯模糊，而是对亮度分档的局部直方图作空间加权。**

`0x18000d100`（`ev_2412843d3f07882f5711bacd7a02a261db8cb108e8af8bdda57aa02d0b8b89a8`）对各档分别：

```
t = min(12288, (Slp1[b]*H_b+8)>>4)
q = piecewise_tanh_approx(t)      # 七段整数 LUT，下面列出
u_b = min(4095, (Slp2[b]*q+128)>>8)
remain = max(0,4095-sum(u_b))
V_b = min(4095,u_b + min(4095,(Bin[b]*remain+512)>>10))
```

tanh 输入节点 `[0,1755,3510,5266,7021,8777,10532]`、斜率 `[60,43,24,12,5,2,1]`、输出节点 `[0,1655,2845,3514,3838,3984,4048]`，分段计算 `min(4095,min(4095,((t-in)*slope+32)>>6)+out)`。这里是 Hamal 内的直方图条件化；不要把它等同后续 OlySpica 的外部降噪。

BlendMode=1 时 `0x18000b3e0`（`ev_5f2a6ca4ad0d1c930c6f09745f467efd8f2326baa1405ad071fa339a97a88dc8`）将 8 档转 16 档：

```
P[0] = V[0]>>1
P[2i] = (3*V[i]+V[i-1])>>3              # i=1..7
P[2i+1] = (3*V[i]+V[i+1])>>3            # i=0..6
P[15] = V[7]>>1
C[b] = sum(P[0..b])
```

累计步骤 `0x18000c920`（`ev_38c0d3b9eceef37ece6857aad4712e43cf694faefdc14efc03722cb631c1bbfe`）形成每个缩略格的局部映射曲线。

## 5. 插值回全尺寸、增益和颜色

对全尺寸每点仍从 `max(C0,C1,C2)` 计算 A、N。`0x18000c9e0`（`ev_f3246d9a1e8670d236fe3555c409b317f75b6682eac86d0f714a034d381030f1`）在亮度方向以 `b=N>>9`、`r=N&511` 对累计曲线作线性插值（0 档下界为 0），使用 `+256>>9`。然后在 80×60 网格以 `x*sx,y*sy` 的 13 bit 小数作双线性插值（`+4096>>13`），范围截到 4095，得到 T。空间坐标在边缘 clamp；与缩略图取样的半像素偏移不同，参考实现保留此差异。

原生 `0x18000c040`（`ev_446eef3200d7411a48acccc3b40977d1313fb79cc2a3255be23c492345456a65`）用 `0x180014800` 的 129 项倒数 LUT 近似 `16*T/maxComponent`。分母归一到 8192..16383，间隔 64 插值，再按移位恢复；原型 `_reciprocal_product` 保留此数值路径。设该比值为 R：

`gain=clip(((Strength*min(R,32767)+32)>>6)+4*(64-Strength),0,32767)`。

maxComponent=0 时直接用 `clip(4*(64-Strength),0,32767)`。因此 Strength=0 给 256，作为不增强的增益；输出仍经过颜色矩阵舍入。

颜色转换 `0x18000b200` / `0x180001810` 使用以下整数矩阵，`round10(v)` 是除 1024 的对称半上舍入，先转 Y/Cb/Cr 并截到 0..255：

```
Y  = round10( 74*C0 + 732*C1 + 218*C2)
Cb = round10(512*C0 - 395*C1 - 117*C2) +128
Cr = round10(-47*C0 - 465*C1 + 512*C2) +128
```

`0x18000b7a0`（`ev_852fa445074e812e4fc3970321716931937107d27b692bb36fa8598996c3ae76`）计算 `Y'=min(255,(Y*gain+128)>>8)`，`gC=clip(((Saturation*gain+32)>>6)+4*(64-Saturation),0,32767)`，色差 `Cb'=clip((gC*(Cb-128)+128)>>8,-128,127)`，Cr 同理。再转回并截到 0..255：

```
C0'=round10(1024*Y'+1901*Cb')
C1'=round10(1024*Y'-191*Cb'-479*Cr')
C2'=round10(1024*Y'+1613*Cr')
```

## 6. 独立实现的合同与剩余边界

`hamal8_reference.enhance(uint8_HxWx3, Settings, return_intermediates=True)` 返回图和所有关键中间量，包括缩略图、全局档计数、自适应节点、8 档局部直方图、条件化结果、16 档累计曲线、T 与 gain。实现不执行设备行为、文件 IPC 或原厂 DLL。

未覆盖：14 bit 分支、其他 Asym/Blend 模式、超过内存阈值后的条带拆分边界、外部 Spica、实际导出中的 RGB/BGR 格式握手和原厂失败后的内存生命周期。当前实现一次处理整个输入。它在完整 8 bit 图像核心上足以供独立复现和数值审查，尚不能替代原厂应用。

单幅输入若多个真实亮度/纹理都已被传感器/8 bit 保存截到同一值，增强函数拿到的输入信息完全相同，无法唯一恢复截断前细节。这是输入信息限制，不是对原厂成像效果作未经实测的断言。相应反例见合成验证。
