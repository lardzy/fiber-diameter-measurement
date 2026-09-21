# DSX1000 / OLS5000 专有文件结构与兼容性分析

分析日期：2026-09-21。范围：用户提供的本地样张、当前 FDM 代码及公开读取器的交叉验证。

**结论：这批样张可以在不依赖厂商 SDK 的情况下读取原始彩图、激光强度、高度及内置标定。DSX 是多页 TIFF，POIR 是封装 OIR 的 ZIP，MPOIR 是封装多个 POIR 和点位布局的 ZIP。主要结构和提取路径已经验证，正式软件导入功能尚未接入。**

本轮只新增研究脚本、报告和衍生产物，没有修改 `src/fdm`、项目依赖或锁文件。16 个输入文件的 SHA-256 已复核，内容未变。研究脚本只读原件，提取结果放在独立目录中。

验证覆盖 5 个顶层专有文件、9 组测量数据、14 个内部 OIR。已导出 32 张原始位深 TIFF；每张均完成 TIFF 写回读取一致性检查，并通过现有 FDM 栅格读取器逐字节核对。独立读取库对照了 42 个 OIR 原始分量，全部逐像素一致。这里的验证是本地结构、像素和元数据一致性验证，不是仪器计量校准、Windows 安装包或厂商全格式兼容性验收。

![单张样张提取预览](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/olympus-formats-2026-09-21/extraction-preview.png)

图中 DSX 高度使用归一化灰度预览，OLS 强度和高度采用文件内置 LUT。原始 16 位数据另外保存，预览不会替代原始测量数据。

**样本清单与实测结果**

尺寸以下均为“宽 × 高”；横向标定列单位均为 µm/px，OLS 数值已乘设备校正系数。

| 样本 | 文件字节数 | 主图尺寸 | 可提取内容 | X 标定 | Y 标定 |
| --- | ---: | --- | --- | ---: | ---: |
| DSX 单张 `1_0002_1.dsx` | 7,361,014 | 1200 × 1200 | RGB 彩图、16 位高度、缩略图、地图图像 | 5.295278410301 | 5.295278410301 |
| DSX 拼接 `merge_1_0001.dsx` | 75,661,768 | 6651 × 2265 | RGB 彩图、16 位高度、缩略图、地图图像 | 5.295278410301 | 5.295278410301 |
| OLS 单张 `26A054517_001.poir` | 7,977,462 | 1024 × 1024 | RGB 彩图、16 位激光强度、高度、INVALID 辅助层 | 0.126124217369 | 0.126293839717 |
| OLS 拼接 `反面_002_G001.poir` | 35,874,547 | 1016 × 4709 | RGB 彩图、16 位激光强度、高度、INVALID 辅助层 | 1.250871602584 | 1.250549112704 |
| OLS 多点位 `26W011304_260821_085523.mpoir` | 39,091,482 | 5 组，各 1024 × 1024 | 每组均有彩图、激光强度、高度、INVALID 辅助层；另有布局 XML | 0.126124217369 | 0.126293839717 |

两份 DSX 样张没有存储单独的激光强度通道。高度图不能冒充激光强度图。MPOIR 的五个点位是独立数据组，布局中均为 `stitching=false`。

32 张测量图的构成：DSX 2 组 × 2 张，加 OLS 7 组 × 4 张。未把缩略图、参考图和地图图像计入测量图数量。

原始目录：[DSX1000、OLS5000样张](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/.tmp/DSX1000、OLS5000样张)。完整来源清单、SHA-256、通道、标定及对照指标见 [evidence.json](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/olympus-formats-2026-09-21/evidence.json)。

**DSX：标准 TIFF 骨架，加两处 XML 元数据**

两份文件均以 `49 49 2A 00` 开头，是 little-endian classic TIFF。偏移 4 的 uint32 指向首个 IFD，IFD 链不按文件位置递增，不能靠顺序扫描猜页面。像素均未压缩。

| IFD 页面 | 内容识别 | 单张形状 | 拼接形状 | 数据类型 |
| --- | --- | --- | --- | --- |
| 0 | XML 中 `emImageData=Color` | 1200 × 1200 × 3 | 2265 × 6651 × 3 | uint8，RGB |
| 1 | `ImageDescription=THUMBNAIL` | 128 × 128 × 3 | 128 × 128 × 3 | uint8，RGB |
| 2 | `ImageDescription=HEIGHT` | 1200 × 1200 | 2265 × 6651 | uint16，小端 |
| 3 | `ImageDescription=MAPIMAGE` | 102 × 300 × 3 | 300 × 300 × 3 | uint8，RGB |

这里形状用数组顺序“高 × 宽 × 通道”。单张彩图像素从文件偏移 8 开始，共 4,320,000 字节；高度像素从 4,375,214 开始，共 2,880,000 字节。首 IFD 位于 7,355,190。脚本另存了每页 IFD、Strip 和描述字段偏移，方便十六进制复核。

元数据主要分布在：

- TIFF tag 270 `ImageDescription`：根节点 `TiffTagDescData`，含每层的像素尺寸、Z 换算、图像尺寸、图像处理设置、文件版本。
- TIFF tag 34665 指向的 EXIF IFD，其 tag 37500 `MakerNote`：根节点 `ExifTagDescData`，含物镜、变倍、台面位置、拼接行列、Z 扫描参数、校正信息等。

本批 `FileVersion=1.1.1.1`，软件字段为 `DSX 1.1.1.1`。拼接样张记录了 2 行 × 6 列及 10% 重叠，但本文件中的主测量像素已经是合成结果，没有发现 12 张完整原始视野可供重新拼接。单张虽然记录 `ZSliceTotal=21`，文件并未存储 21 张完整 Z 平面，不能把采集参数当作像素堆栈。

XML 声明为 `encoding="utf-16"`，但 tag 270 的 TIFF 数据类型是 ASCII，当前实际内容也是单字节文本。`tifffile` 已解码出的 Unicode 字符串可以直接交给 XML 解析器；把原始字节按声明盲目解为 UTF-16 会出错。非 ASCII 文本和其他软件版本仍需补样验证。

**DSX 标定：按图层取字段，从 pm 换算为 µm**

本批彩图读取：

```text
TiffTagDescData/ColorImageData/ColorDataPerPixelX
TiffTagDescData/ColorImageData/ColorDataPerPixelY

sx_um_per_px = ColorDataPerPixelX / 1,000,000
sy_um_per_px = ColorDataPerPixelY / 1,000,000
```

高度图对应 `HeightImageData/HeightDataPerPixelX/Y/Z`。优先使用各图层自己的字段；不要固定取顶层 `ImageDataPerPixelX/Y`，缩放保存后它们可能与最终彩图不同。公开的 DSX ImageJ 提取实现采用相同的 pm → µm 换算及图层字段选择，可作为旁证。[DSX 提取宏源码](https://github.com/peterjlee/asc-ImageJ-DSX-Stitching-Utilities/blob/main/Convert_Active_DSX_Image_And_All_Others_in_Same_Directory_and_Extract_HMaps.ijm)

两份 DSX 都给出 `ColorDataPerPixelX=5295278.41030118`，即 **5.29527841030118 µm/px**。单张视野约 6.354334 × 6.354334 mm；拼接图约 35.218897 × 11.993806 mm。

还做了样张内的交叉核对：单张导出 JPEG 右下角的 `1 mm` 标尺端点约相隔 189 px；元数据计算为 `1000 / 5.29527841030118 = 188.847483 px`，与像素取整一致。这比依靠物镜倍率猜比例更直接。

TIFF 中的 `XResolution=YResolution=96`、`ResolutionUnit=INCH` 是显示/打印 DPI，不能用于显微测量标定。`MapDataPerPixelX/Y` 属于地图层，也不能用于主彩图。

高度比例记录为：单张 **0.0774721130728722 µm/计数**，拼接 **0.088523343205452 µm/计数**。这里只确认了原始计数和比例字段；绝对 Z 原点、无效点及厂商高度显示规则尚未完整验证。彩图中的 Z 字段不应解释为彩图像素的物理高度。

**POIR：ZIP → 两个 OIR → 按 UID 索引读取图像层**

单张 POIR 的目录是：

```text
26A054517_001.poir                       ZIP / DEFLATE
  26A054517_001_LSM3D^3D_LSM.oir          12,990,025 bytes
    INTENSITY                            uint16
    INVALID                              uint16，声明有效位数 7
    HEIGHT                               uint16
    元数据、参考彩图、缩略图、LUT
  26A054517_001_COLOR3D^XY_Camera.oir       9,772,337 bytes
    RED / GREEN / BLUE                   各一个 uint8 平面
    元数据、参考彩图、缩略图、LUT
```

拼接 POIR 同样包含一个 LSM OIR 和一个 Camera OIR，主尺寸均为 1016 × 4709，图像已经完成拼接。不同 POIR 的成员顺序不同：多点位包内的 Camera 成员先于 LSM，不能把 ZIP 第一项固定当作激光数据。

POIR 使用 ZIP64 扩展和 data descriptor：local file header 中大小可为 `0xffffffff`，应交给支持 ZIP64 的读取器，以中央目录和扩展字段为准。Python 标准库 `zipfile` 已成功读取本批所有容器。

OIR 头部及索引的本批实测结构：

| 偏移 | 字段 | 本批含义 |
| --- | --- | --- |
| `0x00` | 16 bytes | ASCII `OLYMPUSRAWFORMAT` |
| `0x20` | little-endian uint64 | OIR 文件总长度 |
| `0x28` | little-endian uint64 | 文件末尾块索引位置 |
| 索引位置 | 4 bytes | `ff ff ff ff` |
| 索引位置 + 4 | uint64 数组 | 各块绝对偏移；本批各 OIR 均有 17 个索引项 |
| 各块偏移 | uint32 + uint32 | payload 长度、块类型；其后为 payload |

块类型本批确认：`0=metadata`、`1=frame properties`、`2=BMP thumbnail`、`3=UID`、`4=PIXEL`、`5=empty/null`。UID 块与后续 PIXEL 块配对；XML 字符串前 4 字节为其长度。探针只在已索引的元数据块内定位 XML，再按长度验证解析，不在整个像素区搜索标签。

这一索引式结构与公开的 `oirfile` 实现一致；本轮用 2026.9.6 版逐像素交叉验证。[oirfile 项目及格式说明](https://github.com/cgohlke/oirfile)

主图 UID 类似：

```text
t001_0_1_<channel-guid>_0
```

参考图则以 `REF_CAMERA0_...` 开头。本批主图都是单帧；`3D_LSM` 包含高度、强度和辅助平面，不代表文件中保留了原始三维焦面堆栈。参考图的尺寸和比例可能完全不同，不能替代主测量图。

**通道识别必须用 XML GUID，不应依赖块顺序或数组顺序**

主通道说明位于 `imageProperties/imageInfo/phase/group/channel`。LSM 的 `imageDefinition/imageType` 明确标识 `HEIGHT`、`INTENSITY`、`INVALID`；Camera 的 `elementChannel/elementType` 明确标识 `RED`、`GREEN`、`BLUE`。

本批 LSM 像素块实际顺序是强度、INVALID、高度，而 XML 中另一处是高度、强度、INVALID。Camera 主帧 `depth=3` 表示三个颜色分量，各 `elementChannel/depth=1` 才是每个分量的字节深度。按 GUID 匹配后，应组合为 `H × W × 3` 的 RGB 数组。

也实测了通用库的限制：`oirfile 2026.9.6` 的单张 Camera `asarray()` 分量顺序是 **GREEN / BLUE / RED**，拼接 Camera 是 **BLUE / GREEN / RED**；它没有从本批 Camera 元数据给出坐标比例。LSM 返回的 `coord_scales` 是名义像素长度，未包含设备校正系数。因此该库可用于像素读取/互证，但不能直接把返回数组顺序和标尺作为本项目的导入结果。这是本批文件的实测结论，详见 [OIR 逐像素对照记录](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/.tmp/olympus-format-analysis-2026-09-21/oir-crosscheck.json)。

**OLS 标定：像素长度和设备校正系数是两层数据**

读取主图通道的：

```text
imageProperties/imageInfo/phase/group/channel/length/x,y,z
imageProperties/imageInfo/phase/group/channel/pixelUnit/x,y,z
imageProperties/acquisition/microscopeConfiguration/pixelCalibration/x,y,z
imageProperties/acquisition/microscopeConfiguration/userPixelCalibration/x,y,z
```

本批长度单位明确为 `MICRO_METER`。已验证的 XY 换算是：

```text
sx = imageInfo.channel.length.x × microscopeConfiguration.pixelCalibration.x
sy = imageInfo.channel.length.y × microscopeConfiguration.pixelCalibration.y
```

当前 Gwyddion OIR 读取源码同样将图像像素长度乘以 `pixelCalibration`。不要引用旧版中“未应用校正”的实现来代替对当前样本的检查。[Gwyddion OIR 读取源码](https://browse.dgit.debian.org/gwyddion.git/plain/modules/file/oirfile.c)

| 数据组 | 名义 X / Y | `pixelCalibration.x / y` | 校正后 X / Y，µm/px |
| --- | --- | --- | --- |
| 单张及五个点位 | 0.125 / 0.125 | 1.00899373895262 / 1.0103507177364701 | 0.1261242173690775 / 0.12629383971705876 |
| 拼接 | 1.25 / 1.250549197906079 | 1.0006972820670001 / 0.999999931867928 | 1.25087160258375 / 1.250549112703571 |

多点位布局提供了很好的文件内独立字段核对：每个区域记录宽 **129151**、高 **129324**。按 nm 解释为 129.151 × 129.324 µm；用上面的校正后比例乘 1024 得到 **129.151198586 × 129.324891870 µm**，与布局值的整数截断吻合，差值均小于 1 nm。不乘设备校正系数只能得到 128 × 128 µm。

仅取 `0.125 µm/px` 会分别漏掉约 **0.899% 的 X 修正和 1.035% 的 Y 修正**。校正后单张的 Y/X 差异约 **0.13449%**，拼接约 **−0.02578%**，需要保留双轴比例。

本批 `userPixelCalibration` 全部为 1。非 1 时是附加乘数、替代系数还是已折入别处，本批不能独立证明；探针完整保留该字段并标出非单位值未验证，不静默套用未经验证的规则。DSX 的 `ImageCommonCalibrationValueX/Y/Z` 本批也都是默认的 1000000，非默认情况仍需补样。

另一个容易读错的位置：单张 Camera 元数据中有独立 `cameraChannel/length/x=10.045340447294397`，与主图 `imageInfo` 中的 0.125 不同。这与参考彩图路径同处元数据集合，不能用泛化搜索“第一个 length/x”作为主图比例。采集配置还存在 Z 步长 0.24；它也不等于最终高度计数的 Z 比例。

高度通道主图中的 Z 比例：单张 **0.0005843610835242034 µm/计数**，拼接 **0.007673651118093 µm/计数**；多点位各点位不同，已分别记录。单张另有 `frameProperties/additionalData` 的 `ZScales` 数组，1024 项均为 5030203193157，其完整编码和绝对原点语义未验证。可以保留原始高度并展示相对变化，但尚不应宣称已复现厂商所有高度/粗糙度计算。

**内置 LUT 和“原始数据”与“显示图”的区别**

OLS 除了原始像素，还存有按 channel GUID 关联的 LUT XML。当前 LUT 有 65536 项，每项 4 字节，前三字节可以复现 RGB 显示；第四字节在本批表中为 0，不应直接作为透明度导致图像全透明。像素显示范围来自 `acquisition/imagingParam/productData/scale[@ChannelId]`。

本批强度使用 `Gray_Gamma1.5`，高度使用 `Rainbow1`。按显示上下限把原始 16 位值映射至 0～65535 的 LUT 索引，已能重建与厂家 JPEG 十分接近的显示图。应保留 16 位原始平面，LUT 只影响显示；不要用 JPG 或伪彩色像素反推原始高度。

`INVALID` 层的 XML 名称已确认，但实测是覆盖 0～90 的连续整数，存储为 uint16，声明有效位数为 7；不是简单的 0/1 掩膜。没有证据支持把“非零”全部剔除，或把 90 固定解释为某一种错误类型。目前只应作为独立辅助层保留。

DSX 的原始高度层已经完整提取，但本轮没有复刻其厂家 H.jpeg 的伪彩色渲染。探针用灰度归一化预览，因此 DSX H 的 JPEG 差异指标只说明显示方式不同，不能作为高度解码失败或已复刻渲染的依据。

**MPOIR 和拼接布局**

```text
26W011304_260821_085523.mpoir             ZIP / STORE
  matl.omp2info                          XML，点位/区域/台面布局
  26W011304_009_X001_Y001_G001_A001.poir
  26W011304_010_X001_Y001_G002_A001.poir
  26W011304_011_X001_Y001_G003_A001.poir
  thumbnail
  26W011304_012_X001_Y001_G004_A001.poir
  26W011304_013_X001_Y001_G005_A001.poir
```

每个 POIR 再展开为 Camera + LSM。应按 XML 中 `matl:area/matl:image` 关联归档成员，保留 group ID、区域坐标、点位索引及源成员名。五组均为 1 × 1 区域且 `stitching=false`，可以按需选择点位/通道，不应自动合成为连续大图。

布局中引用了 `Map_A01.oir`，但 MPOIR 内没有该成员。缺的是被引用的地图文件，不影响五个已有点位的主图和标定提取。

拼接目录另附的 `matl.omp2info` 是 1 列 × 5 行的采集/拼接布局，指向 `反面_002_G001.poir`。现有包提供的是已拼接图，没有五张独立原始视野供重新配准。拼接 Camera 与 LSM 主图尺寸一致，已通过导出图对照，应直接读取现有合成平面。

**像素与显示验证证据**

1. 独立探针通过 OIR 索引读取像素，再按 XML GUID 关联通道；与 `oirfile 2026.9.6` 对应 GUID 分量逐像素比较，**14 个 OIR、42 个分量全部一致**。
2. 所有测量图以原始 dtype 写出 TIFF 再读取，**32/32 一致**。再经当前 `fdm.services.raster_io.read_raster_file()` 读取，**32/32 原生像素 SHA-256 一致**，证明 RGB8 / GRAY16 已能进入现有栅格表示。
3. metadata-only 模式枚举所有文件和标定，**14/14 OIR 的 PIXEL payload 解码读取计数为 0**，没有产生 TIFF/PNG。为审计原件，命令还会计算源文件 SHA-256；压缩 ZIP 为访问尾部索引仍可能需要解压前面的字节，不能把“未解码像素”宣传为“完全没有图像区 I/O”。
4. 原始 16 个输入文件的 SHA-256 在提取前后保持一致。
5. 选择 `--channels color` 实测只产生 **9 张彩图 TIFF**；7 个 LSM 成员均未解码其像素，验证了按需取图路径。

OLS 与随附 JPEG 的对照如下。MAE 是 0～255 通道值的平均绝对差。彩图及高度的“编码对照”先对提取/重建图应用参考 JPEG 的量化表和 4:2:0 色度抽样，再比较解码结果，单独控制 JPEG 对高频颜色的影响；它不是替换原始导出数据，也没有拟合空间配准或颜色变换。

| 样本 / 通道 | 直接对照 MAE | 直接对照最大差 | JPEG 编码对照 MAE | 编码对照最大差 |
| --- | ---: | ---: | ---: | ---: |
| OLS 单张彩图 | 3.87752 | 112 | 0.29003 | 6 |
| OLS 拼接彩图 | 9.24308 | 126 | 0.27511 | 5 |
| OLS 单张激光强度（内置 LUT） | 0.08701 | 1 | 0.12103 | 2 |
| OLS 拼接激光强度（内置 LUT） | 0.08654 | 1 | 0.12117 | 2 |
| OLS 单张高度（内置 LUT） | 1.67926 | 186 | 0.17354 | 4 |
| OLS 拼接高度（内置 LUT） | 2.32802 | 255 | 0.19370 | 4 |

原始 RGB 本来就不需要与有损 JPG 逐像素相等。编码对照后的低残差，以及独立读取器的原始像素完全一致，共同支持通道解码与排列正确；仍未声称 JPEG 字节或厂商整个显示管线完全一致。

DSX 彩图与导出 JPEG 的上方 90% 区域对照，单张 MAE 1.19415、PSNR 42.91 dB，拼接 MAE 1.32714、PSNR 35.77 dB。该区域用于排除底部标尺；其他叠加/边界和有损编码仍会造成差异，未宣称逐像素相等。

**与现有项目的接入关系**

当前仓库已有 `tifffile`、NumPy、Pillow，提取探针可以直接在已有 uv 环境运行。现有 `RasterPlane` 支持 RGB8、GRAY16、GRAY32_FLOAT，因此本批原生像素没有根本兼容障碍；工作重点是容器/通道、来源身份和标定语义。

| 当前落点 | 现状及所需变化 |
| --- | --- |
| [raster_io.py](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/services/raster_io.py:455) | 只接受普通图像后缀，TIFF 读取在第 719 行附近明确拒绝多页。仅把 `.dsx` 改为 `.tif` 或加到后缀列表，不能完成导入。建议独立显微容器读取服务返回选定平面及元数据。 |
| [image_loader.py](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/ui/image_loader.py:54) | `ImageLoadRequest` 主要用路径描述一张图。需要表达容器路径、内部 POIR/OIR、点位、通道和帧，支持异步枚举与按需读取。 |
| [main_window.py](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/ui/main_window.py:19227) | 新开图像目前按文件路径去重。同一个 POIR 的彩图和激光图、同一个 MPOIR 的不同点位，应有不同数据身份。导入过滤器、目录导入及拖放入口也要一并识别格式。 |
| [models.py](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/models.py:375) | `Calibration` 只有一个 `pixels_per_unit`。这不足以完整表示 OLS 的 X/Y 差异；应先设计双轴标定及旧工程兼容。 |
| [图像挂载与标定装载](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/ui/main_window.py:19773) | 现有流程读取 sidecar 并处理项目默认标定。设备内置标定需要带来源进入该流程，不能覆盖已保存的用户标定，也不能被默认标定无声替换。 |
| [RasterSemantic](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/image_processing_models.py:20) | 现有语义没有专门的物理高度。首阶段可把高度作为保留原始数值和 Z 元数据的独立图层；正式高度测量前再明确语义、无效值与单位。 |

建议读取边界表达以下三种操作：

```text
inspect(path) -> datasets、channels、尺寸、每轴标定、来源字段
read_plane(dataset_id, channel_id) -> RasterPlane + 元数据
read_calibration(dataset_id, channel_id) -> X/Y/Z 比例及其证据，不解码图像
```

“仅获取标尺”应优先返回结构化数值，后续由本软件生成标尺。它不要求读取/识别 JPEG 上已经画好的标尺。若把标定应用于另一个已导入的导出图，还需核对其来源、缩放和旋转；裁剪通常不改变像素间距，缩放则需要按尺寸比例换算，不能只按同名文件匹配。

双轴标定不能用平均比例直接冒充完整支持。例如线段实际长度应为 `sqrt((dx*sx)^2 + (dy*sy)^2)`，面积应为 `area_px*sx*sy`，折线应逐段计算。现有先求像素长度再调用 `px_to_unit()` 的路径需要相应调整，圆/椭圆等测量也需评估。若先交付仅横向标尺，界面和数据中应明确这一范围。

对这批 X/Y 相等的 DSX，可以映射为现有模型的 `pixels_per_unit=1/sx=0.18884748308127636`、`unit=µm`。OLS 单张对应的 X/Y 分别约为 7.928691419 / 7.918042576 px/µm，无法由一个标量同时精确表示。

实现次序建议：先做只读格式枚举、通道选择、标定提取；同时完成双轴标定设计，再接入彩图/激光图和工程持久化；最后扩展高度、辅助层和更多版本。可将提取结果保存为项目已有的无损资产，并在 `ImageDocument.metadata` 保留原始容器、成员、GUID、标定原字段及源文件指纹，避免同一容器不同图层混淆。

**仍需补样或进一步验证的边界**

- 非默认 `userPixelCalibration`、非默认 DSX `ImageCommonCalibrationValue`；厂家软件对照读数或带已知长度标准器的样本可提高标定确认强度。
- 其他 DSX/OLS 软件版本、纯二维拍照文件、不同倍率/变倍、缩放后的拼接、关闭某通道的文件。
- 大于 4 GB、压缩 TIFF、分块像素、真正的 Z/T 堆栈或 OIR companion files。本批未覆盖，研究探针也不宣称支持。
- `INVALID` 数值解释、ZScales 编码、绝对高度原点、厂商后处理及完整高度测量一致性。
- DSX 厂商 H 图伪彩色风格；本轮只验证了其原始 uint16 平面和比例字段。
- 地图缺失情况下的完整多点位空间浏览与拼接重建。本批可独立读每个已有点位，无须等待地图补齐。

**交付文件与复现**

- [结构探针 probe.py](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/olympus-formats-2026-09-21/probe.py)：研究级、只读原件，支持全通道、选择通道及 metadata-only；未接入正式业务。
- [独立读取器对照 verify_oir.py](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/olympus-formats-2026-09-21/verify_oir.py)：固定 `oirfile==2026.9.6`，按 GUID 比较原始分量。
- [汇总证据 evidence.json](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/research/olympus-formats-2026-09-21/evidence.json)：样本指纹、标定、通道、像素及显示对照。
- [完整提取和结构记录](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/.tmp/olympus-format-analysis-2026-09-21/extracted/analysis.json)：关联 32 张 TIFF、预览和每个内部 OIR 的 XML/块索引。
- [标定模式记录](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/.tmp/olympus-format-analysis-2026-09-21/metadata-only/analysis.json)；[FDM 无损读回与 metadata-only 检查](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/.tmp/olympus-format-analysis-2026-09-21/fdm-readback-and-metadata-check.json)。

在项目根目录运行（输出目录可另选）：

```sh
uv run --no-sync python docs/research/olympus-formats-2026-09-21/probe.py \
  --input '.tmp/DSX1000、OLS5000样张' \
  --output '.tmp/olympus-format-analysis-2026-09-21/extracted'
```

追加 `--metadata-only` 只生成结构/标定记录；追加 `--channels color` 只解码彩图，或 `--channels intensity` 只解码激光强度。选择模式仍会枚举其他成员的元数据，未做生产级缓存和 I/O 优化。

独立像素互证使用临时 uv 环境，不修改本项目依赖：

```sh
uv run --no-project --with oirfile==2026.9.6 --with tifffile --with pillow \
  python docs/research/olympus-formats-2026-09-21/verify_oir.py \
  --input '.tmp/DSX1000、OLS5000样张' \
  --output '.tmp/olympus-format-analysis-2026-09-21/oir-crosscheck.json'
```

本次结果：`oir_count=14`，`native_planes_equal=42`。公开读取器本身也注明格式来自样本逆向、支持范围有限；本报告的兼容结论以实际样本验证为边界。[oirfile 已知范围](https://github.com/cgohlke/oirfile#notes)
