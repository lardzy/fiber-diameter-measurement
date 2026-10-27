# EFI 参数来源与高度数据合同

本补查连接“采集参数→算法浮点高度→编码→进程间元数据→APP显示/保存对象”。所有结论来自副本静态 CIL，不执行原厂程序集。这里的分辨率是编码比例，不是已证明的光学高度精度。

## 1. 频带权重输入是光学截止量

`OlybAcquisitionDomain.AcquisitionService.SGPreProcessTask` (`0x06000011`) 的 IL004a 调 `DaoHelper.GetLensCutoff`，然后 IL0055 将结果作为第二参数交给 `EFIParam.GetDefaultParam(tempPath, cutoff)`。不是直接把屏幕显示倍率作为参数。

`OlybMemoryDatabase.DaoHelper.GetLensCutoff` (`0x06000bf6`) 实际公式：

\[
c=\frac{2NA}{M_{objective}\,M_{optical}}\cdot\frac{1}{550\times10^{-6}}.
\]

`LensOpticalParamMaster..cctor` (`0x06000a1e`) 设置波长常量550。若把该波长解释为nm，则转换后的量纲是图像侧的 mm⁻¹，这是公式量纲推导，不是通过设备标定得到的新测量。

NA来自 `LensOpticalParamMasterDao.Find(lensId,opticalZoom)` 的精确配置查询；缺失时记录错误并抛 `WorkflowDaoException`。本方法没有插值或从倍率猜NA。物镜倍率来自当前LensMaster，变倍来自当前ZoomMagnification实体。SGSharp随后依据该截止量及INI可覆盖阈值生成3D/EFI各自的频带权重，详见 [EFI规格](../efi-3d/implementation-spec.md)。

同时上层覆写实际图宽高、通道、8/16位深、行步幅、层数与缓存预算。仅调用底层默认构造器并不足以复现实际采集profile。

## 2. 输入 Z 不是绝对台面位置

普通精细采集 `_OnImagedProcessAcquisitionFine`→`_EnqueueZstackImage` 传入 `float(Zpitch*CurrentStep)`，CurrentStep从1开始。`EFIInputImageInfo` 同时保存层号、ZHeight、图像指针和字节数，`_SGInputImageThread`将其交给 `EFIMV_InputImage`。

该Z是算法序列内的名义坐标。运动专项已证实内部Z移动量的nm量纲，但这条表达式没有读取实际位置、绝对起点或曝光时编码器位置；所以也不能凭该字段认定曝光与位置同步。倒序、方向和实际第一层位置都要独立记录。

独立数据模型建议（非原厂已有接口）：

`z_absolute = z_first_absolute + sign * (z_nominal-z_first_nominal) * unit_scale`。

这里三个标定量均显式提供，不能根据一张高度图猜测。对非均匀实际Z，保留每层实测/名义坐标，不用上述等距简式覆盖原值。

## 3. 浮点高度转编码：整数除法与容器位宽

`OlybImageProcessDomain.FineExtendFocusImageFilter.ExecuteExtendFocusImageSG` (`0x060001c7`) 接收原生输出的float深度数组，在成功分支调用 `_GetExtendHeightDataSG` (`0x060001cc`)。

令 `p=AcquisitionCondition.Zpitch`，`N=BuildingCount`。`DaoHelper.GetExtendZResolution` (`0x06000bf3`) 执行：

```
q = 65535 / N                # uint32 整数除法，不是浮点65535/N
helper = float32(float32(p)/1000) / float32(q)
s = float32(helper * 1000)
u = uint16(clamp(floatDepth * float32(1/s), 0, 65535))
```

转换为UInt16时截断；随后 `stelem.i4` 放入 **Int32数组**，以 `CamByte=4` 的工作缓冲区复制出去。因此该阶段是“16位数值域装在32位元素里”，不能按UInt16步幅直接解读工作指针。设置的实体 `ZResolution=s`。

普通路径首层名义高度为p，所以其编码约q，通常不是0；没有在该方法中先减第一层Z。负值和超范围值被夹边，丢失信息。N=0或N>65535的数值退化在原型中显式拒绝，不声称原厂拥有相同保护。

## 4. 进程间元数据与APP边界

`OlybProcessJoinDomain._ImageResolutionForImageProcess` (`0x0600007e`) 对普通3D图设置Z比例为 `GetExtendZResolution(p,N)*1000*1000`，约为上面s的1000倍。特殊快速模式另由Z上下界和BuildingCount求有效pitch。**不同边界有额外单位换算，不能复用一个无单位变量名。** 若p以nm表达，则s以nm/码表达，通信值数值对应pm/码。

`_GetHeightImageDataInfo` (`0x0600005e`) 仅在实际高度图传输分支赋真实宽高，声明 `Bits=16, Channels=1`，并复制该Z比例到 BaseImageDataInfo。非高度图分支仍声明16位但尺寸为0；接口存在不等于当前有高度图。

APP `CommonUtility.ImageEntityBase.GetZResolution` (`0x060005d6`) 从对应TIFFTagsEntity.ImageData.DataPerPixelZ读取比例。`ImageEntity.SerializeTifTagDescriptionData` (`0x060002e5`) 将取得的AcquisitionInfo.ImageDataPerPixelZ放入TiffTagDescData；`OlysDataStoreService.ImageDBStoreFacade.SetImage_CommonInfo` (`0x0600009d`) 将其传给DB参数 `p_db_image_data_perpixel_z`。这证明比例被写入保存用对象/数据库请求；没有运行实际TIFF导出或数据库事务。

APP显示还乘maker/user Z校正，表面校正再引入编码scale、offset、空间ROI。完整物理读回必须把这些项一起解析。当前数据合同已足以独立实现显式单位的高度图；**每种私有文件、TIFF Tag数值单位和真实导出结果仍须样本核对**，不声称已完成所有原厂文件兼容。

## 独立验证

[reference.py](reference.py) 提供光学截止量、float32量化步骤和显式原点映射；[validate_reference.py](validate_reference.py) 的 [13项检查](reference-validation.json) 通过，包括首层非零、整数bins、Int32/UInt16边界、量化误差、饱和、独立绝对原点和退化count拒绝。该检查不涉及真实光学准确度。

## 证据与定位

| 模块 | 完整成员证据 | 主要位置 |
|---|---|---|
| CTRL/OlybAcquisitionDomain.dll | `ev_891cb106c231976c46418566a57ec65e2fc1312be125613c325169caa6de500f` | [sg-preprocess-cil.txt](sg-preprocess-cil.txt)、[sg-output-cil.txt](sg-output-cil.txt)；原记录在[采集原证据](../../acquisition/raw/OlybAcquisitionDomain.dll.members.json.gz) |
| CTRL/OlybMemoryDatabase.dll | `ev_fc5f7d5ef1644e9899021e77c833b1615a828d67daf31b9c904498e0b9d5286d` | [lens-cutoff-cil.txt](lens-cutoff-cil.txt)、[lens-parameters-cil.txt](lens-parameters-cil.txt)、[z-resolution-cil.txt](z-resolution-cil.txt)；[原记录](../../motion/rea-evidence.json.gz) |
| CTRL/OlybImageProcessDomain.dll | `ev_9608f5546b072cfe45109c88ec2a520edaec01fb7864c17feeaacd091a863d73` | [fine-efi-cil.txt](fine-efi-cil.txt)、[sg-height-conversion-cil.txt](sg-height-conversion-cil.txt)；[原记录](../stabilization/phase-correlation-evidence.json.gz) |
| CTRL/OlybProcessJoinDomain.dll | `ev_b28983bf89a6f28ecaeb257f439dfe93b1e867e246a0d32a22aa3ce7e959c88f` | [processjoin-resolution-cil.txt](processjoin-resolution-cil.txt)、[height-image-info-cil.txt](height-image-info-cil.txt)；[原记录](../../entrypoints/rea-evidence-bundle.json.gz) |
| APP/CommonUtility.dll | `ev_54f8cd0938416c64dd8a72abe7dc812768890f4b82db502ab777d2f71c2b1cef` | [app-height-units-cil.txt](app-height-units-cil.txt)、[app-height-base-cil.txt](app-height-base-cil.txt)；[完整记录](managed-evidence.json.gz) |
| APP/OlysDataStoreService.dll | `ev_08a3dac24f297294df03e83f0a0b6fd2bd1cc09ea7a261fcf4b1a35cbc77ab63` | [app-common-info-cil.txt](app-common-info-cil.txt)；[原记录](../../deep-analysis/rea-evidence-bundle.json.gz) |

所有CIL转录通过原PE SHA和方法IL SHA核验后解码；完整身份在各原始Evidence的subject字段，汇总见最终证据索引。
