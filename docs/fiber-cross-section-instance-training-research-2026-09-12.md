# 纤维横截面实例分割训练方案调研：8GB RTX 起步

调研日期：2026-09-12。用户已确认：数据是**纤维横截面**，数据量暂未统计；本地有 RTX 30／40 系显卡，最低需要兼容 8GB 显存。

本文按“8GB 设备能够进行至少一种实用的训练／微调，也能够部署推理”规划。已核查当前项目源码和上游官方文档／源码；没有读取用户的完整业务数据集，也没有安装训练环境、启动训练或完成 RTX 实机测试。下文的 batch、分辨率、样本规模和显存预算都是实验起点，不是实测保证。项目核查起始 HEAD 为 `d969d05`，当时工作区存在其它改动。

**建议采用二维实例分割的预训练微调，保留 COCO RLE 作为主数据；首轮比较 RF-DETR-Seg Nano 与 YOLO26n-seg，StarDist 2D 在形状检查通过后加入。**沿用现有 YOLACT 作为实际效果对照。模型选择以横截面的实例分离、孔洞、面积误差和复核成本为准。

若只能先推进一条：需要扣除孔洞、保留不规则截面时，优先 RF-DETR-Seg Nano；若目标始终是完整的单一外轮廓、优先快速建立 8GB 基线，则先 YOLO26n-seg。暂时未统计数据，不影响开始数据审计和小规模微调试验。

**1. 先确定横截面标注的测量含义。**

一个实例应对应一个纤维横截面；相邻、接触的两个截面仍是两个实例。以下三个量需要区分：

| 目标 | 标签含义 | 对模型选型的影响 |
| --- | --- | --- |
| 外轮廓包围面积 | 外轮廓内全部计入，包括内部中腔 | 单外轮廓 YOLO 标签可适用；近似星凸时可考虑 StarDist |
| 纤维实体截面积 | 外轮廓内扣除真实孔洞／中腔 | 主数据必须保留孔洞；优先独立栅格掩膜／RLE 路线 |
| 外轮廓与中腔分别测量 | 保留对象身份，并分别输出外形和内部区域 | 首版可先保留完整实例 mask 和孔洞信息，后续再评估专门的内部区域预测 |

项目已有孔洞表示能力，因此建议主数据保留原始语义，训练用派生格式另行生成。把中腔填满、把细颈截面断成两个对象，可能只改变少量像素，却会改变面积、数量或类别统计。

“横截面实例分割”和“棉／粘纤／莱赛尔／莫代尔等类别判断”也应分别评价。类别可靠且均衡时可直接训练多类别模型；类别混乱或稀缺时，可先训练单类 `fiber` 的实例分割基线，再增加分类实验。主数据中的原类别始终保留，不能因为单类实验而丢弃。

**2. 当前项目已有数据闭环，但还需要训练适配层。**

| 已核查的项目能力 | 训练时的实际含义 |
| --- | --- |
| [训练导出说明](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/docs/training-data-export.md:3)和[导出服务](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/services/dataset_export.py:715)支持 COCO、YOLO、实例标签图 | 可以从现有面积对象构建监督数据，主程序不需要先加入训练框架 |
| [COCO 导出](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/services/dataset_export.py:751)采用未压缩 RLE | 可以保留孔洞、一个实例的多个连通区域及独立实例的重叠 |
| [完整性预检](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/services/dataset_export.py:311)仅提出警告 | 有面积标注、甚至成功导出，都不等于整幅图已完整标注 |
| [普通图片样本构建](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/ui/main_window.py:14226)按面积测量对象收集标签 | 线段测径、计数点不能直接变成实例 mask；还要复核存量面积对象的质量 |
| [数字切片样本构建](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/ui/main_window.py:14472)保留焦层、原点、覆盖与来源信息 | 训练必须使用对应的源视野／焦层，不能拿显示截图或另一焦层代替 |
| [数据划分](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/services/dataset_export.py:681)按 `source_group_id` 划分 train／val | 已避免基本的同源切块泄漏，但没有独立 test；跨项目重复来源、同一样品不同切片仍需更高层分组 |
| [现有面积引擎](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/runtime/area-infer/app/engine.py:232)加载 YOLACT | 新模型的 `.pt`／`.pth` 不能直接作为 YOLACT 权重替换 |
| [旧引擎轮廓转换](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/runtime/area-infer/app/engine.py:356)只取最大外轮廓并简化 | 新方案应传递权威 mask／RLE，避免模型预测正确、结果回写时又丢掉孔洞 |
| [ONNX Provider](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/services/model_provider.py:42)是 CPU 二值分割接口 | 新实例模型仍需专门的输入、输出解码及 GPU 适配，不能只更换 ONNX 文件 |
| [离线引擎服务](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/services/segmentation_engines.py:60)没有推理方法 | 当前能管理 μSAM／SAM3 引擎包，不代表已经完成这些模型的训练或推理接入 |

现有[模型类别表](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/runtime/area-infer/app/model_metadata.py:18)保存了多组纤维类别，部分权重的实际索引顺序与显示名称顺序不同。新数据集和模型必须保存明确的 `class_id → 类别名` 映射；不能沿用文件名猜测类别顺序。

合并旧数据时，还要检查类别粒度。例如已有“再生纤维素纤维”和更细的类别名称时，应先统一标注口径或建立层级映射，不直接把粗类与其细类当成互斥同级标签。保留原始类别和映射依据，不确定类别的实例另行复核。

**3. 三个主要候选各自解决什么问题。**

| 候选 | 在本项目中的定位 | 8GB 起步策略，均待实测 | 主要限制 |
| --- | --- | --- | --- |
| **RF-DETR-Seg Nano** | 需要孔洞、不规则边界、多类别输出时的主候选 | 384 输入、batch 1、混合精度、梯度检查点、梯度累积 | 默认查询数有限；训练资源和数据适配成本高于 YOLO Nano |
| **YOLO26n-seg** | 快速建立可训练、易部署的基线；后续比较 s | 640 输入、batch 1～2、AMP；比较 mask ratio 4 与 2 | 标准 polygon 标签不能无损表示孔洞和复杂多连通实例 |
| **StarDist 2D** | 接触密集、近似星凸横截面的专项对照 | 256／384 patch、batch 2，从普通精度开始 | 形状表示有约束；孔洞与明显非星凸形状不适合直接表达 |

RF-DETR-Seg 的官方 Nano／Small 默认输入分别为 312／384，参数量约 33.6M／33.7M。因此，Nano 并不是与 YOLO Nano 同量级的参数规模；其名称主要表示该系列内部的配置档位。这里建议从 384 做 FDM 的初步实验，是为了减少小横截面被过度缩小，仍需检查显存和像素精度。[RF-DETR 分割模型说明](https://rfdetr.roboflow.com/latest/learn/run/segmentation/)

RF-DETR 当前源码的 COCO loader 明确支持压缩与未压缩 RLE，这是它适配 FDM 孔洞标签的重要依据。但部分教程仍只展示 polygon；不能把任意旧版本默认视为兼容。正式实验应锁定包含该实现的发行版本或提交，并验证“导出 mask → loader tensor”逐像素一致。[官方 COCO loader](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/datasets/coco.py)、[相关变更记录](https://github.com/roboflow/rf-detr/blob/develop/CHANGELOG.md)

RF-DETR-Seg Nano 和 Small 在核查时都默认 `num_queries=100`、`num_select=100`。对一块包含数百个截面的图，这是实质性容量限制。首轮优先裁成实例数量显著低于上限的块，例如常见块包含约 20～60 个实例，并检查最密集块；不把 60 当作统一硬阈值。确需增大查询数时，要核对预训练权重适配并重新测显存，不能只改输出 top-k。分割输入尺寸还有整除要求：Nano 为 12 的倍数，Small 为 24 的倍数。[官方配置源码](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/config.py)、[模型变体说明](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/variants.py)

YOLO26n-seg／s-seg 已有预训练权重、训练、验证及 ONNX／TensorRT 导出路径。官方列出的融合后推理参数量分别约 2.7M／10.4M，不能直接当作训练时参数量或显存用量；其 COCO 速度也不是本项目 RTX 上的实测速度。[Ultralytics 实例分割说明](https://docs.ultralytics.com/tasks/segment)

YOLO 的 `mask_ratio` 默认 4，会降低训练标签 mask 的空间分辨率；建议在同一验证集比较 4 与 2，必要时再试 1。`overlap_mask=True` 会把实例合入一张标签图，重叠处由较小实例覆盖较大实例；需要保留独立重叠监督时应评估 `False` 的显存代价。推理时提高 mask 输出分辨率，不能补回训练时已经丢失的孔洞或细节。[Ultralytics 训练参数](https://docs.ultralytics.com/modes/train)

StarDist 对团块状、近似星凸对象有针对性，但“横截面”不自动意味着“星凸”。建议先把现有真值轮廓用 64／128 条射线重建，检查每类的面积偏差、轮廓 IoU 和最差样本；若凹陷、细颈或孔洞在表示阶段就明显失真，继续训练也无法消除这类表示误差。官方 FAQ 中的形状重建经验阈值不是本项目测量精度要求。[StarDist 官方 FAQ](https://github.com/stardist/stardist-docs/blob/main/docs/faq.md)

StarDist 也支持实例分类配置，并非只能输出无类别对象。不过它使用 TensorFlow，而项目当前主要依赖 PyTorch／ONNX。Windows 上现代 TensorFlow 的官方 GPU 路径为 WSL2，原生 Windows GPU 支持停留在 TF 2.10；因此它适合独立环境试验，部署成本也要计入选型。[StarDist 模型配置](https://github.com/stardist/stardist/blob/main/stardist/models/model2d.py)、[TensorFlow 安装说明](https://www.tensorflow.org/install/pip)

其它候选的建议位置如下：

| 候选 | 建议 |
| --- | --- |
| 现有 YOLACT | 先测当前权重的召回、面积偏差与修正时间；若与现有数据域接近，可再做一次低成本微调对照 |
| TorchVision Mask R-CNN R50-FPN | 需要成熟的独立 mask 训练接口时作为备选；从 512～640、batch 1 做资源试验，不预先保证 8GB 结果 |
| μSAM | 优先辅助标注和困难边界修正；如需定制交互能力再微调，不必把它作为首轮全自动主模型 |
| CellposeDINO-B／Cellpose-SAM | 显微实例分割的补充对照；本项目的孔洞、类别体系和 8GB 训练必须单独验证 |
| SAM3、完整三维实例模型 | 当前二维横截面需求没有显示出必须引入它们的理由，首轮暂不投入 |

TorchVision 官方微调接口直接接收每个实例的二值 `masks`，可作为 RLE 解码后训练的成熟参考。[官方实例分割微调教程](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial)

μSAM 官方资源表列出了 8GB、ViT-B、batch 1、仅训练 prompt encoder／mask decoder 的配置；当前源码低显存配置也包含 ViT-T，与文档表并非完全一致。不能把这些证据写成“8GB 支持 ViT-B 全参数训练”。[μSAM 训练资源说明](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#training-your-own-model)、[当前训练配置](https://github.com/computational-cell-analytics/micro-sam/blob/main/micro_sam/training/training.py)

Cellpose 当前已提供较小的 `cpdino-vitb`。但其训练源码会将 BF16 网络转回 FP32，并使用 AdamW；推理省显存不等于训练也一样省。它的实例标签接口也不能直接理解 FDM 的特殊忽略值。[官方模型清单](https://cellpose.readthedocs.io/en/latest/models.html)、[训练源码](https://github.com/MouseLand/cellpose/blob/main/cellpose/train.py)

**4. 数据准备中，最先处理这些会直接影响训练有效性的问题。**

**完整标注。**一幅视野中有 100 个目标、只画了 20 个面积对象，常规监督训练会把另外 80 个目标当成背景。首轮使用已经完整复核的视野／ROI；只标少量对象的图片可以继续用于补标，不直接当完整监督样本。空图也要确认是无目标负样本。

**YOLO 跳过对象可能制造假背景。**[当前导出逻辑](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/services/dataset_export.py:838)默认跳过孔洞／多连通实例，但仍导出原图。若直接训练，图上被跳过的真实目标就成了未标目标。需要保留复杂拓扑时使用 COCO RLE；YOLO 派生集可选完全兼容的整图／完整 ROI，或在明确采用外轮廓测量口径的独立实验中转换全部相关标签。不能只删难标实例后继续使用原图。

不同实验若使用不同训练子集、类别范围或面积口径，结果中应明确标注，不能把它们作为完全同任务的模型排名。正式比较优先使用共同、完整且口径一致的评估集。

**忽略区域必须被训练器真正忽略。**[实例标签 TIFF](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/services/dataset_export.py:908)用 `4294967295` 标记重叠／缺失区域；语义标签用 `65535`。Cellpose、StarDist 等 loader 未必识别这些值，可能将其当成异常实例。COCO 能保存独立重叠 mask，但 FDM 当前 COCO JSON 没有通用的像素级 valid-mask 监督协议；不能因此假定缺失图块自动不参与背景损失。首版优先只抽取覆盖完整的 ROI；后续若使用不完整覆盖，要实现并验证 loss 层的 ignore 逻辑。不能简单把未知区域改为背景零值。

**数字切片局部样本要重新检查完整性。**当前按标注时的焦层／局部快照分组，并从该组面积对象生成标签。如果同一目标区域在多个相交视野中分别标注，应核查每个导出块中可见的所有对象是否都带入，不能仅凭全局“已完整标注”勾选推定局部块完整。来源版本变化、焦层未核实、边界截断应进入审计清单。

**先分组，再切块，再增强。**建议按实体样品／制样批次建立稳定分组；至少不能低于原始图片或整张切片。同一样品的相邻视野、全部焦层、派生图、增强图和重复导出留在同一集合。当前的图片 document ID 或切片路径不一定能识别跨项目复制件，需要在训练数据清单中统一来源身份。

来源充足时，可按组做约 70%／15%／15% 的 train／val／test，并兼顾类别覆盖。来源很少时做分组交叉验证，另外积累真正新样品测试集。只有一个独立来源时，切出数千块也不能构造可信的独立测试。导出的 `all` 目录不应被重复指定为 train 和 val 后报告泛化效果。

**保留原始位深、尺度和固定类别表。**高位深图保留原件，训练另做一致的输入转换；[当前可选 uint8 转换](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/services/dataset_export.py:559)是逐样本有限值 min-max，未必是最适合类别判断的归一化策略。RGB／灰度、位深、μm/px、照明／染色和物镜信息应入数据清单，推理复用同一套预处理。

**5. 第一版数据集不需要追求很大，先盘点独立性和质量。**

应先生成一份数据审计表，至少包含：

- 独立实体样品、制样批次、原图／切片、可用完整 ROI、实例数量。
- 每类实例数及其覆盖的独立样品数，避免少数类全部来自同一张图。
- 孔洞／多连通实例比例、接触密集比例、每块实例数的中位数／P95／最大值。
- 输入中的截面直径像素分布、物理标定、不同倍率和采集条件。
- 来源未核实、局部未标全、截断、缺失覆盖、重复图片和疑似错误类别数量。

建议先用约 20～50 个完整视野／ROI 做数据加载与训练流程试验；随后整理约 200～500 个完整裁块作为首轮比较集。若密度允许，可以争取覆盖数千个实例，但这些数量只是工作量规划参考，不是最低训练门槛。比增加相似裁块更有价值的是增加独立样品、少数类别、制样差异和困难边界。

如果盘点后只有少量独立样品，仍可开始微调并绘制学习曲线，但应把结论限定为当前数据范围。没有数据统计前，不应承诺达到某个准确率，也无法可靠估算训练总时长。

**6. 分块与增强以保留横截面的像素细节为前提。**

建议先比较原图尺度附近的 512／640 裁块；RF-DETR 可用较小原始裁块或适度缩放得到 384 输入。训练裁块不是越大越好：大块压缩到小网络输入，会让小截面和中腔消失。RF-DETR 的默认 312 输入不能直接作为 FDM 的精度结论。

可先把“较小截面在网络输入中直径仍有约 16～24 像素”作为检查起点，细小孔洞还需要单独统计；这个像素范围不是计量保证。原始分辨率不足时，单纯上采样不会增加真实边界信息。

增强从随机翻转、旋转、适度亮度／对比度变化开始。模型若还需识别纤维类别，颜色扰动必须符合实际采集变化。第一轮关闭强 Mosaic、MixUp、Copy-Paste 和明显弹性形变，先获得容易解释的基线；再把增加某种增强作为独立实验。

块边缘的截面保留“截断”标记，标签与裁块一起变换；首轮测量误差统计使用完整可见实例。整片推理可采用约 15%～25% 重叠，依据实测对象尺寸调整，用 mask 和位置去重；计数与测量必须在去重后的对象集合上计算。

**7. 8GB 的训练配置应从保守设置开始。**

| 显存档位 | 建议起点 | 优先增加什么 |
| --- | --- | --- |
| 8GB | YOLO26n：640、batch 1～2、AMP；RF-Seg Nano：384、batch 1、检查点、累积 8～16 步 | 先完成含验证阶段的稳定运行，再试 batch 或输入尺寸 |
| 12GB | YOLO26n／s；RF-Seg Nano／Small，batch 1～2 起步 | 若边界是主要问题，优先更合适的输入尺度，再增加模型大小 |
| 16GB | 小模型更高分辨率或较大物理 batch；增加 StarDist／显微模型对照 | 比较面积精度与训练效率 |
| 24GB，例如部分 3090／4090 | 可扩大上下文、batch 和候选模型范围 | 最终入选权重仍回到 8GB 目标机验证 |

同为 RTX 30／40 系，不同型号显存容量不同，应按实际显存分档。两张 8GB 卡常规数据并行也不会自然变成一张 16GB 卡来容纳单个大样本。

RF-DETR 官方给出 8GB 的小 batch、梯度累积与检查点配置，但也明确硬件表是起始示例，资源需求取决于任务、尺寸等。这里采用更保守的 Nano／384 起步；它依然需要本项目的密集横截面数据验证。[官方资源配置](https://rfdetr.roboflow.com/latest/learn/train/advanced/)、[参数与显存说明](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/)

下面是拟采用的实验配置示意，不是仓库已经存在或已验证的训练入口。先完成数据适配、锁定包版本和预训练权重，再执行。

```python
# RF-DETR：数据应已转换为本地 train/valid/test COCO 目录。
from rfdetr import RFDETRSegNano

model = RFDETRSegNano(
    resolution=384,
    gradient_checkpointing=True,
    amp=True,
)
model.train(
    dataset_dir="prepared_fdm_coco",
    output_dir="runs/fdm_rfseg_nano_384",
    epochs=100,
    batch_size=1,
    grad_accum_steps=8,
    lr=5e-5,
    lr_encoder=5e-5,
    multi_scale=False,
    expanded_scales=False,
    checkpoint_interval=5,
)
```

RF-DETR 的 `gradient_checkpointing` 属于模型构造参数，不能照旧教程放进 `train()`。固定尺寸用于先控制峰值显存；学习率是小数据试验起点。当前配置的 `mask_downsample_ratio` 涉及模型结构／权重兼容性，不应当成与 YOLO `mask_ratio` 完全等价的开关随意修改。首轮先通过裁块尺度和输入分辨率保证小截面细节，再做结构改动实验。[构造参数要求](https://rfdetr.roboflow.com/latest/learn/train/advanced/)、[模型配置与兼容性检查](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/config.py)

```python
# YOLO：只用于标签语义完整、拓扑适配已通过检查的数据版本。
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
model.train(
    data="prepared_fdm_yolo/data.yaml",
    project="runs",
    name="fdm_yolo26n_640",
    imgsz=640,
    epochs=100,
    batch=1,
    amp=True,
    mask_ratio=2,
    overlap_mask=False,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    patience=25,
    workers=2,
    seed=20260912,
)
```

两套 API 首次使用命名权重可能下载文件；离线运行时提前准备，并使用已记录的本地权重路径。YOLO 的训练参数由官方接口提供，上面关闭部分增强、选择低 batch 是本项目的实验建议。[训练 API](https://docs.ultralytics.com/modes/train)

RF-DETR 的本地 COCO 快速入口通常要求 `train/_annotations.coco.json`、`valid/_annotations.coco.json` 等；FDM 当前是 `coco/annotations/instances_train.json` 和 `images/train/`。需要重排或建立适配目录，并同步修改 `file_name`、类别映射及集合清单；不需要上传数据到 Roboflow。[官方数据目录约定](https://rfdetr.roboflow.com/latest/learn/train/dataset-formats/)

8GB 资源验收应覆盖：代表性和最密集裁块的前向、反向、第一次优化器更新、验证及保存恢复；仅跑一次前向不够。先用几十个 step 检查峰值，再完成一个训练／验证周期。记录 `max_memory_allocated`、`max_memory_reserved` 和显卡总占用，并给 Windows 桌面留出余量；可以先以总占用约 6.5～7GB 为预算目标，再按目标机器调整。

梯度累积扩大的是一次优化更新的样本数，不能消除 batch 1 的激活峰值。发生 OOM 时先降物理 batch、关闭放大尺寸的增强、使用检查点、检查实例密度；再调整输入尺寸或冻结部分编码器，并记录精度影响。不能通过静默丢弃密集图中的对象来“满足 8GB”。

**8. 评价必须覆盖测量误差，不能只看框检测成绩。**

| 指标 | 建议的统计方式 |
| --- | --- |
| 实例召回、精确率、mask AP50／AP75／AP50:95 | 全体及每类分别统计，统一评估实现、输入还原和阈值 |
| 合并／拆分、漏检／误检 | 每图和每独立样品记录；接触密集区域单列 |
| 面积相对误差 | 对一对一匹配实例计算绝对误差、带符号偏差、中位数与 P95 |
| 等效圆直径误差 | 用相同面积口径和物理标定计算，不替代其它业务定义的直径 |
| 孔洞与边界 | 孔洞丢失／误填、边界距离、小截面与小中腔分组表现 |
| 类别判断 | 混淆矩阵、macro-F1、每类召回；类别错误与分割错误分开记录 |
| 测量汇总 | 去重后的数量、各类面积总量、必要时下游比例的偏差 |
| 人工修正成本 | 固定一批样品，记录每百个实例的修正次数与总耗时 |
| 运行资源 | 8GB 目标机显存峰值、冷／热启动、每块及整片延迟、长期批处理稳定性 |

配对面积指标不能只报“识别成功且容易的对象”；同时报告所有未匹配对象，防止通过漏掉难例获得漂亮的面积误差。人工对象的几何面积与栅格化 mask 面积也可能不同，应统一像素网格做模型对照，再单独测量栅格化／轮廓回写带来的业务误差。

以圆形截面为例，面积为 `A` 时，等效圆直径为 `2 × sqrt(A / π)`；小误差下，直径相对偏差约为面积相对偏差的一半。面积与直径都应使用约定的孔洞口径，不能在不同模型之间混用。

RF-DETR 当前训练文档说明：检测和分割模型默认 best checkpoint 选择依据是 **box mAP**。FDM 不能直接把这个 `best` 当作测量最优模型，应保留周期 checkpoint，并按验证集 mask 指标、面积偏差和复核成本选取；若后续定制训练回调，应把选择规则固定下来。[官方 checkpoint 说明](https://rfdetr.roboflow.com/latest/learn/train/)

密集图还需检查评估器的 `maxDets`、模型查询数及推理输出上限。若另用扩大的检测数上限做业务评估，应同时报告配置，不把它与标准 COCO AP 混为一项。只在验证集调阈值和后处理，独立测试集保留到最终选择完成后。

训练前可让两次独立人工复核覆盖一小批代表性对象，测出边界与面积的人工重复性，作为模型误差是否可接受的依据。本文不替业务定义未经确认的合格阈值。

**9. 建议按四个阶段推进。**

| 阶段 | 工作 | 产出及进入下一步的依据 |
| --- | --- | --- |
| 数据审计 | 盘点面积对象、来源、类别、孔洞、完整性和密度；建立独立测试来源 | 数据审计表、固定类别表、分组清单、需补标清单 |
| 数据适配与小试 | 准备完整 ROI；检查 RLE／mask 往返；用 8～16 个裁块做小规模过拟合检查，再做 8GB 资源试验 | 叠加复核图、标签一致性结果、实际显存曲线；确认模型能学到所给标签 |
| 第一轮模型比较 | 当前 YOLACT + RF-Seg Nano；拓扑兼容时加入 YOLO26n；StarDist 先形状检查后决定训练 | 同一独立来源划分下的 mask、面积、分类、修正耗时报告 |
| 精调和部署验证 | 优胜路线比较分辨率、类别策略、少量增强及 n／s；必要时多种随机种子复跑 | 冻结的数据／权重／配置版本；8GB Windows 真实工作流验收 |

训练环境与 FDM 主程序分离。项目当前[旧 GPU runtime 锁定文件](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/runtime/area-infer/requirements.torch.gpu.txt:1)是 Torch 2.4.1／TorchVision 0.19.1，不宜直接把新训练框架混入该环境。先为入选路线准备独立环境，锁定 Python、PyTorch、CUDA wheel、训练包、预训练权重及数据清单；日志和样品保存在本地。

部署可复用现有[隔离 worker 生命周期](/Users/lishuyang/PycharmProjects/fiber-diameter-measurement/src/fdm/services/area_inference.py:175)，但新增模型适配器。输出建议包含每实例的 class、score、bbox、mask／RLE、源图／焦层／原点、输入变换和模型版本，经过人工复核后进入面积测量对象。先验证 PyTorch 推理与训练评估一致，再考虑 ONNX／TensorRT，并在转换后重新测面积与孔洞，不只检查“能够加载”。

第一份实现任务宜限定为“本地横截面训练数据审计、格式适配和可重复训练／评测脚本”。训练 UI 和正式画布接入放在模型效果得到证据之后。暂不提供训练时长承诺；先测每 epoch 时间，再结合数据量、早停和对照次数估算。

许可证随模型版本记录即可：RF-DETR 当前 Seg 系列列为 Apache 2.0；Ultralytics 采用 AGPL／Enterprise 路径；Cellpose 的代码许可、权重和训练数据条款需分别核对。FDM 自身 GPLv3 不能替代对具体组件及分发方式的核查。这些记录用于后续打包选型，不影响本次本地技术试验。[RF-DETR 许可](https://github.com/roboflow/rf-detr#license)、[Ultralytics 许可](https://www.ultralytics.com/license)、[Cellpose 官方说明](https://github.com/MouseLand/cellpose)

本文结论是可执行的实验顺序和适配边界，尚未产生模型优劣、准确率、训练耗时或 8GB 实机通过的结论。
