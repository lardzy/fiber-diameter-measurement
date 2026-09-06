# 画布性能基准

连续魔棒识别、剔除和刚提交后的快速拖动，使用新增的交互基准：

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/benchmark_canvas_interaction.py \
  --image-size 4096 --objects 100 --vertices 10000 --subtracts 10 \
  --frames 90 --preview-baseline-ref 45c023d \
  --output .tmp/canvas-benchmark/interaction-4k.json
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/benchmark_area_pipeline.py \
  --output .tmp/canvas-benchmark/area-pipeline.json
```

第一条命令每帧移动 18 个逻辑像素，在连续交互期间安装一次几何编辑、提交一个
新面积并改变缩放；分别统计鼠标处理、画布绘制、事件分发、结果安装及后台等待。
`--preview-baseline-ref` 从指定本地 Git 提交读取旧魔棒预览函数，在相同数据与
Qt 环境下比较，省略时只测当前实现。第二条命令比较 2K/4K/8K 整图与局部
剔除后处理，并通过真实 MainWindow 入口测量新增面积的界面安装和撤销命令成本。

`input_to_qimage` 终点是 QImage 渲染完成，不能等同于操作系统显示呈现。
`interaction_dispatch` 是每帧之间的事件循环开销，必须一起检查，不能用快速鼠标
处理掩盖慢的后台结果回调。`settle_ms` 是动作停止到当前精确缓存就绪的等待；
等待期间可以继续使用同一文档的完整预览。首帧中的图片和载入提示也不等于全部
测量对象精确绘制完成。开发机记录见 [本轮实现与验证报告](canvas-interaction-performance-2026-09-07.md)。

下面的传统基准保留，用于精确画面的回归、缓存热帧和数字切片路径验证。

开发环境可通过离屏 Qt 画布运行确定性测量对象场景：

```bash
python -m fdm.canvas_benchmark --scenario length_labels_500 --json
python -m fdm.canvas_benchmark --scenario areas_holes_300 --json
python -m fdm.canvas_benchmark --scenario offscreen_5000 --json
```

普通图片画布为默认值；同一份确定性数据也可以选择数字切片画布路径：

```bash
python -m fdm.canvas_benchmark --scenario areas_holes_100 \
  --canvas-kind digital_slide --json
```

数字切片场景通过真实的 `DigitalSlideCanvas.set_slide_document()` 初始化，
使用非零全局 viewport origin，并在平移阶段调用 `move_viewport_by()`。基准使用
确定性的内存 store 提供 manifest 和 viewport raster，以免性能采样混入
临时 SQLite 创建和磁盘读取；后台扩大缓冲仅记录请求，不启动文件系统线程。
结果中的 `scenario.digital_slide` 会记录初始化路径、最终 origin、同步 viewport
渲染次数和缓冲请求数，防止该场景退化成普通图片画布的别名。

上述命令默认测试 `direct` 路径，即逐对象矢量/文字 sprite 直接绘制。要测试
精确缩放级别的被动叠加图块热帧，需显式启用：

```bash
python -m fdm.canvas_benchmark --scenario length_labels_500 \
  --overlay-cache --overlay-cache-timeout-ms 10000 --json
python -m fdm.canvas_benchmark --scenario areas_holes_300 \
  --overlay-cache --overlay-cache-timeout-ms 10000 --json
```

图块模式在冷帧后处理 Qt 事件，等待所有可见图块完成后才采集热帧。等待存在
明确超时；若超时，`render_path.effective_hot` 会标记为
`overlay_cache_timeout_fallback`，不得把该结果当成纯缓存热帧。

## 生产渲染架构与安全边界

画布先通过 `MeasurementSceneIndex` 的 RAW 边界做视口和 dirty region
空间查询，再对候选对象执行精确裁剪或命中判断。视口外对象的数量增加时，不应
迫使每帧重新遍历和构造所有面积路径。查询结果始终保持文档中的原始绘制顺序，
避免重叠对象的视觉层级发生变化。

面积几何严格分为两类：

- `RAW` 是项目中保存的原始 rings，也是测量值、质心、孔洞面积、命中、编辑、
  撤销、保存和图片导出的唯一几何来源；
- `SCREEN_PROXY` 只是经过拓扑和偏差校验的未选中对象屏幕显示代理。代理失败、
  超出缓存预算或被禁用时直接回退 RAW，不得修改持久几何或测量结果。

画面按被动层和活动层合成。所有已提交测量对象的普通主体及标签属于被动层，
可以在当前精确缩放级别缓存为透明图块；选中对象的高亮增量、控制点、悬停效果
和绘制/拖动预览属于活动层。活动层保留 RAW 几何，稳定的主体、剔除块和选中
反馈优先复用栅格缓存，活动提示点和控制点仍即时绘制。面积被选中时，被动层
保留普通填充和标签，活动层只补充加粗轮廓和控制点，既不会覆盖标签，也不会
改变半透明面积之间的文档层叠顺序。真正开始编辑或拖动时，才精确
恢复旧主体覆盖的局部背景，再绘制预览，背景在同一次拖动中复用，避免反复重画
相邻复杂对象。

可见画布在精确图块缺失时，先显示同一文档的完整预览；目标视口的图块全部
就绪后再切换到当前精确缩放画面。首次打开没有预览时显示原图和简短载入提示。
新确认的魔棒对象接续已经显示的草稿栅格，后台正式主体准备好后替换，避免在确认
后第一次拖动时同步重建 RAW 路径。预览只参与显示，吸附、命中、测量和导出
继续使用精确几何。隐藏画布的诊断/像素对照路径保留直接绘制，便于验证结果。

后台快照以不可变原始坐标字节和轻量绘制命令覆盖面积、直径线、折线、点和
标签。全工作区共享一个独立 Qt 栅格进程、最多两个桥接线程，避免原生复杂描边
占用 Python GIL 而拖慢 GUI 回调。进程只接收值快照，不持有窗口、测量对象或
源图片。对象变化增量更新索引和相关图块；切换仍打开的图片保留近期缓存，
关闭图片释放所属缓存。选中面积的可见性筛选使用保守标签边界，不能为了图块
筛选同步计算密集轮廓的精确质心。

被动图块使用全局图像坐标网格，每块为 `512` 个逻辑像素，四周额外保留
`2` 个设备像素 bleed 以覆盖抗锯齿边缘和避免接缝。图块身份包含文档、
当前精确 zoom、DPR、图块坐标、样式 generation、局部 tile epoch、面积
填充状态和设备像素相位。缩放、DPR、主题/样式、几何或 generation 不匹配的
结果不得发布；离开当前视口的请求会被取消，已经在途的晚到结果会被丢弃。

数字切片中的测量几何使用整张切片的全局坐标。viewport origin 只参与
全局坐标到当前窗口的变换，平移或切换 viewport 不得把测量对象改写为局部
坐标；命中、测量、项目保存及导出仍全部使用 RAW 全局几何。

## 有界缓存与资源预算

生产路径中的显示缓存均有独立上限：

| 缓存 | 上限 | 说明 |
|---|---:|---|
| 面积 RAW 路径与校验后的屏幕代理 | 64MiB | 本帧正在使用的路径临时固定；预算不足时使用未缓存 RAW，避免顺序扫描 LRU 抖动 |
| 测量文字 sprite | 32MiB | 全工作区共享的字节 LRU，不按画布重复分配；缓存完整文字、背景和描边图像 |
| 面积控制点显示集合 | 16MiB | 只减少屏幕上重复控制点的绘制；精确命中仍检查全部 RAW 顶点 |
| 被动叠加图块 | 128MiB 且最多 256 项 | 双条件 LRU，缓存当前精确 zoom/DPR/style/generation 的透明图块或精确命令 |
| 后台图块快照 | 每类共享缓存各 128MiB | 包括已取消但仍在途的任务；超预算拒绝入队，可见画布保留完整预览 |
| 完整文档显示预览 | 64MiB 且最多 8 项 | 全工作区共享，最长边最多 1536 像素，显示用途 |
| 草稿与新对象预览 | 几何估算 16MiB + 栅格 48MiB | 主体、已确认剔除、当前识别各自版本化；同一预算也承接确认前后画面 |
| 后台 RAW 坐标快照 | 32MiB | 不可变坐标字节的共享 LRU，不改变持久几何 |
| 活动对象与编辑背景 | 32MiB | 稳定高亮和局部背景的共享栅格缓存 |
| 独立进程 RAW 路径 | 64MiB | 后台路径复用；不占 GUI 线程的构形时间 |

后台 pending 预算与完成图块缓存是两个独立预算。`pending` 表示仍登记在当前
可见请求表中的请求数量；底层 `CanvasOverlayCacheStats.pending_bytes`
表示所有尚未回收的在途快照估算字节数，其中也包括已经取消但 worker 尚未
返回的任务。后者不会因为取消信号发出就提前释放预算，避免快速平移/缩放时
短暂突破该类缓存的 128MiB 上限。精确图块、完整预览和草稿栅格各有一份共享
pending 预算，不应把三者合计误报为 128MiB；这些是上限，不是预分配，也不
包含原始图片和进程运行时。基准 JSON 的 `overlay_cache.tiles.pending_bytes` 及
等待/空闲阶段同名字段均采用这个估算口径，而不是已完成图块的常驻 `bytes`。

发布初期保留两个独立回退开关：

```bash
FDM_DISABLE_AREA_SCREEN_PROXY=1
FDM_DISABLE_CANVAS_OVERLAY_CACHE=1
```

前者只关闭面积屏幕代理，仍保留 RAW 直接渲染；后者关闭被动叠加图块，仍使用
空间查询、文字 sprite 和 RAW/经校验代理的直接绘制。两个开关都不得改变项目
JSON、测量数值、命中结果或导出轮廓。

使用 `--list-scenarios` 查看完整场景列表。当前列表还包括 500/1,000
个长度对象的标签开关组合、100/300/500 个含孔洞面积，以及
200,000/600,000 个面积坐标。

每次结果包含：

- 冷帧和热帧的 P50、P95、最大值、均值与原始样本，单位为毫秒；
- 平移、缩放、选择、面积落点和拖动各自的动作时间、后续同步帧时间及
  `paintEvent` 增量；非面积场景会明确将面积落点标为不适用；
- 默认 500ms 空闲观察窗口内的额外 `paintEvent` 数、缓存活动及剩余后台
  请求，可用 `--idle-ms` 调整（正式性能采样应保持默认 500ms）。空闲采样前
  会设置 `WA_DontShowOnScreen`、把画布置为 Qt visible，并排空预期的 show、
  选择、代理预热和图块工作，随后才开始计数；隐藏 QWidget 的 `update()` 不再
  被误判为“没有持续重绘”。只有 settle 成功的结果才标记 `valid=true`，
  `quiescent=true` 还要求 paint、缓存活动和所有 producer/pending 均为零；
- 对象数、RAW 坐标数、标签状态、固定 seed、图片和画布尺寸；
- 当前选择的 `document` 或 `digital_slide` 画布类型；
- 请求与实际生效的渲染路径；
- 图块等待时长、是否超时、图块数、缓存字节数、冷热阶段命中/未命中率；
- 完成、待处理、generation 晚到丢弃和其他防御性丢弃计数；
- 面积路径、文字 sprite、面积控制点和图块缓存在冷帧前、冷帧后、热帧后及
  交互后的条目数与字节数；
- 一个不计入耗时样本的 UI 线程 trace 帧中的显式
  `drawPath/drawImage/drawPixmap/QPicture.play` 调用数；
- 可取得时的当前/峰值 RSS；
- Python、FDM、PySide、Qt、平台、DPR、CPU 与 Git revision。

生产缓存仍保留一个聚合 `dropped` 计数；基准在三个完成发布入口观察本次 run，
只有完成 key 已不属于当前 canvas generation/viewport 时，才增加
`generation_late_drop_count`。空图、超预算、当前 generation 的取消等进入
`other_defensive_drop_count`，二者之和等于 `defensive_drop_count`。
schema-v1 兼容字段 `late_drop_count` 和活动中的 `late_or_rejected` 继续表示
聚合防御性计数，新的 generation 字段才用于晚到结果门禁。pending 快照在
入队前因 128MiB 预算被拒绝时不会占用缓存，也不会必然增加丢弃计数；应结合
`pending_bytes`、完整预览和目标图块就绪状态判断后台压力。任何丢弃都不能
解释为测量丢失，当前文档的完整预览负责过渡显示，精确快照负责最终替换。

可用 `--objects`、`--coordinates`、`--frames`、`--warmup`、`--width`
和 `--height` 调整规模。场景几何由固定公式生成，不读取项目文件，也不写入
项目或撤销数据。

`render_workload` 提供 direct、cached 和交互混合帧的明确分项。所有冷帧、
热帧和交互耗时都使用原始 `QPainter`，不安装计数包装。耗时采集结束后另渲染
一个不计时 trace 帧，临时包装 `QPainter.drawPath()/drawImage()/drawPixmap()`
及 `QPicture.play()`，且只统计 canvas 所在线程的显式调用；后台 worker
不会混入。这些调用数用于识别
“每对象多次矢量绘制”或“热图块未生效”等结构性回归，不用于跨机器耗时比较。
QPicture 内部的 C++ 命令回放不会逐条经过 Python 包装，因此图块命中场景应
同时结合缓存命中率、图块数量和 `drawImage` 次数判断。

每次 run 开始前都会清空并核对面积路径、文字 sprite、控制点与图块四类
进程级缓存；已取消但尚未返回的图块快照会在有界等待内排空。run 结束后再次
释放这些缓存，连续执行的第二次“冷帧”不会复用第一次的文字或路径。可见长度
和面积场景还会确定性地生成不同的格式化结果值，避免 500/1,000 个对象实际只
命中一个文字 sprite。

如需保存结果，`--output run.json` 会写入已忽略的
`.tmp/canvas-benchmark/run.json`。绝对路径也必须位于仓库的 `.tmp/`
目录中，避免把机器相关的性能结果加入 Git。输出名会先补 `.json` 后再执行
解析路径和 containment 校验，因此把仓库 `.tmp` 目录本身作为文件名、符号
链接逃逸或任何仓库外路径都会被拒绝。

标准测试只使用极小对象集验证场景、JSON schema、离屏运行链路及图块等待，
不对耗时设置绝对断言，也不会默认对完整场景启用图块缓存。
开发机和非固定 CI 的结果用于同机前后对比、缓存行为与持续重绘诊断，不设跨机
绝对时间门禁。只有固定的 Windows 低配性能机才读取 `timing_ms.cold`、
`timing_ms.hot`、交互延迟和 `rss` 执行绝对时间门禁；macOS 或普通 CI 的
绝对毫秒值不得替代该验收。
