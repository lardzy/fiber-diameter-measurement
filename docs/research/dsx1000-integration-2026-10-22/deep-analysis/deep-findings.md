# DSX1000 第二轮静态分析：保存闭环、坐标鲜度与现场可观测信息

分析日期：2026-10-22。对象为供应的软件副本；未启动原厂程序、连接设备、修改配置或执行日志备份脚本。这里的“已证实”仅指该版本文件中的控制流，现场仍须核对文件身份和行为。前置材料见 [第一轮接口分析](../interfaces.md)。

## 结论与对收集工具的要求

本轮将“APP 保存结果返回”和“UI 位置异步刷新”进一步落实到调用关系。推荐路线仍为同机 RDK 桥接、普通二维单帧逐层采集；数据收集工具暂不自动判定运动到位或采集成功。

| 问题 | 本轮已证实 | 工具应如何处理 |
|---|---|---|
| `SHOT` 回复后的保存链 | 非拼接 APP 图像转交同步执行自动保存分支、等待工作项并返回 Result；CTRL 收到结果后更新转交结果 DAO | 记录 SHOT 回复、SHOTSTS、文件族与日志；回复不等于新图完成 |
| `SHOTSTS=0` | 转交结果处理与停止处理都可置 0；协议不携带 Result、拍照 ID 或实际路径 | 显示“原厂报告状态 0”，不显示“保存成功” |
| 坐标鲜度 | Focus 的 UI 数据更新明确经过 `BeginInvoke`；ZPOS 读更晚更新的字段；XY 也读取缓存 | 目标值、报告值、人工值分开；相等轮询不作为新鲜度证明 |
| 错误查询 | 内存 ErrorCode 粘滞；每次 final reply 会消费共享错误文件；不是 GETERRCODE 专属清零机制 | 保留全时间线；历史错误不强归属于当前拍照，默认观察不查询 |
| 原厂快/精细设置 | 切换 ExtendMode 可重算 pitch；UI pitch 保留三位小数，RDK 移动路径仍有整数 µm 量化 | 记录切换模式后的实际参数，保留小数原值，不静默取整 |
| 日志可用性 | 副本配置 `LOG_FORCE_FLUSH=0`、Debug/Step 类别关闭 | 日志到达时间与事件时间分列；没看到日志不代表动作没发生 |

## 1. APP 自动保存、命名与完成边界

### 1.1 普通非拼接转交顺序

以下调用链来自完整 CIL，所有箭头为调用顺序，而非靠方法名称推测：

```text
CTRL ProcessJoinService.TransferImageDataTask [T1: 0x0600009b]
  → MicroscopeClient.ImageTrancefer → ApplicationServiceClient.ImageTransfer [T2]
  → APP ApplicationServer.ImageTransfer [T3: 0x06000077]
  → 同步 dgtImageTransfer.Invoke
  → OLYCoordinator._ImageTrancefered [T3: 0x06000033]
  → 非拼接分支 PerformAutoSaveInfoOperation [T3: 0x0600003c]
  → ExportImage / 数据库存储 / 临时图像分支
  → WaitOnAllWorkItems → 返回 Result(IsSuccess, ErrorCode)
  → CTRL 写 TransferImageResultInfoDao 并 Update
  → UI TransferImageResultInfo [T4: 0x060048f4]
  → 必要时恢复临时自动保存开关 → UI 结果处理 → AcquisitionShotStatus=0
```

关键偏移：APP `_ImageTrancefered` IL_03ef 调自动保存，IL_04e4 等工作项，IL_0541 返回；CTRL IL_00b7 等同步 WCF 返回，IL_01d5/01e2 写成功与错误字段，IL_01ee 更新 DAO。UI 状态 0 写在 IL_00c1。[`image-transferred.txt`](image-transferred.txt)、[`ctrl-transfer.txt`](ctrl-transfer.txt)、[`app-wcf.txt`](app-wcf.txt)、[`wcf-client.txt`](wcf-client.txt)、[`ui-transfer.txt`](ui-transfer.txt)

这收窄了第一轮“保存与状态关系未闭合”的缺口：**正常非拼接转交结果路径，UI 的结果处理发生在 APP 自动保存处理返回之后**。但它仍不能成为无条件外部成功屏障：关闭自动保存时不一定有最终用户文件；取消也会置 0；保存失败仍会进入结果处理；协议缺少此次拍照身份及 Result 载荷。未证明所有选件、错误弹窗和图像格式的外部完成行为都一致。

### 1.2 文件名与格式不是 SHOT 参数的直接拷贝

`PerformAutoSaveInfoOperation` 先经数据库列表、图像列表 `GetUniqueFileName` 再创建实体；保存目录由 AutoSaveInfo 决定。`ExportImage` 遍历所选格式，因此一次拍照可能有多个文件。`GetFileNameBaseOnFileType` `0x06000047` 对部分格式保留基名，对另一些格式添加 `_C`，高度图还可能有 `_H`。

`GetUniqueName` `0x0600003e` 检测目标存在后追加或递增 `_数字`；其这次调用用于特定临时图像分支，**不能将它推广为所有最终格式的命名规则**。图像列表的另一套 `GetUniqueFileName` 检查列表名称；文件系统、内存列表与数据库碰撞行为不完全等价。[`autosave.cil.readable.txt`](autosave.cil.readable.txt)、[`imagelist-selected.txt`](imagelist-selected.txt)

工具应提供独立实验目录与不可复用的拍照基名，记录调用前文件基线，并展示此后所有新增/变化文件；不猜扩展名，不把 `_C`/`_H` 当作额外焦层，不自动将“第一个出现的文件”绑定为最终图。实际保存格式在原厂 UI 中核对。

### 1.3 保存器返回与文件关闭

- `OlycImageList.DataListModel.ExportImage` 有直接 `IImageFileStoreFacade.SaveImage` 路线；`OlysDataStoreService` 再调用 `LoadSaveModule.ImageExportContext` 的策略。LoadSave 的 BMP/JPEG/WMP 等托管编码路径包含流 Close/Dispose；失败返回 -1 的部分入口会删除不完整输出。[N1/N2/N3]
- `OlyImageExporter.MainWindow.ExportImage` `0x06000008` 中编码器 Save 后 finally IL_00a0 Close，外层 IL_00ae Dispose，才返回；它不是“进程启动就表示文件完成”。`DataListModel.LaunchProcess` `0x06000240` 会读取 stderr、循环 `WaitForExit(100)` 并检查退出码；stderr/错误路径不同于正常返回。[N1/N4]
- 原厂 DSX 路线 `LoadSaveModule.DSXImage.ExportImage` `0x06000043` 调 `FileFormatWrapper.FileWrapper.SaveDSXImageToDisk`，catch 后返回 -1。对应 C++/CLI/原生存储实现的文件关闭、缓存持久化和各编码格式细节，本轮没有证实。不能把托管 BMP/PNG 的 finally 泛化到 DSX/TIFFExt。
- 文件 Close 也不是断电级持久化保证；文件大小稳定和复制哈希通过只说明收集器观察到的字节状态。收集程序不需要依赖原生句柄猜测即可准备现场观测，因此本轮未为“绝对 flush 保证”扩张到全面原生反编译。

## 2. 位置缓存更新的真实偏序

原设备层 `FP?`/`XYP?` 有硬件回复，见 [运动证据](../motion/motion-findings.md)。RDK 查询并不在每次请求时执行这些命令。

本轮补到以下中间层：

```text
Z 业务对象更新
 → FocusController.ReceiveResultData [T5: 0x06000114]
 → 构造 UiData.ZInfo，设置 ZCurrentPos
 → FocusResultData.set_ZInfo [N5: 0x060007fe]
 → 先写 _zInfo，再 UpdateUiData.BeginInvoke（异步）
 → Focus._UpdateZInfo [T6: 0x060051a3]
 → Focus.set_UpdateZCurrentPos [T6: 0x0600515f]
 → 写 _ui32ZCurrentPos
 → RDK ZPOS? 读取 Focus.RawZCurrentPos 并转换显示坐标
```

XY `StageController._SetResultPositionXY` `0x06000265` → `StageResultData.set_MovePositionXY` `0x06000abc` 先更新结果缓存，再 BeginInvoke 通知 UI；`STGPOS?` 直接读该结果缓存，不等 UI 文本刷新。ZPOS 则多经过 Focus 对象字段更新，二者不能使用同一假定延时。[`domain-position.txt`](domain-position.txt)、[`uidata-position.txt`](uidata-position.txt)、[`focus-update.txt`](focus-update.txt)、[`position-cache.txt`](position-cache.txt)

第一轮已证运动任务在 finally 释放 RDK 等待，XY 后续场景才有位置读取任务。因此：

- 回复到来、UI 数据写入、UI 字段刷新之间不存在本轮已证的外部统一屏障。
- 一串相同坐标可能是缓存尚未更新，也可能是真正静止；没有新鲜度标识不能区分。
- Z 显示换算仍为 `ZMoveRange/2 - raw`，不是原始设备方向；软件脉冲换算不证明编码器精度。
- 明天应同时记录运动请求/回复、位置日志、报告坐标与人工原厂读数。若日志不包含足够位置身份，先保留“新鲜度未证”，不以增加轮询次数包装成已验证反馈。

## 3. GETERRCODE、异步保存错误与取消

### 3.1 两种错误状态

`CRdkDataManager.SetErrorCode` `0x06000056` 同时写 `ErrorCode` 和 `CurrentCommandActionError`。每条命令处理器入口调用 `InitializeCurrentCommandActionError` `0x06000057`，它只清后者，**不会清历史 ErrorCode**。`ProcGETERRCODE_REQ` `0x06000018` 先读取历史 ErrorCode 并转成字符串，然后调用 `_SendFinalReply`。[T7/T8；`rdk-error-state.cil.readable.txt`](rdk-error-state.cil.readable.txt)

### 3.2 被消费的是共享错误文件

APP 保存错误可写 `C:\Users\Public\OLYMPUS\DSX-BSW\Common\System\RdkErrorCode.ini` 的 `[COMMON] RDK_ERROR_CODE`。`_SendFinalReply` **每次**调用 `ReadRdkErrorCode`；后者调用 `ReadInfo(..., true)`，在读到值后通过 `File.Delete` 删除共享文件。访问使用 `CRdkSharedInfo_Mutex`。这不是只读 GETERRCODE 的私有动作；SHOTSTS、位置查询等最终回复也可能消费共享异步错误。[T9；`shared-error-full.txt`](shared-error-full.txt)

由此存在两个具体歧义：保存错误可能在 SHOT 的派发回复之后出现，被后续另一条查询首先读到；GETERRCODE 的文本参数又是在 `_SendFinalReply` 读取共享错误之前获取，最终可变成错误应答而非预期数字。不存在每次拍照专属错误队列或关联 ID 的证据。

**现场工具：**GETERRCODE 按钮标记为“历史/共享错误”；保存原始请求、原始回复及时间，不自动清零，不把前次错误归为当前请求，不把 0 作为成功。默认观察不进行任何协议查询。只复制/读取指定日志，不调用原厂的读取后删除函数，也不主动碰错误文件。

### 3.3 取消的边界仍然存在

`ENDSHOT=0` 调停止流程然后回复，未等待停止完成；`StateAcquisitionStop` `0x060049c6` 也写 SHOTSTS=0。没有通用 XY/Z STOP 的 RDK 命令证据。接收处理器同步执行，在长命令等待时同连接取消可能排队；第二连接广播与共享状态不能当可靠旁路。断开连接不代表停止，重新连接不代表坐标可用。工具超时后保持“未知”，禁止自动重发相对移动。[T7/T4、第一轮入口证据]

## 4. 原厂启动、快/精细参数与日志准备

### 4.1 启用条件

继续沿用第一轮已证启动链：APP 正常登录/上下文 → 原厂协调器创建 WCF 和系统上下文 → 启动带 `_dsxApplicationRdkOnlineMode` 参数的 CTRL → 首个 RDK 客户端连接参与初始化 → APP 状态握手。端口配置 35001，地址来自本机 DNS 解析，不保证 loopback。单独运行 CTRL、绕过正常 APP 登录或强制终止旧进程，都不属于收集工具功能。

现场先核对实际版本、原厂正常启动方式及 RDK 选件/授权；没有监听或启动方式未知时只收集证据，不提供“修复配置”按钮。Windows 10 1909 同机运行的兼容性仍须用交付的收集工具验证。

### 4.2 模式会影响实际 Z 参数

`Acquisition.set_ExtendMode` `0x06000415` 记录模式后调用 `AcquisitionCalculator.CalculatePitch`，再重算距离。`set_ZPitch` `0x06000433` 对 UI 值保留三位小数，并记录 `[Extend]-[Pitch]`；`set_ZStep` `0x06000435` 记录层数。日志原文误将 StepCount 也标 `um`，工具不能据此给层数附加长度单位。[T10]

新增 `CalculatePitch` `0x06000091` 证明某些模式切换乘除 `FINE_STEP_RATIO` 与 `SUPERFAST_STEP_RATIO`，末尾为 `×1000 → conv.i4 → ÷1000`。`CstAcquisition` 构造器 `0x06000426` 给 Fine=1、Fast=0.5、SuperFast=0.5，预设范围层数小/标准/宽为 10/30/100。这些是算法常数，**不是本机某倍率的正确层距，也不保证默认当前设置**。[N5/N6；`extend-mode.txt`](extend-mode.txt)、[`extend-calculator.txt`](extend-calculator.txt)、[`extend-constants.txt`](extend-constants.txt)

因此现场表单保留“用户输入/原厂显示”和“模式切换后的实际设置”，记录倍率、镜头、上下界、pitch、step、处理模式、曝光和图像格式。业务使用大于 1 µm 不意味着可忽略小数；本工具拒绝静默量化移动目标，正式桥接另行处理 RDK 整数显示坐标限制。

### 4.3 日志只读收集

`Tools/LogTool/Logging.ini` 指向 `C:\Users\Public\OLYMPUS\DSX-BSW\Log\CurrentLog`，LOCAL_TIME=1、LOG_FORCE_FLUSH=0；Debug/Step 关闭，Comm/Error/UI 等开启。因此落盘时间可能晚于实际事件，特定 Debug 字符串缺失也很正常。不要更改配置来“获得更好日志”。

`DSXLogBackup.vbs` 会归档旧日志并删除源日志（行 128），还会删除超过保留时间的 CAB（行 166）；仅分析，没有执行。收集器应复制所选文件，记录原件/副本哈希和读取时间。默认候选目录使用 `%PUBLIC%\OLYMPUS\DSX-BSW\Log\CurrentLog`，配置实际路径优先作为人工核对项。

[log-signatures.json](log-signatures.json) 给出静态已发现的字符串和含义，包括 `Send Iamge Data...` 的原始拼写；仅作为人工定位候选，不作完成状态机。应保留 logger 类型、线程/进程信息、原始时间和收集器单调时钟，明确日志轮转、截断、重复和延迟。

## 5. 证据索引、复现与未证边界

完整文件 SHA-256、MVID 与方法 IL SHA 在 [analysis-source-index.json](analysis-source-index.json)。可读 CIL 每节都有原始 Evidence ID、目标 SHA、token、IL SHA；`print-cil.mjs` 使用既有 REA 解码器并先校验文件及 CIL 哈希，不加载 CLR。临时抽取的完整记录源自下表已有 bundle，重复副本清理后可直接从原 bundle 复现，例如：

```sh
node print-cil.mjs rea-evidence-bundle.json.gz ev_06e2960bc201588960f5c7f1982113f134d7c23bbc512f3e4426592dc381eb40 set_ZInfo set_MovePositionXY
```

脚本中的 REA 解码器路径为本次分析环境的既有安装；它不是 Windows 收集工具依赖。

| 标签 | 原始 Evidence ID | 文件/作用 |
|---|---|---|
| N1 | `ev_32ac28f68b9eca705c1eff3de44286f194716b1f0621657ee873df32ff2a7753` | OlycImageList：命名、直接存储、Exporter 等待 |
| N2 | `ev_08a3dac24f297294df03e83f0a0b6fd2bd1cc09ea7a261fcf4b1a35cbc77ab63` | OlysDataStoreService：保存策略分流 |
| N3 | `ev_c376108d9576ff2ecbe114a65b96dee68457fe56008ee26146a49b171e4ec9ff` | LoadSaveModule：编码/返回/关闭/原生边界 |
| N4 | `ev_9bb4aacf448bfe2a72e8fe7aa6a0bf20410f360391e99d02978be9c26d000c8a` | OlyImageExporter：编码器 Close/Dispose |
| N5 | `ev_06e2960bc201588960f5c7f1982113f134d7c23bbc512f3e4426592dc381eb40` | OlyuUiData：异步缓存通知与常数 |
| N6 | `ev_5cc84ca4ffbb505c2fca022fcd99c6c0be706dd03aa243ea92748b8bfe4d4110` | OlyuUiCalculator 精确 AcquisitionCalculator 类型 |
| T1 | `ev_b28983bf89a6f28ecaeb257f439dfe93b1e867e246a0d32a22aa3ce7e959c88f` | OlybProcessJoinDomain，旧 entrypoints bundle |
| T2 | `ev_9f8dbdde3fd7b44578f750e7303507672484bea2ebf5e11c4fc5052af0befa80` | WCF client，旧 entrypoints bundle |
| T3 | `ev_843208320387c5be1655ded4cca7240e491442b3ec62025009b2df6b3a04cc1a` | APP Coordinator，旧 entrypoints bundle |
| T4 | `ev_378406d3cc5d33fff9bac20201e3e02f62d6c8d92469fb1ab8c86b8b422a8ea1` | UI 状态，旧 motion bundle |
| T5 | `ev_5eae8ebc2f0558be8a834b544dfeb1050714610bd409b804628601e92cc6aefb` | DomainController，旧 motion bundle |
| T6 | `ev_8a7b1da07c5ca25204fabb45c2361af6782e97fd30d6c3db2a409eb5fd86dd94` | UI Focus/Stage，旧 motion bundle |
| T7 | `ev_65fec493efc403e113aea715f07bba33ccb14f9aa7c2e35d543eb784aa47a93b` | RDK handler，旧 entrypoints bundle |
| T8 | `ev_b6476514101ca3e1981909681091f2aa25ebc7839621d0763994fe7d99992c05` | RDK 数据，旧 entrypoints bundle |
| T9 | `ev_829e6bd66270cc328ef7e962fb1f80592e012849ae6f8f870559bbb22bff144e` | 共享错误，旧 entrypoints bundle |
| T10 | `ev_d7dbd43d2f0023c5080a8597d697e5890582641af43fba882388cf806865b7b3` | Acquisition UI，旧 entrypoints bundle |

新增原始证据共 12 条（6 项检查及各自 summary），保存在 [rea-evidence-bundle.json.gz](rea-evidence-bundle.json.gz)，压缩前后身份和往返校验见 [evidence-manifest.json](evidence-manifest.json)。原配置身份见 [config-identities.json](config-identities.json)。

尚未证实：设备实际许可/启动方式；运动应答对应的机械位置容差和余振；缓存与某一请求的可观测身份关联；各格式实际文件集合及像素/标定保真；原生 writer 的全部关闭/持久化行为；断线/进程退出/限位时的可靠停止与原厂接管恢复；Windows 10 1909 实机性能。上述缺口必须留到现场证据，不能以此报告或模拟测试宣称通过。
