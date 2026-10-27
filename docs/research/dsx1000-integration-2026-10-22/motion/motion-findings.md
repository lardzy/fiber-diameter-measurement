# DSX1000 运动与完成反馈静态分析

分析日期：2026-10-22。仅分析提供的软件副本；未加载程序集执行、未启动原厂程序、未连接设备、未修改驱动/配置/FDM 源码。

## 结论

**XY/Z 均存在原厂使用的设备命令应答屏障，可以形成比固定等待更有依据的控制链。** 普通逐层 Z 扫描在软件中确实等 Z 移动请求返回，再放行下一次相机触发；这不是仅凭方法名称作出的判断。

但离线证据没有证明设备应答一定代表物理到位且光学稳定，也没有证明任何公开 RDK 成功回复足以作为取图条件。发现三项必须纳入接入设计的限制：

1. Z 的 `Moved` 来自位置通知，不能作为“运动结束”事件使用。
2. RDK 对 Z 到达限位、STOP 等情况可能按正常行为处理；`CommandWaitRelease` 在 finally 执行。因此必须结合最终状态、实际位置容差、取消状态和新帧身份判断成功。
3. RDK 的位置接口经过 UI 坐标与整数微米量化，且位置查询读取 UI 缓存。不能将内部 XY 312.5 nm/pulse、Z 10 nm/pulse 的转换粒度宣传为 RDK 精度或设备定位精度。需要小于 1 µm 的 Z 层间距时，应优先验证原厂批量采集配方/内部 nm 路径。

## 证据范围与身份

所有程序集来自：

`/Volumes/LARD/2.中转/OLYMPUS/OLYMPUS/DSX-BSW/DSX-BSW-CTRL/`

以下 token 只对对应 SHA-256 与 MVID 有意义。MVID、完整签名、CIL 方法体、异常区域、原始方法体哈希见本目录 JSON；完整 REA 记录另存 `rea-evidence.json.gz`。

| 简称 | 文件 | SHA-256 | REA Evidence |
|---|---|---|---|
| ST | OlydStage.dll | `2b81ecb535e5e9f8b96b2dfdf7e871a2a9310ffbf3d59e36fdb8300130ff6194` | `ev_8a1d2a1a382d16fb79f914c50837e789cf8fa654e8ed058ee30020fa68e67ebe` |
| SD | OlybStageDomain.dll | `2c8091cb74dd3b20ff14363d8810b739c0d972008ec6fc7b7560d659c150e0ff` | `ev_f358e75a4fb3a68e478952f023c231ddf2bebe9ebd8428237e2a7d4906ae8f4c` |
| ZD | OlydMicroscope.dll | `f3d42e588a5ea19570db681ec1886b05da8e7a69d858ff04d6698d58a654d810` | `ev_cd772ef6ffceef23ce1f6c17757bac986e7fd8d56df2539e0c994aac8a9e1b0a` |
| MD | OlybMicroscopeDomain.dll | `7f443e9916c0f1a5230a0093f73a68dddd546ee369e72779619a9b7042df2f9e` | `ev_3a11f4f672b22c536ff4a0b883514fbf5e6f5de8288ae07abde4955e5183cc74` |
| DEV | OlydDevice.dll | `5e1da5ce793f71258b95fd06f3960e1376841403acfe5e7b6f6753fde8a7183d` | `ev_c66f21c1077a53c53582e76d6e5227554440c9c3e2d311a0e8d689d59491cb0b` |
| PORT | OlydPort.dll | `5f1c56f28a6f5c6a0ca5a0dc1918133bc4404b13fd5aa186fa909ed9e9ce804b` | `ev_d193d31052eeead1ce4148d5861e465877578ddc558080699b2be969c4770040` |

补充证据：

- DB：OlybMemoryDatabase.dll，`ev_fc5f7d5ef1644e9899021e77c833b1615a828d67daf31b9c904498e0b9d5286d`；同步事件链见 `event-synchronization.*`。
- UI：OlyuMicroscope.dll，SHA `f35ce70de806aa1103e5e6c8b8d672d3ba1de2576f07beb1ed53e1f6141eb067`，Focus/Stage 选择证据 `ev_8a7b1da07c5ca25204fabb45c2361af6782e97fd30d6c3db2a409eb5fd86dd94`。
- DM：OlyuDomainManager.dll，SHA `c89dd1dfa5903c26620d4b588d33f707e391401bfbe0798f68fac21892578a00`，`ev_39d30b0eff53f7d043487746bef513de6c267af445646e80a6f5de31b297ba7c`。
- DC：OlyuDomainController.dll，Focus/Stage `ev_56533c7dcccb747a2f1b084724aaacd10ee1d5dfab1f8430b29b947475aaa542`，基类 `ev_3d05d5ee1dc51fdeb1c788d1de1e8bbe252d3a87e2c72888509acbce51f029f5`。
- CALC：OlyuUiCalculator.dll，`ev_4566a595099128c86f22718a13c0b5abcd7f38afca9a976715f034e16e099a8c`。
- UNIT：OlyuUiUtility.dll，SHA `644e4384819490a05d652eea49813b315f9e585df3f076faf4bd98129ef0bb8c`，`ev_855f51c173d088e2d3819fc4f1397ad027a52731108f639331e3ebd35228f1e1`。
- UI 状态：`ev_378406d3cc5d33fff9bac20201e3e02f62d6c8d92469fb1ab8c86b8b422a8ea1`。只用于说明 SHOT 开始/结束状态及异常释放；不冒充移动硬件反馈。

## XY/Z 能力清单

“已证实”指副本中代码路径存在且已检查 CIL；不代表实机验收。

| 能力 | 静态结果、精确入口 | 剩余验证 |
|---|---|---|
| XY 绝对移动 | ST `SsuStage.SubMoveAbsolutely 0x0600007a` → `SubAsyncMoveAbsolutely 0x0600007b` → `XYG` → 等待对应应答 | 运动应答与物理到位的时序；设备配置匹配 |
| XY 相对移动 | ST `MoveRelatively 0x06000059`，`XYM` | 方向、旋转/对齐下坐标转换 |
| XY 位置读取 | ST `GetCoodinates 0x0600005f` → `XYP?`，检查恰有 2 参数，解析设备脉冲后转换 | 设备回读值是否编码器实测，RDK 缓存刷新时点 |
| XY 状态 | SD `MoveAbsolutely 0x06000062` 设 STAGE_MOVING，设备调用返回/异常后 finally 设 STAGE_IDLE | 此状态是软件状态，不是已证明的独立硬件 busy 位 |
| XY 位置通知 | ST `EnableNotifyXYPosition 0x0600006f`，`CallbackReceivedNotifyCommand 0x06000071` 处理 NXYP | 通知周期、丢失/重连、最终位置新鲜度 |
| XY 停止 | ST `DecelerationStop 0x0600005b` → `XYSTP` 请求 | 停止时间、终止任务与后续接管 |
| Z 绝对移动 | ZD `MoveAbsolutely 0x0600007e` → `FG`、同步请求；`AsyncMoveAbsolutely 0x0600007f` 返回 handle | 位置/曝光同步及允许范围 |
| Z 相对移动 | ZD `MoveRelatively 0x0600007c` → `FM`、速度、F/N 方向、脉冲长度，等待应答 | 层间距对应真实焦面变化、反向间隙 |
| Z 位置读取 | ZD `GetZPosition 0x06000084` → `FP?`，解析设备返回位置 | 位置反馈性质、取图时坐标关联 |
| Z 位置通知 | ZD `CallbackReceivedNotifyCommand 0x06000089` 收到 `NFP` → pulse→nm → 触发 `Moved` | `Moved` 只证明位置通知，不能当运动完成 |
| Z 状态/停止 | MD `MoveRelatively 0x0600015d` 设 Z_MOVING/IDLE；ZD `StopZ 0x06000080`、`StopZAsync 0x06000081` → `FSTP`；MD `WaitMoveZStopResponseTask 0x06000150` 等停止应答 | 中断期间的命令配对、实际停住、停止后的可恢复性 |
| 限位/错误 | ZD SetNearLimitPoint/SetFarLimitPoint、MD `IsErrorForZHardwareException 0x06000162`；SD 范围检查与硬件异常分类 | 不改当前限位；必须验证限位/取消并不被误记为到达目标 |

## 请求已发送与请求完成有明确区别

核心已检查链路：

```text
业务运动调用
  → AHardwareWithPort.AsyncSendCommand [DEV 0x06000022]
    → 仅排入硬件发送队列、返回 CommandHandle
  → AHardwareWithPort.SendThreadFunc [DEV 0x06000026]
    → IPort.Send(command, timeout)
      → ASerialPortBase.SendThreadFunction [PORT 0x0600007b]
        → SerialPort.WriteLine
  → AHardwareWithPort.AsyncReceiveCommand [DEV 0x06000023]
    → APortBase.Receive [PORT 0x06000007]
      → ASendCommand.WaitForResponse [PORT 0x06000029]
        → EventWaitHandle.WaitOne(timeout)
串口收到响应
  → ASerialPortBase.SerialDataReceivedEvent [PORT 0x0600007c]
    → 按命令键寻找 pending command，写入响应参数
    → ASendCommand.SetResponse [PORT 0x06000028] → EventWaitHandle.Set
等待返回
  → AsyncReceiveRequestCommand [DEV 0x06000024]
    → 验证参数数目、ParamToResult、超时/错误分类
  → AStageHardware / AMicroscopeHardware 对非零结果抛硬件异常
```

因此“排队成功”“WriteLine 成功”“等到相应返回”是三个不同阶段。这里的 timeout 是响应故障上限，不是为每张图固定睡眠：例如 Z 同步请求使用 30000，XY XYG 的异步请求使用 4500000；单位按 WaitOne 毫秒参数使用。不可照搬为 FDM 的固定等待时间。

PORT 接收路径另有通知表，并通过异步通知回调派发；它与请求响应 SetResponse 路径分离。当前解析显示 pending 请求按命令名匹配，没有证明支持任意同名命令并发；桥应先串行化采集和运动任务。

**物理语义仍需实机证明：**原厂将运动命令的应答用作完成屏障，但副本不足以判断控制器是在真正停止、进入位置容差、还是完成控制序列时发出应答，也不能证明余振消失。现场必须把应答、位置回读、曝光、新帧时间戳放到同一条时间线。

## 普通逐层 Z 扫描的同步屏障

MD `Microscope3DAcquisitionService.StartZMovePitchTask 0x06000081` 将 `_MoveZPitch 0x0600008a` 注册到 AcquisitionEventDao.EntityUpdateHandler。停止任务 `0x06000082` 解除订阅。

`_MoveZPitch` 检查事件值为 3，读取 AcquisitionCondition.Zpitch，再同步调用：

```text
DeviceController.Microscope.ZDrive.MoveRelatively(speed=3, direction=0, pitch)
  → FM 请求 → WaitForResponse → 处理返回
```

DB 同步链是实质证据：

- `AcquisitionEventDao.UpdateEvent 0x06000ab0`：设置事件值后调用 DaoBase.Update。
- `DaoBase.Update 0x06000028`：IL_0042 调 OnEntityUpdateHandler。
- `OnEntityUpdateHandler 0x0600003f`：IL_000d 调 delegate.Invoke；没有 BeginInvoke/线程排队。

相机代理独立恢复的 `SnapService._ReceiveResumeRepeativeSnapEvent` 在 UpdateEvent(3) 返回后才 Set 相机循环等待句柄，循环随后才允许下一 SiaCapture。结合以上链路，**普通步进 Z 栈有软件级“当前帧处理→Z 移动应答→下一帧触发”的同步顺序**。相机部分独立证据 ID 为 `ev_834f41d1d8bf823d3b53aa85461b76af8c1dbbe939c362670a25b7cdfa0611a9`，本报告只对本代理检查的运动/DB 部分负责。

SuperFast3D 另有绝对运动等待任务（MD `0x06000089`），但不能把其存在直接解释成曝光与连续运动已准确同步。连续模式应保持候选状态。

## RDK → 业务层 → 运动层

入口代理负责 RDK 请求解析，本目录证实以下下游：

- XY：UI `Stage.StageMoveAbsolutelyCheckRangeWithoutAngle 0x06001312` → `_Exec... 0x06001340` 设置 TargetPosX/Y → DM StageBehavior `0x0600018a` → 场景派发 → DC StageController `0x0600024b`。该方法构造范围检查、速度、绝对运动、位置读取、恢复速度等任务，场景号 721。
- XY 任务 SD `MoveAbsolutelyWithoutAngleTask 0x06000045` → `MoveAbsolutely 0x06000062`；IL_0045 调 IStage.MoveAbsolutely，finally IL_004c–0067 设置 IDLE 并 CommandWaitRelease(21)。
- Z：UI `Focus.ExecuteZLengthMove 0x06005180` → DM FocusBehavior `0x06000249` → DC FocusController `StartZLengthMove 0x06000108`，设置 TSK_MIC_Z_MOVE_BY_SPECIFIEDLENGTH，场景号 807。
- Z 任务 MD `MoveZBySpecifiedLength 0x06000149` → `MoveRelatively 0x0600015d`；IL_0040 调设备 ZDrive.MoveRelatively，finally IL_00d2–0101 设置 IDLE 并 Release(8)、Release(7)。

**RDK 运动最终应答有实际等待链，但释放并不保证目标到达。** XY Release 发生在业务 MoveAbsolutely 的 finally 内，后续场景位置读取尚可能未完成；Z 对限位与 STOP 有按正常行为处理的分支。应在协议回复成功后再确认对应任务未取消、位置新鲜且满足容差，之后才认可运动完成。

RDK 的 SHOT 与运动语义不同：DM `AcquisitionBehavior.StartStandardModeSnap 0x06000072` / `StartFullModeSnap 0x06000073` 调 `ControlBehavior._ExecuteScenarioTable 0x06000003`，后者调用 `ScenarioController.ReleaseEvent 0x06000016`（仅 EventWaitHandle.Set）后返回，另一个 RunScenarioTable 线程再执行采集。结合入口代理找到的 SHOT Release(14) 顺序，可证明 SHOT 最终回复是启动派发后的回复，不能当图像已采集或落盘完成。

## 单位、显示坐标与精度

- ST `ChangePulseToNano 0x06000076`：pulse × 312.5 后按原厂规则舍入；`ChangeAbsoluteNanoPosToPulse 0x06000077` 除以 312.5，并有 µm 舍入匹配修正。因此要保留原厂转换与软/硬坐标关系，不能简单用 `µm×常数` 替换已有电机步数。
- ST `ConvertAbsoluteCoordinatesFromSoftToHard 0x0600006a/6b/6c`、`...HardToSoft 0x0600006d` 包含原点/对齐/旋转变换。RDK “without angle” 路径仅说明选用该任务参数，不证明用户坐标就是设备原始坐标。
- ZD `ChangePulseToNano 0x0600008a`：pulse × 10；绝对/相对 nm→pulse `0x0600008b/8c` 除以 10，再经过 `0x0600008d` 舍入。是该版本的软件量纲，不是测量精度保证。
- CALC `FocusCalculator.ConvertZPosForDisp 0x06000003`：`display_nm = ZMoveRange/2 - raw_nm`，UI Z 符号/原点经过转换。
- CALC `ConvertZPosToUMForDisp 0x06000002`：上述结果 ×0.001，再 conv.i4；UI `ExecuteZLengthMove` 的参数签名就是 i4。因此该路径在计算相对移动之前使用整数 µm 的显示基准。
- UI `Stage.GetDisplayStagePotionForRDK 0x060012fd` 从 UiResultData.StageData.MovePositionXY 缓存取坐标，转换 nm→µm，再按 0 小数位、AwayFromZero 舍入，最终格式化 X,Y。UNIT `0x060000f7/0x06000114` 的 switch 目标已经从经过 SHA 校验的原始 CIL 字节补核，见 `unit-switch-targets.json`。
- RDK Z 查询也由入口代理确认读取 UI 当前属性，未证明每次查询发起 FP?。现场必须测量缓存时延与更新边界。

## 现场针对性验证

1. 先观察原厂一次短 XY 移动与一次单 Z 移动，记录队列、发送、设备应答、位置通知、原厂状态和相机曝光/帧完成。核对原厂日志中“stopped successfully”与真实到位的时间关系。
2. 用小步长重复定位，比较请求坐标、设备 FP?/XYP? 原始回读、UI/RDK 回读；明确 RDK 的整数 µm 量化及缓存刷新。记录对齐、旋转、ZMoveRange 和方向。
3. 特意验证被限位截断或取消的 Z 移动不会被桥记成成功；RDK 的正回复不足以通过该项。退出/失联/超时同理。
4. 普通单位置五层扫描确认事件3发生在上一帧处理之后、下一帧触发之前，每一层都有唯一实际 Z 与新帧身份。先不采用连续 SuperFast 路线。
5. 若事件与实际位置都正确，单独测最短必要稳定时间；不把原本全部固定等待直接删为零。再做 3×3×5 的 45 张图验证，并评估运动、稳定、曝光、处理、写盘各阶段耗时。

## 文件索引与覆盖限制

- `OlydStage/OlydMicroscope/OlydDevice/OlydPort/OlybStageDomain/OlybMicroscopeDomain.selected.json`：六个核心程序集精选完整方法体。
- 同名 `.resolved-il.txt`：人工阅读版，保留 IL 偏移与 token，引用解析到类型/成员名。
- `event-synchronization.*`：Dao 更新同步 Invoke 的关键证据。
- `rdk-control-path.*`：UI/DomainManager/DomainController 到运动与 RDK 释放的连接、SHOT 异步派发对照。
- `units-and-rounding.*`、`unit-switch-targets.json`：显示坐标变换、单位与舍入。
- `OlyuMicroscope.ui-state-selected.json`、`OlyuMicroscope.shot-release-selected.json`：提供入口代理交叉验证的状态/SHOT 分支。
- `rea-evidence.json.gz` 与 `rea-evidence-index.json`：本代理独立 REA 会话的完整保留记录与可检索索引。

核心选中程序集/类型元数据与 CIL 解析为 complete、无解析问题。全 OlyuMicroscope 扫描超过 REA 64 MiB 表示预算，随后改为精确类型选择并成功；不能据此宣称已经穷尽整个 UI 程序。MCP 大结果超过传输预算时，分析完整记录仍保留，使用分页消费并导出。本报告未进行原生驱动/控制器固件分析，不证明机械准确度、重复性、光学稳定、设备占用交接或实机离线部署。

