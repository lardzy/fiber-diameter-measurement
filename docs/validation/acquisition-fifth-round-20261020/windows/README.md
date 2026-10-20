# 第五轮 Windows 验证（0.4.10）

本报告区分源码 Qt 自动化、实际安装版 UIA、模拟采集性能与真实相机验收。验证环境为 Parallels Windows 11 ARM64；AMD64 Python 和安装版在 Windows 仿真环境运行。未连接 Microview，不能代替离线相机设备验收。

## 最终 4391 候选包与实际安装验证

应用版本保持 **0.4.10**，构建 ID `4391f019bca842c094b7b281c1d927a7`。本机安装包 `dist/internal/acquisition-fifth-round-20261020/FDM-0.4.10-round5-4391f019bca8.exe`，398,839,390 bytes，SHA256 **`7f5bd6f03900dd61586e2ee9ba35f82720fce6eb910da6d40a5b3ceaac1c89a1`**。内部 `--exclude-area-models` 保留模板和 Microview 运行库；4,818 个载荷文件、435 个生产源码与最终 `0c6ad5ff…` 源清单一致。前序 4211、f52e、e731 的本机安装器已于 2026-10-20 按用户要求清理，其构建身份和失败记录保留；详见 [清理清单](../../artifact-cleanup-20261020.json)。

独立目录安装 exit 0。普通未提权用户 `B1EA\lishuyang` 执行实际安装版 revision 5 自检通过：真实核验子进程 PID 5604、OpenCV 4 线程；same/noisy/drift/reference reset/child process/closed 等七个门禁全部通过，确认退出及共享内存清理。完整结果为 `native-final-installed-selfcheck.json`。

**9 个不同实际安装场景通过**，由两个具名 run 聚合，不能称为一次连续 9/9：前五项（空态和四主题）来自 `20261020-081754Z-08100dcb`；其余四项（1366×768、1067×600、1600×900@125%、Excel）来自 `20261020-082258Z-49fe7f9d`。原第一批的 1366 场在相册及还原全部通过后，记录菜单被并行回传命令唤起的 Parallels Terminal 遮挡；失败截图 `native-final-record-menu-failure.png` 和 UIA 树保留。隔离 guest 命令后原 probe 和产品均未修改，四项串行通过。未把它写成产品菜单缺陷，也未更改断言或直接调用内部停靠函数。最终选用的九项全部正常退出，未强制清理；干扰失败本身的清理记录仍保留。

全部相册场景检查 **GetWindowRect 完整外框**、monitor workarea 和最大化还原后的原尺寸。四主题原生 PNG 均为 1066×791，对应 `[8,8,1066,791]`；125% 场为 1326×971，对应 `[11,11,1326,971]`。原生完整截图及操作页位于 `native-final-installed-screenshots/`，包含标题、筛选和窗口边框，未移动窗口用于截图。四主题已逐张视觉检查。125% 是进程级 Qt 缩放，系统 DPI 未改变，仍不代表真实跨显示器切换。

从常驻按钮实际导出 Excel 16.0 报告通过：52 个输入、144 条直径、12 个计数结果、486 个指定保留公式；编号 `0000123`、VBA、12 个工作表和打印区域保持。报告 SHA256 `80a387d8ac80d6335a450983c25a518c5ddca4ad16486daa8bce5014fb77b078`；原模板未改变，只关闭本任务实例。`native-final-excel-verification.json` 附安装 run 的构建来源，原始 Excel 验证 JSON 保留在 guest 归档。

`final-installer-identity.json`、`final-installed-selfcheck.json`、`final-installed-gui-summary.json`、`final-excel-verification.json` 现在是上述 4391 结果的明确别名；旧 f52e 对应文件保留为 `history-f52e-*`。这一安装与界面通过**不覆盖**下述严格候选全程可见性 FAIL，也不代替真实相机验收。

## 核验进程修复后的源码验证（相册外框修复前）

此阶段源码清单为 `final-process-source-manifest.json`，SHA256 `564525e8e25388920fc632505adc63469bba758804c7668f22b95bfd6be6137d`，2,592 个文件；源码 ZIP SHA256 `d6be2e52dadd6bf4feab38d6f97fb004c74bbfb106be329e9e45f9be7ea679a8`。这次包含正式持久核验子进程、提前消费、业务重绘合帧及图库局部重绘；不是下面的内存原型。

- Windows 最终定向 12 模块：**384 passed、50 subtests passed，44.95 秒**。
- 最终原生压力：100 次菜单共 **779/779** 个样本可见、0 次转换；100 组连续移动产生 100 个新批次；200 次真实导航仅启动采集一次。110.792 秒，源码指纹未变化，详见 `final-source-stress-result.json`。
- 原有四主题／分辨率 16 场景矩阵属于初次 4211 源指纹，样式未变化，但不将它标记为对最终进程代码重新执行的矩阵。

最终性能使用相同冻结生产代码重新成对测量，直接运行生产验证脚本；不沿用旧包基线或内存 wrapper。30 秒负载后停止新增测量，保留实时帧、相册及待写入工作继续 3 秒；负载 FPS／P95 不包含恢复尾段。

| 每条测量间隔 | 基线 FPS / P95 帧龄 | 核验 FPS / P95 帧龄 | 幅度门槛 | 严格全程候选可见 |
|---|---|---|---|---|
| 1,000 ms | 30.2969 / 9.2265 ms | 30.3011 / 21.4502 ms | 通过 | 通过，只有一次初始显现 |
| 150 ms | 30.2936 / 17.3632 ms | 30.2579 / 37.8869 ms | 通过：FPS −0.1177%、帧龄 +20.5237 ms | **未通过**：一次保护暂停后恢复 |

150 ms 组在 10.060 秒隐藏、10.513 秒恢复，控制器暂停 451.25 ms，原因 `verification_late`；之后约 22.88 秒持续可见，没有周期性反复闪烁。3 秒尾段的 91 次绘制全可见、无转换。仍保留严格轨迹 FAIL，不把恢复通过替换为全程通过。1 Hz 保存 30/30 张，150 ms 保存 199/199 张，两组关闭后的帧预算均为 0。详细数据、每帧轨迹、源哈希与判据在 `final-performance/`；历史失败没有覆盖。

四组实测 DPR 均为 1.0、Qt 因子 0.5、逻辑与原生客户区均为 768×576，首画前台 PID 属于对应测试进程。短等待预算约 0.85 帧周期（本组 28.37 ms），核验结果发布帧龄上限约一帧周期（33.38 ms），两个时限分别记录，没有放宽像素对齐阈值。最终核验子进程末段 working set 约 61–62 MiB、private 约 171–173 MiB；这是阶段末观测，各进程 peak 不相加冒充同时全应用峰值。

## f52e 重建候选包（相册原生定位未通过）

版本保持 0.4.10；构建 ID `f52ea0bef9024a608556f525e3d28b68`。本地候选为 `dist/internal/acquisition-fifth-round-20261020/FDM-0.4.10-round5-f52ea0bef902.exe`，398,787,662 bytes，SHA256 `7cd7d0de1a80e8be067653cf22d87f91e95f2c29b9dcf5fd5ae5d658948db038`。内部 `--exclude-area-models` 保留模板与 Microview 运行库，4,818 个载荷文件及 435 个生产源码核验通过；完整身份在 `history-f52e-installer-identity.json`。

独立目录实际安装 exit 0。普通、未提权用户 `B1EA\lishuyang` 执行安装版 revision 5 自检通过：真正创建独立核验子进程（PID 1048、OpenCV 4 线程），same/noisy/drift/reference reset/child process/closed 等 7 个严格门禁全部通过，并核验子进程退出、共享内存解除。详见 `history-f52e-installed-selfcheck.json`；没有沿用旧包 revision 4 的结果。

实际安装版 UIA 的 **9 个不同场景全部通过**，每项正常退出，无强制清理；`history-f52e-installed-gui-summary.json` 明确记录聚合来源。7 项来自 `20261020-065151Z-792c3b52`；125% 与 Excel 分别来自单场 `20261020-070443Z-699f07c3` 和 `20261020-070717Z-92891522`。前两次整批遇到 Parallels 255 传输失败，检查确认失败场景未启动产品后才补跑，原始失败记录保留，**不是一轮连续 9/9**。经 Explorer 的普通用户启动、混合默认值、高级三种存图方式、四主题相册、历史复核/固定小窗、记录停靠、常驻 Excel 均通过；不含真实采集设备。

1600×900@125% 是进程级 Qt 缩放、原生 Windows 窗口，系统 DPI 未改变。无设备/标尺的安装版准备态完整可见两类，第三类局部，保存与 Excel 按钮常驻；不能与源码准备就绪四类目标混同。实际截图在 `final-installed-screenshots/`。补充原生 HWND 检查（`frozen-gui/20261020-071125Z-1e323b1c`）发现真实定位缺陷：普通相册原生矩形 `[-5,-50,1066,791]`，工作区 `[0,0,2582,1548]`，标题及左边界越出屏幕。此前 UIA 仅返回客户区，9 项功能通过不能替代完整窗口视觉验收；本包相册窗口定位暂未通过，失败原图和几何保留，等待修复。

最终 Excel 16.0 实际导出通过：52 个输入、144 条直径、12 个计数、486 个保留公式；编号 `0000123`、VBA、12 个工作表及打印区域保持。结果见 `history-f52e-excel-verification.json`，报告 SHA256 `2a28bdd3f8bc71263f9a19f84c404305312d8bede6c06e50123597bb01711147`。只关闭本任务实例，原模板未改变。

## 相册原生边框修复后的源码验证

保留初次原生失败及两次源码修复观察。第一版使用 Qt frameMargins 后标题恢复，但 Windows 隐藏的 resize 边框不在 Qt 的左右/底部 margin 内，GetWindowRect 仍左越界 5 px。第二版读取本窗口原生 outer/client 差值修正边界，但发现最大化还原过渡期间尺寸被扩大。最终源码 `measurement_evidence_gallery.py` SHA256 `fea6e3ba036badfed4efb2e3fccc00cd8c72bad838af0cb5fadc8fd70800d524` 同时保存正常态恢复目标，避免过渡 Move/Resize 覆盖它；没有更改采集核验或图片质量。

Windows 实际原生外框验证通过：DPR 1.0 首显和最大化还原均为 `[8,8,1066,791]`，客户区尺寸保持 1040×720；DPR 1.25 均为 `[11,11,1326,971]`，客户区逻辑尺寸同样保持。旧负坐标/过大 JSON、暂停后恢复与重复恢复均在工作区内。最大化系统边框外溢单独标注，不冒充普通态失败。对应相册、复核、记录三个模块 **88 passed，9.48 秒**。安装版探针增加 GetWindowRect/monitor workarea 的严格普通态断言和最大化还原尺寸断言，不再用 UIA 客户区代替外框。

最终清单由旧 `564525e8…` 和逐次具名修复清单生成，新的 `.fdm_snapshot.json` SHA256 `159be85e4f92e7e369ec17f2a45fd9a37a1d10b4f1c625d2722564082481977f`。同一组核心核验源码保持不变；旧成对性能结果属于旧图库源码指纹，不能表述为新包逐文件相同。新图库另补 30 秒 1 Hz guard＋相册负载：30.2672 FPS、P95 帧龄 20.6422 ms、29/29 存图、关闭预算 0。**严格全程候选可见仍未通过**：初显后在 12.787 秒隐藏、13.133 秒恢复，运行期一次 `verification_late` 暂停约 350.39 ms；随后约 20.27 秒持续可见，未出现周期性回闪。尾段 91 次绘制全可见、无转换。画布绘制 P95 2.04 ms，末尾 512 个核验 ready P95 18.21 ms、GUI wait P95 0.185 ms，不能用这段分位反推出较早暂停那一帧的耗时。完整结果为 `final-gallery-performance-guard-1000ms.json`，SHA256 `017285126fe3ad53cb9baa5bd79d4b6f018e34d904324f876794f98c8e532628`。这次是追加单组而非新的成对基线，保留原严格 FAIL 标签，不将此前 1 Hz 通过当成本组通过。

## e731 候选包的原生还原失败（历史）

构建 `e7310fc5518545569fce47b007e248a0` 使用上面的 `159be85e…` 清单；安装包 SHA256 `2b66ec0b9dba84e8b8fc5e14780f2968ac86ffb92b14afdb72931d9b619763ba`。包体 4,818 文件／435 生产源码和普通用户安装版 revision 5 实际子进程自检均通过。新严格 UIA 第一项空态通过，第二项发现原生最大化→正常还原缺陷后停止后续场景，**本包不作为最终交付**。

首显原生外框 `[8,8,1066,791]` 已在屏内；调用真正的 UIA `WindowPattern.SetWindowVisualState` 最大化／还原后外框变成 `[8,8,2566,1532]`，没有恢复原尺寸。源码 Qt `showMaximized/showNormal` 路径的通过不能替代该系统路径。随后源码用独立 PowerShell UIA 两次往返复现：系统先发 Resize/Move，此时 Qt 状态仍为 normal，临时 `normalGeometry` 也已变大，而原生 `IsZoomed` 已为 true；即时偏好保存因此覆盖了正常尺寸。`gallery-native-source-uia-failure.json` 保留事件前后、偏好、Qt／原生状态及 clamp/apply 调用顺序。安装版失败结果位于 `frozen-gui/20261020-075048Z-f59413cc/summary.json`，没有放宽断言或以源码通过替代。

e731 安装注册已限定恢复，原安装四个文件哈希保持，完整载荷与安装记录原子改名保留；62 项证据归档 `e731-windows-evidence.zip`（2,047,469 bytes，SHA256 `db0a0ee37ae4bf5f70ee13974434c697a5f7dd3b9371397f15f6be7925431d78`）。

## 原生状态修复后的最终源码验证

最终相册源码 SHA256 `6bc1ad4ce17a18a8fb729ccaf7bae5f6491d5bc0b67e315f3a7a669275f36ae8`，完整源清单 `final-uia-gallery-source-manifest.json` 的 SHA256 为 `0c6ad5ffd4f8c427883b639f8f2df4d12498112aade9edf9b03fcaeb0b8fb314`。只读查询本相册 HWND 的 `IsZoomed/IsIconic`，避免 Qt 状态滞后期间将原生最大化／最小化尺寸保存为正常尺寸；可靠 Qt 状态切换时的 `normalGeometry` 提供查询失败的回退。

源码验证使用独立 PowerShell 的**真实 UIA `WindowPattern`**，与安装版相同的系统状态切换路径，各执行两次最大化／正常还原。DPR 1.0 每次正常外框 `[8,8,1066,791]`，DPR 1.25 每次 `[11,11,1326,971]`，逻辑客户区始终精确恢复 1040×720；最大化过程偏好未被污染。分别见 `gallery-native-source-uia-final.json` 和 `gallery-native-source-uia-final-scale125.json`，完整有限事件轨迹保留。Windows 相册、复核、记录三个模块 **91 passed，37.34 秒**，日志 `gallery-native-source-uia-final-pytest.log`。这是源码原生路径验证，最终安装版仍单独执行。

此修补只影响窗口几何状态保存，不改候选核验、呈现时限或相册图像绘制。没有重复性能测试直到得到通过；上面追加 1 Hz 的单次 350 ms 保护暂停及严格 FAIL 仍为保留边界，其图库源码指纹和本修补差异已明确。

## 初次 4211 候选包和冻结源码

- 版本：0.4.10（未调整版本号）。
- 构建 ID：`4211d52229824686bdda0862fa593c59`。
- Git 基线：`a836e1dc10408acf01f5bb4445c837dc75a7140e`；构建输入包含当前未提交的第五轮更改，以完整源码清单标识。
- 源码清单 SHA256：`095b1dedeab0f83c2eecf97e8d04119a67bf98687a664387e2136764d439ffac`；构建前逐一核验 2,388 个文件。
- 安装包：`dist/internal/acquisition-fifth-round-20261020/FDM-0.4.10-round5-4211d5222982.exe`。
- 安装包 SHA256：`da83e399811810277570eed3e95393b40b81f41e0341b8b8ed4d2e315637832e`；398,788,594 bytes。
- 使用内部 `--exclude-area-models`，保留内部报告模板和 Microview 运行库。未使用 `--public-release`。
- 安装包提取核验通过：4,818 个载荷文件、434 个生产源码；冻结程序自检通过。提取核验与下面的实际安装分别记录。
- 构建后仅同步验证脚本，用 `validation-overlay-*.json` 记录前后哈希；这些覆盖不修改产品，也不重新解释包中的源码清单。

## 源码测试与修复记录

保留失败尝试，未将多轮结果累加冒充一轮全绿。

| 执行 | 结果 | 后续处理 |
|---|---|---|
| 初次 32 文件 | 660 passed，1 failed，1 teardown error | 旧反馈测试改为显式手工模式；相册替换测试修复双重关闭的清理夹具 |
| 第二次 32 文件 | 658 passed，3 failed | 补全后选测试的帧令牌夹具；修复 Windows 同毫秒时间戳的真实相册排序问题 |
| 定向 4 文件 | 85 passed，1 failed | 源码工作台夹具原先只处理一次事件，可能在首帧绘制前落点；改为有界等待实际 presented 和 ready，不强设产品状态 |
| 最终后选＋工作台 | 98 passed，11.03 秒 | 最终生产代码与真实首帧夹具 |
| 最终紧凑布局＋设置＋相册 | 57 passed，29.08 秒 | 覆盖窄窗最小宽度修复、图库同时间戳顺序及最终设置入口 |

源码测试使用模拟帧及独立配置。此前的 native smoke 与主动取消的调度尝试不替代最终矩阵。`source-visual-final` 因与主机性能隔离时段冲突主动停止；仅停止精确匹配本任务脚本和输出目录的进程，保留取消报告。

## 初次 4211 原生矩阵

**16/16 通过。** `source-matrix-summary.json` 对照每个场景的源码指纹，全部与冻结输入一致。主场 `source-main-result.json`：100 次菜单共 774/774 个采样保持可见、0 次转换；100 组连续移动产生 100 个独立批次；200 次真实导航仅启动采集一次。1600×900 的 100%/125%/150% 完整可见类别分别为 9/4/1；1366×768 可见 5 类。使用 Windows Qt 原生窗口、`--base-device-scale 2`、独立配置，四主题 × 1600×900 的 100%/125%/150% 与 1366×768，串行执行。主场要求 100 次真实下拉菜单、100 组连续两次移动、200 次真实导航，其余场景 6 次导航。过程由 Qt 窗口事件和截图检查；Windows 未安装视频编码器，本轮不将 macOS 过程录像表述为 Windows 原生录像。

## 初次 4211 实际安装版 UIA 与 Excel

**安装器 exit 0，独立目录实际安装通过。** 安装前保存固定 FDM 注册项，已安装 EXE 与已核验载荷哈希一致。普通用户 `B1EA\lishuyang` 经 Explorer → 隔离 cmd 启动器 → 已安装 EXE 启动，进程链核验通过，未提权、正常退出、无强制清理。**最终 9/9 场景 UIA 通过**，全部正常退出、无强制清理，已打开的隔离项目文件哈希未变化。原始结果：`frozen-gui/20261020-040531Z-ecd458a3/summary.json`。早期探针尝试分别碰到 Qt InstantPopup 的 UIA Invoke 不展开菜单、ComboBox 只提供 ValuePattern、QListView 不支持 UIA SetFocus 及 posted 双击不能激活；依据真实控件快照改为本应用窗口限定鼠标事件和实际双击，保留所有失败记录，没有为探针修改产品。实际安装到本任务独立目录；普通用户通过资源管理器隔离启动器启动已安装程序。安装版检查独立于源码自动化，包括四主题相册、历史复核及固定小窗、混合模式默认值、三种存档选项、常驻 Excel 入口与记录停靠。

额外 1600×900@125% 场景使用原生 Windows 窗口和进程级 Qt 缩放；系统 DPI 保持原样，不能称为修改系统 DPI 后的验收。`installed-screenshots/` 的四主题相册及 125% 操作页已经逐张视觉检查。安装版未接设备、未选标尺时显示准备提示，125% 截图仅有两条完整类别行及第三行局部；源码准备就绪场景才验证四条完整类别，不能混同。

Excel 16.0 从常驻按钮完成实际导出：52 个输入、144 条直径、12 个计数结果、486 个保留公式验证通过；编号 `0000123` 保持字符串，VBA、12 个工作表及打印区域保持，原模板哈希不变。只关闭本任务创建并身份确认的 Excel 工作簿/实例。报告 SHA256：`4ef99c7f91af40521afe6404d44585dfacb1ee5319f24b6fc9d4b13ba8a1df80`。

## 模拟采集性能

首轮隔离对照：baseline 30.2904 FPS / P95 16.8576 ms，guard 30.2558 FPS / P95 19.6235 ms；FPS 下降 0.1143%、P95 增加 2.7659 ms，性能门槛通过。但 guard 绘制轨迹出现 8 次显隐转换，轨迹验收失败，保留 `performance-guard-attempt.json` 和配对基线继续诊断，**不标为总体通过**。只读轨迹诊断确认 5 次 `verification_late` 暂停、4 次恢复，最后一段约 23.47 秒保持隐藏；候选批次始终为 1，没有旧帧回退，但该环境下候选可用性不满足验收。

额外非产品原型测试：Windows 4 线程 OpenCV 阶段 P95 17.22 ms、缓存参考 FFT 原型 20.71 ms；1 线程分别为 17.03/21.77 ms。每组 148 个比较的位移、响应及决策一致，但原型在 Windows 更慢，不据此修改产品。原型与正式持续采集并发负载分开记录。使用 Windows 原生 Qt、768×576 模拟帧，连续测量与局部 JPEG 写入，相册保持实际可见；baseline 和 guard 各 30 秒串行运行。CPU 密集构建/矩阵不与性能测量并行。判据为 FPS 下降 ≤5%、P95 帧龄增加 ≤一帧；不代表真实相机采集性能。

补充的每 1000 ms 一次测量/存图对照同样未通过候选可见性：baseline 30.2978 FPS / P95 9.5252 ms，guard 29.9851 FPS / P95 21.2496 ms；FPS 下降 1.0323%、P95 增加 11.7244 ms，但出现 31 次显隐转换。不能将限制归因于仅 150 ms 的高压操作，也不能称候选闪烁已在 Windows 全面解决。两组分别保存 199/199、29/29 张证据，相册运行期间始终可见、无解码或预算残留；帧率指标通过不代表候选持续可用性通过。完整对照见 `performance-comparison.json`。

## 临时调度诊断（未修改产品）

为排除环境与主线程消费顺序，使用独立 wrapper 记录 worker 起止、结果消费、预览截止时间、画布绘制和相册刷新。当前实现与内存调度原型均在实际 DPR 1.0、Qt 因子 0.5、逻辑／原生客户区 768×576 下运行；首帧及暂停时前台 PID 属于本测试进程，相册位于同一进程前台。此前两次性能启动同样显式设置因子 0.5，但未直接记录 DPR，因此不将本次观测追溯当作此前记录。

- 当前实现每 1000 ms 测量一条：29.8853 FPS、P95 帧龄 21.0629 ms、35 次显隐转换，仍失败。
- 内存原型将已完成的核验更早消费，并在原截止时间判断前消费已完成 future：30.0942 FPS、P95 21.3188 ms、27 次转换，仍失败。
- 没有改变核验阈值、帧龄预算和生产源码。两组相册刷新最大 0.60／0.82 ms、入账最大 1.53／1.29 ms，画布绘制最大 9.93／12.14 ms；结果不足以证明消费顺序调整能解决候选可见性，未据此重打包。
- 完整时序保存在 `scheduling-production-1000ms.json` 与 `scheduling-deadline-poll-1000ms.json`。临时 wrapper SHA256：`bc6bbfdd384cb1ed65317abe2e8ca12d1ed2520caa89b83a227d91a74d242d85`。

进一步加入核验 worker 的墙钟／线程 CPU 时间、实际 JPEG 编码、缩略图解码和相册 viewport 绘制跟踪。保持当前实现的组出现 43 次显隐转换；仅将业务数据引起的旧帧重绘合并到原本已安排、仍受原截止时间限制的新帧呈现的内存原型出现 37 次转换，两组仍失败。FPS 为 29.7437／29.9548，P95 帧龄为 20.6032／20.7473 ms。记录分别为 `render-production-1000ms.json` 与 `render-defer-business-1000ms.json`，wrapper SHA256 为 `dd1961737c4e84df3af4cca33a6d26bb2a8d06cd32aa33c3d9d4ae022ccbc995`。线程墙钟减 CPU 时间包括调度、锁、GIL 与 I/O 等待，不能直接声称等于 GIL 耗时；该原型同样没有进入产品或安装包。

图库定点试验保留 header 无可见变化时不重绘、缩略图完成时仅重绘对应卡片的原型，30 秒仍有 37 次显隐转换。随后组合局部重绘、去除模型同步后的重复全刷、提前消费已完成核验和业务重绘合帧，减少为 11 次转换，但仍未通过；FPS 30.2631、P95 帧龄 23.6611 ms。两组分别保存为 `gallery-local-1000ms.json`、`gallery-all-1000ms.json`。新 wrapper SHA256 `cc0a78145745e136fe5225e7ce02a695f1af98a3c3b9efab35ab6c984a2ca398`，图库原型模块 SHA256 `b978f68c8aaa7326654fdcfe380437bc3bcfcf972c7f5bde916d8422a976be6f`。初次 wrapper 参数缩写歧义在解析阶段退出，未执行产品，旧脚本和启动失败记录保留；修正 `allow_abbrev=False` 后才执行上述场景。原型尚未进入包，不将转换次数下降解释为验收通过。

独立持久进程＋共享内存核验原型（其它绘制与调度仍为冻结生产实现）在相同 1000 ms 场景产生 15 次转换，FPS 30.1209、P95 帧龄 20.7010 ms，仍未通过。完整 worker／拷贝／IPC 时序为 `process-alignment-1000ms.json`。wrapper SHA256 `9e65cc2e66572c2112a6c796d5197b0cc8bb802753bbe73928a1294ab12b646a`，进程模块 SHA256 `c6550f218e29d6c2d412edb5a0e4595ee3f9e5c0ffc47a2489d6488aee64a3ba`。该组继续使用原 4211 构建对应的 Windows 源码副本；后续工作区图库优化尚未同步，避免污染比较。

持久进程与全部绘制／调度优化合并后的首个 30 秒、1000 ms 内存原型通过现有判据：FPS 30.3014、P95 帧龄 20.5232 ms、1 次转换、关闭后预算为 0。完整轨迹为 `process-all-1000ms.json`（SHA256 `97c28181455bf8b7e5a2ac32b29953ddcf2cb39d9f34fc30be66efa3aca768b5`）；wrapper SHA256 `5ed3c1a7a497cfff91a394bcbfbd6fa5124e189c11727f82ac303eb4b7d0fbc9`。这只是原型可行性结果，当前 4211 安装包尚不包含该组合，不能把它标记为包的性能验收通过；后续仍需生产接入、重建及冻结程序子进程验证。

相同组合在每 150 ms 接受一条并存图的 30 秒压力组仍失败：30.2376 FPS、P95 帧龄 31.2960 ms、4 次转换；15.325 秒暂停、15.675 秒恢复，18.640 秒再次暂停后直到结束保持隐藏。结果为 `process-all-150ms.json`，SHA256 `a849449b90643edcbcc6ba2bc59be447c514aefaf4988dbfcd6124f8706ae241`。保留这个失败，不能以 1000 ms 原型通过替换它。最终生产代码需重新成对测量，增加停止新增后继续预览的独立恢复观察；安静恢复阶段不参与负载 FPS／P95 的计算。

## 环境恢复与边界

初次 4211 测试安装的注册状态已恢复：7 项固定路径逐项身份校验，仅实际改变的第 0 项恢复；原 0.4.7 的 4 个文件哈希不变，候选目录改名保留，未运行卸载器。结果为 `installation-restored-initial-build.json`。旧包 700 项证据归档于内部交付目录 `initial-build-4211d522-windows-evidence.zip`，SHA256 `e2266489e5bb4f3f0b7aa08c0e88752d03e49f1cfcb5cada5d37fe9a5075eceb`（48,667,934 bytes）。f52e 候选同样完成 scoped 恢复，记录为 `installation-restored-f52e.json`；797 项历史证据归档 `f52e-windows-evidence.zip`，SHA256 `4f4686ff38ba6804d84fee87c38a70995924f3b69574da81ad4f1e9aa167d81f`。最终 4391 安装同样完成限定恢复（`installation-restored-final.json`）：7 项固定注册路径按身份检查，仅实际改变的第 0 项恢复；原 0.4.7 的 4 个文件哈希保持，候选安装目录改名保留，未运行卸载器。关机前 `environment-after.json` 确认无本轮 FDM／Excel 残留，原独立 Python PID 3404 及创建时间保持，未终止该后台进程；可见窗口只有原有设置、代理、系统与 Parallels 窗口，未发现新文档窗口。


主机相机共享已恢复 **on** 并读回确认；随后执行普通 `prlctl stop Windows 11` 成功，最终 **stopped**，与测试前一致。未强制停机、未整机回滚，记录为 `vm-before.json` 与 `vm-after.json`。独立新 PowerShell 进程的 `policies-after.json` 与初始策略一致：仅 CurrentUser 为 RemoteSigned，其余包括 Process 均 Undefined。`environment-after.json` 中名为 `fresh_process_policies` 的字段来自继承临时启动器环境的子进程，因此含进程级 Bypass；它不代表永久策略，最终以单独启动、无 ExecutionPolicy 参数的 `policies-after.json` 为准。

最终原始证据 ZIP 为 `dist/internal/acquisition-fifth-round-20261020/final-windows-evidence.zip`：**1,290 项、50,423,038 bytes**，SHA256 **`27e42d4aa9ce5b3b2e8894d2755dc690dd154c1f1a1a6f81d5fbbf288dc55f3f`**，完整 CRC 与传输哈希通过。ZIP 中最终源清单哈希 `0c6ad5ff…`、安装自检构建 ID 和实际 Excel 文件哈希已分别核对。ZIP 保存 guest 原始结果、窗口轨迹、截图、报告和安装记录；本机聚合 summary、四个 final 别名及关机后状态位于本报告目录，由最终交付 `manifest.json` 逐项记录哈希。原失败记录及历史证据归档均保留；已撤回的本机安装器按上述清单清理。

重建前将初次完整 `dist/windows` 和安装包目录保留到 `dist/history`，避免 clean build 删除。旧 build 证据因 PowerShell 长路径迁移限制分两处保留，旧 frozen-gui 字体缓存残余仍在原处；后续 GUI 使用独立 run ID。没有删除历史文件，完整证据 ZIP 已先完成校验，目录映射见 `initial-candidate-preservation.json`。

真实 Microview、离线原生 x64 Windows、相机拔插/长时间采集和实际多显示器 DPI 切换仍待验收。
