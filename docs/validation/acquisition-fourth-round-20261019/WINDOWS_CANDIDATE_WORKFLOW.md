# 第四轮 Windows 候选包流程

产品版本保持 **0.4.10**。独立构建目录为 `C:\tmp\fdm-cu-live-fourth-round-20261019`，保留所有旧候选、原安装和虚拟机已有工作。内部包使用 `--exclude-area-models`，保留内部 CU 模板及 Microview 组件，不使用公开发行裁剪选项。

## 已采集的基线

2026-10-19 本轮实时检查：虚拟机 `Windows 11` 已处于 **running**，由用户当前环境启动，本轮未启动它；相机共享 on。普通用户 `B1EA\lishuyang`，旧安装 0.4.7，CurrentUser 策略 RemoteSigned，Excel 已注册，没有运行中的 FDM 或 Excel。Python、VC 工具、Inno Setup、已核验提取工具均存在；第四轮目录尚不存在。见 [Windows 基线](windows/environment-before.json) 和 [VM 状态](windows/vm-before.txt)。

历史第三轮的 stopped 状态不作为本轮基线。交付完成后保持本轮开始时的 running 状态，不擅自关机。

## 执行步骤

1. 等本机生产代码和针对性测试收敛后，运行 `sync_acquisition_fourth_round_windows.py --build`。只写新的第四轮目录，校验源码及内部运行资源的 SHA-256；既有 Python 环境只提供解释器与依赖，不改动旧源码或依赖。
2. 运行 `build_acquisition_fourth_round.ps1`。默认执行第四轮候选控制器、证据与保存兼容测试，以及前三轮布局、记录、主题、显示和 Excel 回归；`-TestsOnly` 可仅验证。测试通过后构建 `0.4.10-no-area-models` 安装器。已有同名本轮安装器时拒绝覆盖。
3. `verify_fourth_round_installer.py` 提取但不安装，核验所有载荷、内部模板原始哈希、真实 Microview DLL、排除面积模型、冻结程序 self-check、构建 ID 和构建期间源码不变。
4. `receive_acquisition_fourth_round_artifact.py` 仅回传本轮隔离目录内的文件，经 Parallels 私有链路传输并比对 SHA-256。已存在的本机目标拒绝覆盖。交付目录为 `dist/internal/acquisition-fourth-round-20261019/`。
5. 安装前运行 `manage_acquisition_fourth_round_installation.ps1 -Action Backup`，仅备份固定的 FDM 卸载项、文件关联项和旧安装关键文件哈希。使用 `/NOICONS /TASKS=""` 安装到本轮 `dist/windows/installed-candidate`，禁止复用目录；安装完成后捕获这些注册表项的状态，用于确认后续还原对象仍属于本次安装。
6. GUI 测试使用隔离配置和本轮样图，暂时关闭 VM 相机共享避免使用主机摄像头。资源管理器通过本轮 `.cmd` 启动辅助文件设置仅作用于子进程的 `LOCALAPPDATA` 与 `APPDATA`，验证普通用户下实际安装程序启动、显示及正常退出，并记录 Explorer → 辅助进程 → 程序的父进程关系。
7. 回收日志和截图后运行管理脚本 `-Action Restore`。只还原本次改变且仍与安装后快照完全一致的 FDM 注册表项；不调用可能清除旧关联的卸载程序。把本轮候选目录保留为 `installed-candidate-validated`，核对旧安装哈希、用户策略与相机共享恢复，保持 VM 运行。

安装流程由整机 checkpoint 改为上述局部还原：只读核查发现 VM 原有独立后台任务持续运行，整机回滚可能回退其在测试期间产生的状态。本轮不停止该任务、不做整机快照或回滚，也不更改全局执行策略。

## 验收边界

此文档描述流程，不代表候选包已构建或安装验收已通过。结果以本目录后续 JSON、测试日志、安装器身份和验证报告为准。Windows 虚拟机没有 Microview 实体相机；对齐阈值、持续动态采集帧龄和真实设备吞吐仍由离线真机步骤验证，不以模拟图像或虚拟机结果替代。

## 当前执行记录

- 已建立隔离源码与本机原生扩展，最终基底 ZIP SHA-256 为 `c9927f8dd0a798b819028c27be2c19be8e58d6d846ea0e028b9ddd3f5e06c321`。后续小修按独立补丁清单记录，并在来宾重新核验源文件（最初 2182 项，加入局部安装还原脚本后 2183 项）；各 `*-patch.json` 保存补丁与文件哈希。Parallels 嵌套子进程调用有间歇性 255 错误，因此实际使用宿主直接调用 `prlctl`、私有链路传输和完整清单校验；未修改系统执行策略或防护设置。
- Windows 首次完整定向回归为 **558 通过、1 失败**。唯一失败是诊断时钟精度造成派生耗时缺失：此环境 Python 3.12 的 `monotonic` 为 15.625 ms 精度的 `GetTickCount64`，算法使用更精细的 `QueryPerformanceCounter`。见 [时钟实测](windows/clock-resolution.json) 和 [保留的失败日志](windows/pre-timing-fix-tests.log)。诊断计时改用 `perf_counter`，生命周期超时仍使用原时钟；后续 worker、性能、候选与界面定向 **56 通过**，没有将缺失耗时伪造为零。
- 参考帧缓存单独验证 **32 通过**；与标注几何缓存组合后相关回归 **149 通过**。
- 性能对照见 [原始指标比较](windows/performance-comparison.json)。最初测试器的 1 ms Python 观察定时器明显干扰吞吐，因此后续关闭该观察器，产品定时器不变。即时呈现探针仅用于诊断，不能作为产品验收。参考帧缓存后的 8 秒生产策略对照达到数值门限，但 30 秒密集测量暴露了增量标注布局及慢核验呈现等待的问题；失败数据全部保留。
- 最终生产版本增加有界回退：核验超预算后先更新实时像素，候选仍只在精确匹配的已核验帧可见、可点击。30 秒同条件对照由 **30.3015 → 30.3166 FPS**，P95 帧龄 **9.155 → 18.936 ms**，满足下降不超过 5%、帧龄增量不超过一帧 33 ms 的门限；**198/198** 张局部 JPEG 写入成功，相关 Windows 回归 **154 通过**。这是模拟源、实际 Qt/核验/写入并发测试，不是真实 Microview 性能。
- 重压下触发 3 次预览回退；2354 次被动绘制样本中 695 次候选可见，约 **29.5%**。这个比例不是精确的墙钟可点击比例，也不代表慢设备上候选始终可点。暂时隐藏是帧一致性保护；真实导航的逐点接受另行通过。离线设备仍需实测候选的连续可用性及对齐阈值。
- Windows 原生 Qt 的最终 **16/16** 主题/分辨率/缩放场景通过，实际截图像素已验证。含量 1600×900 的 100% 完整显示 9 类、125% 显示 4 类；1366×768 显示 5 类。主场景执行 200 次导航，其余场景 6 次。详见 [矩阵及选图](windows/source-visual-final/)。
- 候选包已完成构建与提取校验：**4818** 个载荷文件、**433** 个生产/运行资源源文件匹配；内部 CU 模板和真实 Microview DLL 保留，仅排除面积模型。冻结程序 self-check 通过。版本 **0.4.10**，构建 ID `bd7168a578074dfb8a5758d918e86c68`，安装器 SHA-256 `1b1b7a01a32d5f077488113e96e97a535fc33539dad530e978644f9420e50546`。见 [安装器身份](windows/installer-identity.json)。
- 实际安装到独立候选目录成功；资源管理器经隔离配置辅助文件在普通用户下启动、主窗口显示、无修改正常退出均通过。见 [实际安装](windows/actual-installation.json) 与 [普通启动](windows/explorer-start.json)。安装交互没有使用用户原安装目录。
- 包内 `source_commit` 为 `unknown`，因为隔离来宾源码不是 Git checkout；不能表述为干净提交的构建。基底提交为 `a836e1dc10408acf01f5bb4445c837dc75a7140e`，完整快照 SHA-256 `099f6056c78b073961b64db06a2e0d76a356390554ccb9e5d9945a0be6e5e94f`，补丁清单和 [完整源清单](windows/source-manifest.json) 共同确认来源。构建后仅验证器夹具路径小修，产品未变。
- 安装版 GUI 第一组记录夹具因多层测试日志路径超过 Windows MAX_PATH 而在产品启动前失败；验证器改用本轮短夹具目录。随后 7 个界面场景通过。Excel 已由常驻入口生成，但初次后置审计因期望 JSON 路径白名单拒绝；保持白名单不变，将期望文件复制进运行目录后，已生成报告的 52 输入、144 直径、12 计数、486 公式、宏与打印区域审计通过。失败记录保留，不把验证器修复描述为产品修复。完整 Excel 整链路重跑通过，应用及 Excel 均正常退出。安装版共 8 个场景具备通过证据，详情见 [最终报告](windows/final-validation.json) 与 [精选截图](windows/installed-gui-selected/)。
- 真实 Microview 相机验收尚未执行。

## 最终交付与环境恢复

候选包位于 `dist/internal/acquisition-fourth-round-20261019/FDM-0.4.10-round4-bd7168a57807.exe`，398,773,132 字节，旁附 SHA256SUMS、源快照清单、发布清单、Windows 验证报告和离线 Microview 步骤。

安装版验证已完成。7 个普通界面场景覆盖空态、四主题、1600×900、1366×768 和 1067×600 可用尺寸；另一个完整 Excel 场景验证固定入口、三操作者全数据、52 项输入（含编号前导零）、144 个直径、12 个计算计数、486 个原模板公式、宏及打印区域。源码矩阵的真实自动 worker、候选点选和帧门禁另列记录，不能把无相机安装版空态检查写成真实硬件采集。

已执行局部恢复：仅第 0 项卸载注册项发生变化并还原，全部 7 个限定 FDM 键签名与基线一致，原 0.4.7 的 4 个关键文件 SHA-256 未变。验证目录保留为 `installed-candidate-validated`，它是本轮候选副本，不是用户原安装。相机共享回到 **on**，虚拟机保持 **running**；新进程中执行策略与基线完全一致。本任务 FDM、Excel、宿主 HTTP 传输服务和验证器均已退出，原后台服务 PID 与创建时间未变。未创建整机快照、未回滚虚拟机、未运行卸载器。见 [恢复结果](windows/installation-restored.json)、[环境检查](windows/environment-after.json)、[无进程策略覆盖检查](windows/policies-after.json)。

原始安装版截图/日志归档 SHA-256 为 `0e2b3a73a740568d6a461e78373938878627b49c85964f318784fe361a512836`。完整回传 ZIP 保存在本机 `.tmp/windows-installed-validation-evidence.zip`，文档仅保留结果、精选截图和必要日志，隔离配置及字体缓存不进入永久文档。
