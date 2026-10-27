# 来宾 FDM / Zemax 工作区空间分类

日期：2026-10-22。本报告只处理根代理已取得的来宾目录/大文件/环境快照，未再次控制虚拟机，未删除文件，也没有对来宾全部文件做哈希。环境快照时间为 `2026-10-22T13:59:01.5320180Z`；实际操作前仍需核对占用与活动进程。

## 结论与计量口径

**最大的旧生成物候选为25个精确路径，共24,010,093,787逻辑字节（22.361 GiB）**；另有7个编译中间产物候选，2,088,510,270字节（1.945 GiB）。这32项互不嵌套，合计26,098,604,057逻辑字节（24.306 GiB），是待核对范围，不是本轮已释放空间或已确认可直接删除量。

另外4份运行资源树和2个传输ZIP占5,985,636,651字节（5.575 GiB），目前只知道可能重复。须先与保留副本逐文件完整比较，不能凭名称和大小认定。

多轮 `installation-backup` 只有13个目录、113,467逻辑字节（包含docs中的历史副本），保留即可；真正的大头是旧 onedir/安装器、编译包、重复资源以及独有长时实验输出。

- JSON与CSV的165893条目录路径及逻辑字节已逐项交叉核对一致。`ord` 前缀为 `[67,58,92]`，路径原本就是单反斜杠，无需修正原始证据。
- 下列候选按互不重叠的目录/文件计算，**没有加总父目录与子目录**。工作区整体大小只用于背景，不能再加到候选总量。
- 目录逻辑字节、按路径计量的NTFS分配字节、扫描首次遇到硬链接时分派的unique字节都不直接等于删除可回收量。unique assignment只是扫描记账方式，宿主虚拟磁盘回收量更需另行测量。
- 本表是选定工作区的分类，未覆盖整个C盘；没有列入候选的源码、配置、证据和用户数据仍默认保留。

## A. 最大一组旧分发/冻结产物候选

共同保留条件：先核实包版本/构建身份及当前注册安装不指向此处；保留当前试测与回退安装器、原始runtime资源、源码与独有更改；抽取并校验归档诊断、manifest和被否决构建的证据；检查树内是否混入用户项目/切片/配置/捕获数据。以下路径**不能直接升级成批量删除清单**。尤其opt4k只选择明确的分发子目录，源码和实验结果另行保留。

| 路径 | 逻辑 GiB | 文件数 |
|---|---:|---:|
| `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\dist\windows-full-native-20261013` | 2.0762 | 4771 |
| `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\dist\windows-before-native-20261013` | 2.0760 | 4769 |
| `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\dist\installer-before-native-20261013` | 1.7638 | 2 |
| `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\dist\windows-core-native-20261013` | 0.8871 | 4728 |
| `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\dist\installer` | 0.7406 | 2 |
| `C:\tmp\fdm-cu-live-20261015\build\cu-live-release\full-onedir-rejected-stale-pyz` | 0.9652 | 4767 |
| `C:\tmp\fdm-cu-live-20261015\build\cu-live-release\full-pre-barrier-onedir` | 0.9652 | 4767 |
| `C:\tmp\fdm-cu-live-20261015\build\cu-live-release\core-onedir-final` | 0.8879 | 4730 |
| `C:\tmp\fdm-cu-live-20261015\build\cu-live-release\core-onedir-pre-wording` | 0.8879 | 4730 |
| `C:\tmp\fdm-cu-live-20261015\build\cu-live-release\core-pre-afterselection-onedir` | 0.8879 | 4730 |
| `C:\tmp\fdm-cu-live-20261015\build\cu-live-release\core-technical-onedir` | 0.8878 | 4730 |
| `C:\tmp\fdm-cu-live-final-r2-20261015\build\cu-live-release\core-onedir-final-r2` | 0.8886 | 4732 |
| `C:\tmp\fdm-cu-live-final-r3-20261015\build\cu-live-release\core-onedir-final-r3` | 0.8886 | 4732 |
| `C:\tmp\fdm-cu-live-final-r4-20261015\build\cu-live-release\core-onedir-final-r4` | 0.8886 | 4732 |
| `C:\tmp\fdm-cu-live-opt4k-20261015\dist\installer` | 0.7428 | 2 |
| `C:\tmp\fdm-cu-live-opt4k-20261015\dist\windows` | 0.9657 | 4779 |
| `C:\tmp\fdm-cu-live-opt4k-20261015\build\opt4k\core-onedir` | 0.8884 | 4742 |
| `C:\tmp\fdm-cu-live-opt4k-20261015\build\opt4k\release-final` | 0.8912 | 4755 |
| `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\build\native-autobuild-validation\packaging-source\dist` | 0.8863 | 4724 |
| `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\build\internal-no-area-2026-10-14\before-sync\previous-outputs\dist` | 0.9636 | 4761 |
| `C:\tmp\fdm-cu-live-20261015\build\cu-capture-freeze-dist` | 0.2690 | 273 |
| `C:\tmp\fdm-cu-live-20261015\build\cu-automatic-frozen` | 0.1602 | 98 |
| `C:\tmp\fdm-cu-live-20261015\build\cu-automatic-frozen-final` | 0.1602 | 99 |
| `C:\Users\lishuyang\Downloads\FDM-0.4.8-acquisition-rebuild-20261015\FDM-0.4.8-acquisition-rebuild-20261015.exe` | 0.3714 | 1 |
| `C:\tmp\fdm-cu-live-20261015\build\cu-live-release\installer-rejected-stale-pyz.exe` | 0.3710 | 1 |

上述25项按路径分配字节合计24,136,306,800，仍不能当作已确认可释放量。包含全量面积模型的旧包也应保留其输入模型来源，不能因当前试测包排除了面积模型就删除原始模型。

## B. 可重建编译中间产物候选

这些是PyInstaller工作文件、临时编译对象和冻结探针中间文件。清理前保存 `Analysis/PYZ/EXE/COLLECT*.toc`、`warn*`、`xref*`、构建日志等独有诊断，确认源代码/依赖和最终保留交付可用；不要清理其父级验证目录。

| 精确路径 | 逻辑 GiB |
|---|---:|
| `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\build\pyinstaller-before-native-20261013` | 0.4645 |
| `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\build\native-autobuild-validation\packaging-source\build\pyinstaller` | 0.4564 |
| `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\build\internal-no-area-2026-10-14\before-sync\previous-outputs\build\pyinstaller` | 0.4647 |
| `C:\tmp\fdm-cu-live-opt4k-20261015\build\pyinstaller` | 0.4666 |
| `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\build\temp.win-amd64-cpython-312` | 0.0091 |
| `C:\tmp\fdm-cu-live-opt4k-20261015\build\temp.win-amd64-cpython-312` | 0.0646 |
| `C:\tmp\fdm-cu-live-20261015\build\cu-capture-freeze-work` | 0.0192 |

本组2,088,510,270逻辑字节与A组不重叠。小型 `.pytest_cache`/`__pycache__` 并非当前收益重点；没有为了凑数字泛扫所有第三方环境缓存。

## C. 必须先完成身份比较的可能重复项

| 精确路径 | 逻辑 GiB | 比较要求 |
|---|---:|---|
| `C:\tmp\fdm-cu-live-20261015\runtime` | 1.2082 | 与保留runtime逐文件比较相对路径、大小及SHA-256；保留所有独有模型/相机/模板资源。 |
| `C:\tmp\fdm-cu-live-preview-fix-20261015\runtime` | 1.2082 | 与保留runtime逐文件比较相对路径、大小及SHA-256；保留所有独有模型/相机/模板资源。 |
| `C:\tmp\fdm-cu-live-save-peaks-20261016\runtime` | 1.2082 | 与保留runtime逐文件比较相对路径、大小及SHA-256；保留所有独有模型/相机/模板资源。 |
| `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\build\native-autobuild-validation\packaging-source\runtime` | 1.2082 | 与保留runtime逐文件比较相对路径、大小及SHA-256；保留所有独有模型/相机/模板资源。 |
| `C:\tmp\fdm-artifact-997f7787dfcf45dc828e10e4fb743528.zip` | 0.3709 | 先只读ZIP中央目录，找到已校验的包/源快照替代物；不能用文件名随机标识判断无用。 |
| `C:\tmp\fdm-artifact-dc47187d3d06498aa348ce45bd7aa7eb.zip` | 0.3709 | 先只读ZIP中央目录，找到已校验的包/源快照替代物；不能用文件名随机标识判断无用。 |

Downloads/FDM原始 `runtime` 为1.210 GiB，本轮保留；其中面积模型约1.112 GiB，不应按缓存处理。上述4份runtime约1.208 GiB/份并不足以证明完全一样；本次尚未读取全部模型字节计算哈希。

## D. 明确保留与避免误删的范围

| 内容/路径 | 本轮观察与保留理由 |
|---|---|
| `C:\Users\lishuyang\Downloads\fiber-diameter-measurement` | 整体15.237 GiB；只在A/B/C中逐项细分，不能删除工作区父目录。`.venv`约1.538 GiB、`.git`约0.135 GiB、runtime以及源码/测试/脚本/文档/样本保留。 |
| `C:\Users\lishuyang\Downloads\fdm-native-validation-installed-20261013` | 0.969 GiB，既有原注册安装。不能因名字带validation就当临时包；历史四个基线身份见保留规则，仍需来宾本轮复核。 |
| `C:\tmp\fdm-cu-live-opt4k-20261015\build\opt4k`中的`hour-awake`、`hour`、`formal*`、`sampler-diagnostic`、`wake` | 分别约2.359、0.553、0.076/份、0.121、0.074 GiB，为实验输出/诊断。不是可重建编译缓存；未经完整归档及身份比较不清。 |
| `C:\tmp\fdm-cu-live-final-r4-20261015\build\cu-live-workspace-hour-windows-final-r4-evidence` | 约0.175 GiB，包含14036个文件，是长时实验证据。 |
| 所有fdm多轮workspace源码、docs、validation_reference、validation_vendor、source.tar等快照 | 某些含当时未提交差异或独有输入，不能因为新版代码在别处存在就删除。 |
| `C:\Users\lishuyang\Downloads\flat-field-grating` | 约0.434 GiB；`.venv-zemax`约0.158 GiB，是项目运行环境，连同patched third_party、模型、脚本和证据保留。 |
| `C:\Users\lishuyang\Downloads\ANSYS Zemax OpticStudio 2024 R1.00 x64` | 约2.386 GiB，是离线安装材料；在离线/授权恢复需求确认前保留，不等同无用缓存。 |
| `C:\Program Files\Ansys Zemax OpticStudio 2024 R1.00` 与 `C:\Program Files\ANSYS Inc` | 分别约1.620、1.396 GiB；环境快照可见ansyslmd/lmgrd/Tomcat许可组件正在运行，不能清理。 |
| Downloads下`fullheight-resolution-2026-10-10`、`leadership-brief-2026-10-10`、`最终候选光路_全高四方案_2026-10-10` | Zemax相关光学数据/汇报交付物，全部保留。 |
| `C:\TextileExecutionBridge` | 约1.143 GiB；环境快照有其中python进程正在运行，属于其他项目运行服务，不是FDM缓存。 |
| `C:\tmp\fdm-cu-auto-stage-reference`、`C:\tmp\dsx-evidence-collector-20261022` | 分别约0.069、0.006 GiB，前者原厂参考材料，后者本轮正在交付的工具，保留。 |

既往清理的52+4个目录不能重复计入A/B。保留当前/回退安装器的宿主全量哈希及历史诊断ZIP验证见 `host-retention-evidence.json`。Zemax共享路径 `C:\Mac\Home\Documents\Zemax` 属宿主数据，不沿共享链接删除。

## 定向身份补查结果

已读取根代理取得的 `guest/candidate-identities.json`，覆盖A组全部25个目标，目标集合与本报告完全一致，补查无错误或跳过重解析点。17份manifest中16份标记 **0.4.7**、1份标记 **0.4.8**；有版本信息的8个EXE条目中6项为0.4.7、2项为0.4.8（其余30条EXE无可用产品版本）。宿主保留的当前试测和回退均为0.4.10且已全量哈希确认。因此这些有身份记录的目标属于较早产物，旧分发物候选判断得到进一步支持。

边界仍保留：12份manifest的 `source_commit=unknown`，不能仅凭提交号宣称可精确重建当年二进制。补查没有发现 `.fdmproj`、`.fdmslide`、`.db`、`.sqlite`，**但零命中不证明没有独有图片、用户配置、修改过的模板、标定或其他普通格式输出**。111个按名称匹配的目录多为 `torch/timm/skimage/cv2/.../data` 等包内依赖资源；数据/配置/模板目录名本身不能判断其内容属于用户还是软件。仍需保留并校验归档manifest、诊断、被否决构建证据及任何独有数据，不能据此直接执行整树删除。

上述补查只提供版本/必要身份字段及有限文件类型排查，没有逐文件证明整个目录重复，也没有改变A/B/C的逻辑体积或升级为已确认可释放空间。完整记录与每目标补查摘要已加入 `guest-findings.json`。

## 定向补查与下一步

机器可读范围见 `guest-findings.json`：`selected_paths`含精确路径、类别、字节和观测边界，`small_identity_reads_requested`给出仅针对A组的有限读取建议。

1. 本轮已完成有限读取：候选包根已有 `release-manifest.json`、`candidate-manifest.json`、`source-manifest.json` 的必要身份字段和EXE版本；未转储完整配置/凭据。
2. 列出候选内部 `.fdmproj`/`.fdmslide`、用户输出和配置路径；存在独有数据则移出候选，不能把包根直接判定为纯生成物。
3. 对需要保留的构建诊断/被否决构建证据，先归档并验证可读性及身份。C组若未来要去重，才对具体替代关系进行全量文件哈希，不做全盘哈希。
4. 审核最终精确路径后才能形成清理实施范围。本任务到此仍为只读分析，不调整页文件、VSS、Windows更新、安装软件或虚拟磁盘。
