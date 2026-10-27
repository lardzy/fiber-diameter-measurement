# 清理实施前保留规则审核

日期：2026-10-22。用户现已授权按前次审计开始清理。本文是对实施代理的保留审核，只读取宿主保存的来宾证据和历史脚本；本审核者没有控制虚拟机或删除文件。历史 `retention/guest-findings.json` 中的“未获删除授权”是当时的快照，不应覆盖本次用户授权；实际删除仍须满足下述身份与归档条件。

## 审核结论

白名单只取 `retention/guest-findings.json:selected_paths` 中类别为 `old_generated_distribution_candidate` 的25项和 `rebuildable_intermediate_candidate` 的7项，共32个互不嵌套路径、26,098,604,057逻辑字节。无需扩大到父目录，也不要包含C组4份runtime和2个ZIP。

25项分发目标已由 `guest/candidate-identities.json` 全覆盖；17份manifest为16份0.4.7、1份0.4.8。当前与回退0.4.10安装器在宿主已核实。12份旧manifest的提交号为unknown，因此本次允许移除过时产物的依据是“已被新交付取代、证据及独有输入已保护”，不能声称每个旧包均可逐字节重建。

推荐成本最小且可复查的方式：**小型诊断、配置、模板和普通数据直接归档；大模型只与明确保留的来源进行逐文件SHA-256比较，完全相同则记录替代关系，不同/缺失则归档唯一内容或暂留该项。** 不需要全盘哈希，也不需要把所有PySide/Torch二进制重新打包。

## 本轮证据中的具体保护点

1. 15个包内 `runtime/content-templates` 各4文件、890,370字节，总13,355,550字节。全部归档成本很小；同大小不等于同内容，可按内容哈希去重，并保留每个原始路径映射。内部模板不能因文件未被Git跟踪而删除唯一副本。
2. 两个Downloads旧full包内各有6个面积模型、1,193,709,670字节。先对照保留的 `C:\Users\lishuyang\Downloads\fiber-diameter-measurement\runtime\area-models`。只有文件相对路径、大小与SHA-256全同才视为已有保留副本。不得以当前试测排除面积模型为由丢弃模型来源。
3. 六个包有 `runtime/segment-anything`，每份78,712,822字节、4文件；这些ONNX也采用上述输入保护规则。模型后缀至少覆盖 `.pth/.pt/.onnx/.safetensors/.ckpt`，不能只处理面积模型。
4. 所有17份包内相机runtime约1.4 MB/份；`runtime/area-infer`约0.58 MB/份，以及 `runtime/caching`每份81,827字节，直接归档即可。`caching`在此是包内资源名，不凭名称认成可丢弃缓存。模板/相机/旧算法源码的独有变体随归档保存。
5. `C:\tmp\fdm-cu-live-opt4k-20261015\build\opt4k\release-final` 的子目录 `core-onedir`之外还有**13个根文件、3,001,361字节**。当前目录汇总未给出完整文件名。将这13项先全部识别并归档（若发现新大文件则暂停该项），不能只保存内部manifest后删父树。
6. `cu-automatic-frozen`、`cu-automatic-frozen-final` 同时含 `dist`和`work`；诊断匹配须递归覆盖`work`内的TOC/warn/xref和根目录spec，不能只扫描dist。
7. `full-onedir-rejected-stale-pyz`与单独的`installer-rejected-stale-pyz.exe`属于被否决构建。保留原EXE/安装器SHA-256、manifest、相关校验报告及错误原因。无需再保留完整失败运行包，但也不能把“删除旧包”写成“当时验证通过”。

## 可直接实施的归档规则

匹配均对**32个已核对根下的文件**进行，大小写不敏感；普通遍历不跟随junction/symlink。模式用于决定保留内容，**绝不能当删除通配符**。规则应输出命中的精确路径、理由、字节、SHA-256与ZIP成员名。

| 优先级 | 匹配 | 处置 |
|---|---|---|
| 1 | `*.toc`、`*.log`、`*.log.*`、`warn-*`、`xref-*`、`*.spec`、`*.iss`、`*.sha256` | 全部归档；不以文件大小上限静默跳过。 |
| 1 | `*manifest*.json`、`*identity*.json`、`*receipt*.json`、`*result*.json`、`*status*.json`、`*summary*.json`、`*report*.json` | 全部归档；包含release/candidate/source/installer身份。 |
| 1 | 所有`*.json/*.jsonl/*.csv/*.tsv/*.xml/*.ini/*.cfg/*.conf/*.toml/*.yaml/*.yml/*.md/*.txt` | 直接归档；即使部分属于依赖默认配置，也比漏掉用户设置代价小。只在本地保留，不转储正文到对话。 |
| 1 | `*.fdmproj/*.fdmslide`、`*.db/*.sqlite/*.sqlite3`及其`-wal/-shm`伴随文件 | 归档并记录发现；若任务仍在写入，暂停包含它的目标，不能复制不一致数据库后继续删。 |
| 1 | 图片`*.png/*.jpg/*.jpeg/*.tif/*.tiff/*.bmp/*.gif/*.webp`；视频`*.mp4/*.avi/*.mov/*.mkv`；文档/模板`*.docx/*.doc/*.xlsx/*.xls/*.pptx/*.pdf`；数值资料`*.npy/*.npz/*.h5/*.hdf5` | 全部归档，包含普通格式的实验输出；不因为`.fdmproj`零命中而忽略。重复内容可按哈希去重。 |
| 1 | 位于`runtime/content-templates/**`、包内`runtime/camera/**`、`runtime/area-infer/**`、`runtime/caching/**`的所有文件 | 全部归档，目录匹配以路径段为单位，覆盖外层`FiberDiameterMeasurement/`、`core-onedir/`和`_internal/`等实际包根。 |
| 1 | 路径段为`profile/profiles/config/configs/settings/calibration/calibrations/templates/capture/captures/output/outputs/results/logs`的完整树；以及`profile-*`实验目录 | 全部归档或从本轮目标中剔除。先列清单，避免名称猜测导致遗漏。目录名`data`只触发审查，不把全部Torch/Python依赖复制成“用户数据”。 |
| 2 | 候选内应用/探针源码与构建入口`*.py/*.pyx/*.pxd/*.c/*.cpp/*.h/*.hpp/*.ps1/*.cmd/*.bat` | 归档；包内少量源码不必为节约几MB排除。编译产物`.obj/.pyd`不因此一起保留。 |
| 2 | 包内所有模型文件；`runtime/area-models/**`与`runtime/segment-anything/**` | 与**白名单以外**的原始runtime比较；按相对路径+大小+SHA-256核实，记录替代绝对路径。不同/缺失内容先归档，无法完成则只暂停相应目标。 |
| 3 | `release-final`前述13个根文件，其他不属于已知依赖/生成物的根旁证、未分类文件 | 必须归档或逐项说明纯生成用途后才能删除；不设“未知即删除”默认分支。 |

为缩小归档，同内容的模板、相机DLL、文本资料可以按SHA-256存一次，但manifest必须逐条保存`source_path -> archive_member -> sha256`，不能只写不同内容的计数。最简单实现仍是保留相对目录结构的ZIP；以上小文件总成本远低于重复的运行时DLL。

无需归档完整字节的已确认生成物：包里的第三方DLL/PYD/EXE、Python标准库ZIP、PyInstaller`*.pkg/*.pyz`及中间对象`*.obj/*.o/*.lib/*.exp/*.pdb`。这项豁免只适用于已经识别为相应包/编译结构的文件；未知`.zip/.dat/.bin`、根目录的用户EXE或模型不能仅凭后缀套用。旧安装器及应用EXE至少保存路径、长度、SHA-256、版本与归属。未知文件如体积很小可直接归档，减少反复人工分类。

## 必须排除或单独保护的边界

- 本轮32项的父级工作区、源码根、`.git`、`.venv`、原始runtime、参考样本、`source.tar`、`validation_reference`、`validation_vendor`、所有安装备份，以及不在32项里的实验目录全部保留。原注册安装整树保留，不只保留四个身份锚点。
- `%LOCALAPPDATA%\FiberDiameterMeasurement`、回退的`%APPDATA%\FiberDiameterMeasurement`及用户目录里同名配置树均不在白名单内，不需清理。当前`settings.py`明确从该处读取`settings.json`，还有`area-models`旧输入目录。若本次脚本以SYSTEM运行，**不能用SYSTEM的环境变量替代用户实际路径**来判断用户配置是否存在。
- Zemax/flat-field-grating/相关光学输出、Ansys许可组件、TextileExecutionBridge、DSX工具和原厂参考目录均不触碰。
- `build/opt4k/hour-awake`、`hour`、`formal*`、`sampler-diagnostic`、`wake`以及r4长时工作区证据不是本轮候选。C组的重复runtime/ZIP仍保留。
- 不清Windows组件、页文件、VSS、更新目录，不调整软件安装、注册表、相机共享或虚拟磁盘。

## 可复用的历史脚本机制与要改的地方

可复用 `.tmp/windows-cleanup-audit-20261021/{archive_stage,delete_stage,cleanup_common}.py` 和 `.tmp/optical-cleanup-20261021/` 的机制：先归档、来宾逐成员哈希验证、传回宿主再验证、以plan SHA锁定范围、核对原注册安装/活动进程/受保护源码、拒绝重解析点、逐项记录部分失败及磁盘前后值。

不可直接复用旧删除清单或旧的硬编码“52项/4项”；本轮为32项，包含单文件和Downloads目录。旧`targets.py`只检查`.fdmproj/.fdmslide`及TOC/log等，覆盖不足，应换成本报告匹配规则。旧脚本只用总字节数检测目标改变，可能漏过等长替换；本轮至少锁定完整相对路径、长度、mtime/file ID（可用时）清单，并重新验证所有归档/模型替代文件的SHA。无须把第三方大型DLL全部反复哈希。

旧脚本的`busy()`会输出完整CommandLine；本轮只输出PID、进程名、可执行路径及匹配原因，必要的命令行匹配在内存完成，避免参数/凭据落盘。注册表仅记录FDM既有卸载项必要身份，不做更改。

## 实施验收与单项暂停条件

1. 生成独立不可覆盖的本轮plan：32个精确绝对路径、类别、快照字节、保留列表、plan SHA。断言大小写无重复、无父子嵌套、无白名单与受保护路径相交。先确认宿主两份0.4.10安装器仍可读且哈希正确。
2. 新做来宾preflight：目标存在/类型相符、无reparse祖先/后代、无FDM/探针/构建相关在用进程、原注册安装仍指向保留树。产生文件清单和归档命中清单，明确每个未归档文件的生成物类别；不能仅依赖前次目录汇总。
3. 完成上述小型归档和模型来源比对。ZIP CRC通过、每成员SHA与源文件一致，复制到宿主后再次检查ZIP SHA及成员。归档源路径应在32项之外；宿主验证成功是删除前置条件，不必再向用户重复索要已给授权。
4. 删除前再次核对plan SHA、活动进程、目标树清单、所有需要归档/保留输入SHA、原安装锚点。逐项执行；目标发生新增/变化、未知大文件、锁定/访问失败、归档不匹配、模型没有保留副本、疑似业务数据在写入时，**仅暂停该目标**并记录，继续其他已满足条件的目标。不要把未处理的失败项扩为父目录强删。
5. 结束后报告逐项存在性、完成/跳过/部分失败、实际移除逻辑字节，来宾C:自由空间前后差及采样时间。验证源码/原安装锚点/注册关系/两份宿主交付包不变；归档可重读、所有受保护路径仍存在。实际释放量受系统同期写入影响，不等于候选逻辑字节；宿主虚拟盘文件大小另列，不宣称压缩回收。

本次已发现的可操作缺口是模型逐文件替代关系、32项新文件级清单、`release-final`13个根文件身份以及归档验证尚须由实施阶段补齐。这些都可自动补齐；不需要因研究报告原先写“候选”而再次询问用户。若模型比较未完成，优先跳过对应含模型目标，仍可处理其他已满足条件项。

## 审核输入

- `retention/guest-findings.json`、`retention/host-retention-evidence.json`、`guest/candidate-identities.json`。
- `guest/transfer/scan/directories.json`，本次重新用其核对包内runtime、模板及release-final旁证体积。
- `.tmp/windows-cleanup-audit-20261021/{targets,archive_stage,delete_stage,cleanup_common}.py`、`.tmp/optical-cleanup-20261021/targets.py`。
- `runtime_assets.toml`、`scripts/build_support.py`、`scripts/build_windows_onedir.py`和`src/fdm/settings.py`的来源/配置规则。当前源码仅帮助解释输入类别，不用于假定旧包与当前文件相同。
