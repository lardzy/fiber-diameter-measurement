# Windows 0.4.9 候选安装包验证

最终候选构建、解包校验、自检和 5 组冻结界面验证通过。最后发现的 715 px 窗口增高问题已修复并重新构建；最终冻结程序在 166 次逐阶段尺寸断言中保持 1366×690 或 1280×670。交付文件位于 `dist/internal/ui-redesign-20261016/`，内部模板保留，面积模型排除。

该结论不包含真实 Microview 连续运行、实体 Windows x64 性能或实际覆盖安装。旧候选与失败证据保留，见 `README-intermediate.md`、`intermediate-geometry-audit.json` 和交付目录的 `intermediate/before-small-height-fix/`。

## 交付身份

| 项目 | 值 |
| --- | --- |
| 文件 | `FDM-0.4.9-ui-redesign-20261016.exe` |
| 大小 | 398,573,949 bytes |
| SHA256 | `26113c7b25e7fd2f1451f09ab06a8f9d361954f77a24898bdb310fb8c0fe5562` |
| build_id | `a2185916a592452f9e3a36a5270b4494` |
| 构建 | `scripts/build_windows_installer.py --exclude-area-models` |
| 配置 | full 内部包，包含 3 份报告模板，不含 area-models |
| 来源 | 独立同步目录，419 个 src/native/runtime 文件最终哈希核对通过 |

源仓库 HEAD 为 `006035ad81f7b188632ada1a12ad746849bcc019`，本轮包含未提交工作。隔离构建目录无 Git 元数据，包内 `source_commit=unknown`；准确内容由 `evidence-final/source-manifest.json` 记录，最终快照 SHA256 为 `f447bd996f27fc20c1763525c05a52a6a7486cdb04b259e7781bb0e22d48d0fb`。不得将 HEAD 等同于全部本轮代码。

`installer-final-transfer.json` 记录 VM 到 host 传输核对，主代理另在 host 独立计算 SHA256，一致。交付目录同时提供中文离线验收清单、`SHA256SUMS.txt` 与 `交付校验.json`。

## 环境与隔离

- Windows 11 ARM64 虚拟机，build 26200，运行 x64 Python 3.12.13 / MSVC 1944 / PySide6 6.10.2 / OpenCV 4.13.0.92；PyInstaller 6.19、Inno Setup 6。
- 独立工作目录：`C:\tmp\fdm-cu-live-ui-redesign-20261016`。没有覆盖原开发目录、已有安装或旧安装器。
- 旧 VM 安装器 `C:\Users\lishuyang\Downloads\FDM-0.4.8-save-publish-fix-20261016.exe` 只读核验，SHA256 `bf7a32367b490e8510858bd3d9cc39b2262e41582ff33273b78001a582f3e403`，未修改。此前报告指向的 host 旧包路径本轮实际不存在，已在 `resources-before-build.json` 如实记录。
- 冻结 UI 每轮设置独立 LOCALAPPDATA / APPDATA，旧配置和旧项目也只复制已知测试副本。没有使用用户真实项目进行写入。

## 回归与原生采集桥接

| 检查 | 结果 |
| --- | --- |
| Windows 初轮基础回归 | 136 passed |
| 最终扩展回归 | 307 passed + 24 subtests；4 failed 为同一导出页四主题 1 px 溢出 |
| 导出页最小修复后定向复跑 | 21 passed，包括四主题布局与默认全部已打开图片范围 |
| 小客户区修复后 Windows 路由回归 | 29 passed，包括 6 组尺寸/恢复试验页组合 |
| 原生桥接 + 真实 MainWindow + 测试 SDK | passed，errors=[] |
| onedir 打包自检 | passed |
| 同一安装器解包文件校验 | 4,818 个文件全部通过 |
| 同一安装器解包自检 | exit 0，ok=true，errors=[] |

没有在导出页 1 px 修复后重新跑整个 311 项组；已通过的 307 项证据和受影响的 21 项复跑分别保留，未把初轮失败擦除或伪装为整组重跑。

原生验证涵盖自动进入采集、隐藏 SDK HWND 与 Qt 像素显示、AFTER_PROCESS 图像、共享 helper、首点击固定测量帧、数字补计不选类、自由测量 30 μm、20 次入口操作、5 次子页往返、2 次停启及记录/标尺保留。其输入是 `build/test-sdk/MVAPI.dll` 测试 DLL，**不是物理 Microview 设备**。原始 Qt paintEngine 警告保留；最初 PowerShell 将 stderr 警告当错误，调整测试包装器后按真实退出码及结果断言通过。测试 SDK SHA256 为 `9773333999dd42ba53b9b2ec4b39324e4b1a02ec380c04cdaa7a4f1a03ce7757`，已确认未混入正式包。

原生 MainWindow 测试执行于最后的小高度可见性时序修复前；桥接实现未变，最终包再次通过内置 native/bridge transport 自检，最终路由由新增 Windows 29 项及 5 组冻结 GUI 覆盖。

正式包原厂 SDK：MVAPI.dll `1d22ce270ea4aa37fb1073943a707c82f44f255164e480e1877d606b531933ea`；saa7130.dll `cbb60864f7c2526ce32515132bc04b89eb2c5e177ec5af62804e8b68e7cb6315`。

## 冻结界面与旧数据恢复

直接运行**交付 EXE 解包得到的程序**，不是源码 UI 或另一个构建。

| 场景 | 客户区 | 结果 |
| --- | --- | --- |
| 石墨深色，干净配置 | 1366×690 | 29 次尺寸断言通过 |
| 实验室浅色，干净配置 | 1280×670 | 29 次尺寸断言通过 |
| 深蓝，干净配置 | 1366×690 | 29 次尺寸断言通过 |
| 暖灰，干净配置 | 1280×670 | 29 次尺寸断言通过 |
| 已知 0.4.8 配置及项目副本 | 1366×690 | 恢复保存及 50 次尺寸断言通过 |

每轮验证功能入口、样品编号/名称、操作者、标尺、先选/后选、保存按钮可见；20 次主入口调用以及实时/含量/直径页切换保持导航；操作/记录/报告切换中固定元信息位置不变。四轮只访问页面后没有无意义保存提示，全部自然退出 exit 0。低高度操作区允许滚动，固定元信息和保存按钮仍可访问。

旧测试项目原有 **20 条补计**；新版本连续 20 轮补计及重复保存，每轮逐一核对落盘，最终 **40 条补计**。这里没有创建或虚构直径测量。旧设置和项目原件未变，输出是新隔离副本。

实际虚拟桌面为 2582×1644，Windows availableGeometry 为 2582×1548，OS DPI=192；测试设置 `QT_SCALE_FACTOR=0.5`（Qt DPR=1），另将应用客户区缩为上表大小。它覆盖标题栏/任务栏留白下的小客户区操作，**不等于实体 1366×768 屏幕在 150%/200% 缩放下的验收**。

冻结界面实时输入来自 VM 的 OpenCV Camera #0（Parallels 共享摄像头，前轮曾为黑帧/噪声，最终轮有实际摄像画面）。它证明冻结版预览和布局可显示；不属于真实显微相机质量、Microview 流程或设备性能证据。运行日志中的无对应 OpenCV 设备/backend 警告保留。原始截图含现场环境/人像，仅作为本地验证证据，公开展示应使用主报告的模拟显微图。

## 小高度缺陷的红绿验证

首轮 probe 只在进入前检查 client_size，视觉复核发现进入含量页后实际增加到 715 px。新增逐阶段 GetClientRect 断言后，原冻结包明确失败（`strict-geometry-red.log`），撤回候选没有被冒充为通过。

根因是切页时旧测量工具栏短暂参与新试验页最小尺寸计算，以及状态刷新中先显示再隐藏工具栏。修复只调整 `acquisition_workbench.select_page` 和 `MainWindow._sync_workspace_mode` 的可见性时序，没有缩小固定元信息或削减可达操作。Windows 源回归 29 项通过后重新构建单个最终候选；最终 5 组共 166 次实际客户区检查全部匹配请求，保存/恢复也通过。

启动最终浅色组时 Parallels 曾返回 255 `Invalid argument`；确认未创建该轮目录或应用进程后继续剩余组，没有重复启动未知状态的应用。工具错误原文保留在 `final-frozen-clean-light-parallels-start-error.log`，其后最终组均通过。

## 文件、自检编码与失败证据

- 第一次解包文件验证在可编辑模板处失败：release manifest 按契约将模板 `sha256=null, editable=true`，验证脚本误把 null 当预期哈希。追查后源文件、onedir、解包三方模板字节一致；脚本改为必须满足可编辑模板路径契约，并使用**最终源快照哈希**严格核对。没有跳过模板；模板验证器修订本身无需重建或修改安装器。后续重建针对小高度生产修复。
- 全部 4,818 文件通过后，解包自检已返回 exit 0，但验证脚本以 UTF-8 读取 stdout 失败。原始 14,861 字节保留于 `evidence/extracted-selfcheck.raw.stdout`（另有 host 原件），SHA256 `dcc1a7db4f11740b8ec80dfb2623adbc4f7f5c7965cf2b1386b915e1b85d41d7`。
- 原始输出为完整 CP936 JSON；首个 UTF-8 无效字节在 offset 1334，字段 `functional_checks.analysis_pipeline.batch.summary`，内容为“批处理完成：成功 2 张，失败 0 张，共 2 张。”。严格 CP936 解码并重新编码与全部原字节一致，没有 replace/ignore，未发现混杂日志。转换后的 UTF-8 JSON 独立断言 ok/errors/build_id。最终候选也采用相同严格处理：原始 14,867 字节，SHA256 `f3a02c45dc18456cc7611f8dad07b66d8e4d9cb9e9e00af5984baeab2a19d998`，见 `evidence-final/selfcheck-encoding.json` 与 `evidence-final/extracted-selfcheck.json`。
- 构建器原有 stdout 捕获使用 `errors=replace`，其中文展示不作为本轮严格自检依据；完整原始字节和严格规范化报告补足这一边界。终端转录可能仍有乱码，最终结论读取 `evidence-final/` 中 UTF-8 JSON 和原始 stdout；`evidence/` 保留撤回候选记录。
- 归档时 Windows 字体服务锁定各隔离 profile 下的 Microsoft/Windows/Fonts 缓存。最终 ZIP 仅明确排除这 120 个系统字体缓存文件，前轮保留 132 个、最终轮保留 106 个应用报告/日志/截图/UIA/项目/设置证据；清单分别见 `evidence/evidence-archive-inventory.json` 与 `evidence-final/evidence-archive-inventory.json`。最终 ZIP SHA256 为 `533693841363eeff19215e10bb3590b2ab3a6f166e4ebe961cf90c7080ae51c7`。

最终包中 README.md 已核对为 0.4.9 文档，源文件与解包文件 SHA256 均为 `fe25c721b0986af7f487df8a10b031e7cde7401921fd637c87cf843a96f93596`。

## 尚未验收

本轮只执行安装器构建与**同一 EXE 解包核验**，没有执行 Inno 安装器覆盖现有安装，也没有验证安装/升级/卸载注册表及文件关联。固定 AppId 与管理员安装可能影响已有安装，因此保留原安装环境。

真实 Microview 设备至少 30 分钟连续运行仍待目标电脑验收。4K 连续动态核验仍未通过（主报告保留本机长尾/拒收证据）；不得以本轮 VM UI、原生测试 DLL 或静态图像结果替代。随包中文清单列出目标电脑步骤。
