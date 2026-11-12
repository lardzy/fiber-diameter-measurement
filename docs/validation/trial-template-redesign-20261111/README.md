# 含量试验模板重做验证

日期：2026-11-11；安装验收完成于北京时间 2026-11-12 00:16。版本：**0.4.14**。基线提交：`ec21a0c085ab77e785710fde49ad29db0fa44fab`。

## 修复范围

- 默认直接使用程序内模板；不再由旧个人副本遮蔽更新。外部选择、缺文件及配置损坏不静默回退。
- 报告页与主设置“采集与试验 → Excel 模板”共用来源配置，分别管理含量和直径。
- 模板按完整工作簿处理，打开时保留自己的标准、组距和初始报表页；不再自动执行 CU“分组”。显式重算恢复工作表可见性、活动页及滚动位置。
- 选择模板后先建立候选工作簿、校验来源并写入本次数据，成功才切换关联。失败保留原工作簿。连续选择时，较早请求失败不阻塞最新选择。
- 模板更换失败与量化输入同步分别记录；同一路径更换失败后继续测量，旧簿可以更新，但界面仍显示更换未完成及原因。重新选择有效源、成功替换或结束关联后才清除。
- 一次性导出与实时工作簿使用同一成功绑定来源，不因为后来修改全局默认而改用另一模板。
- 增加实际源文件哈希、工作簿名和可导出的诊断，以及包内安装验证入口。

源码实现没有改版本或项目数据格式。上述验证完成时尚未提交、未推送；其他 DSX 工作保留。用户当时单独授权构建测试包和安装验证。模板修复未进入后续加载优化提交的遗漏及补交验证见[2026-11-12 交付复核](../template-delivery-20261112.md)。

## 模板身份及比较口径

使用用户提供的 `sheet.xlt` 副本，SHA-256：

`56f6f1510932d1d1a33b34b6831671b17f5ab4366d26dea320a371f705d90417`

该文件含 12 张工作表，初始活动页“特种毛(报出)”。重点核对四张新增页：特种毛(报出)、麻棉(报出)、分组加权统计、织物纤维含量报告。

对照基准是在同一桌面 Excel 中正常打开原模板。比较全部工作表的名称、顺序、可见性、控件数量和活动页；四张新增页抽样检查 A1、B2、D5、F8、G10、K20 的公式、格式、字体、填充及合并区域。原始数据核对 D25 中的 18.25。抽样格式检查不声称遍历了每个单元格。

正常打开保留模板自己的 Workbook_Open/Activate 事件，遵循 Office 现有宏策略，没有改动信任中心。生成用于“同名替换／缺失输入页”测试的临时夹具时，才在 SaveAs 前关闭事件和提示，避免兼容性对话框阻塞自动化；这一处理不用于产品正常打开验证。

## 自动回归

| 项目 | 结果 |
|---|---|
| macOS 模板、协调器、报告入口及偏好回归 | 190 通过，2 项 Windows 专用检查跳过 |
| Windows 相同回归 | 192 通过，最终构建前执行 |
| 设置导航、试验设置与报告相关补充回归 | 52 通过，4 个子用例通过；与上面有重叠，不相加 |
| Windows PowerShell 生产函数／请求分支契约 | 21 项检查通过，使用内存 COM 替身，无 Excel |
| 发布自检相关回归 | 27 通过，6 个子用例通过 |
| 源码差异空白检查 | `git diff --check` 通过 |

发布自检初次在 macOS 受限沙箱中有 4 项失败：共享内存 `/psm_*` 被系统拒绝。HEAD 相同探针、相同源文件也复现；正常权限执行后 27 项全部通过。这不是 Excel 功能回归。

## 前期 Windows 源码完整流程

Windows 11 虚拟机、桌面 Excel 16.0，真实 Qt 主窗口和生产 PowerShell worker。源码阶段 `native-flow-v4/result.json`：`passed=true`，`installed_validation=false`。后续失败状态持久化与默认模板验收补充，使用下节最终安装版再次完整验证。

选择阶段 31 项检查及第二个程序进程的重启阶段 7 项检查全部通过，覆盖：

- 正常“采集与试验 → 含量 → 报告 → 选择模板”路径；旧个人模板存在时不遮蔽程序默认。
- 既有测量完整写入所选模板、12 页及初始显示保持、源哈希与 worker 一致。
- 真实“重新统计报表”菜单，分组宏执行后恢复新增页和初始显示；原始值与人工命名备注保留。
- 真实导出按钮、保存对话框及后台导出；重新用独立 Excel 打开磁盘报告核对页面、抽样格式及 18.25 原始值。
- 离开并返回试验复用工作簿；同路径替换重新读取不同内容。
- 缺文件不回退、输入映射错误不替换成功绑定、重新选择有效文件可恢复。
- 原工作簿及人工备注保留；第二个 FDM 进程读取保存选择并建立不同 Excel 进程身份。
- 无未捕获 Qt 槽异常，用户原模板前后字节哈希一致。

测试只自动填入文件选择路径、停用相机启动和确认丢弃测试项目。导出完成后的系统 `os.startfile` 被拦截，以防报告被送入用户既有 Excel；随后独立 Excel 读回是真实执行。这里不声称覆盖系统文件关联双击行为或真实相机。

前两次流程失败分别是测试目录缺少程序默认模板、探针使用本地化 COM `Comments` 属性。第三次通过产品打开／重算／导出后，在生成替换夹具的兼容性提示处超时。以上均保留失败记录；最终不是跳过步骤通过。超时夹具的 Excel 按创建时记录的 PID＋启动时间精确清理，用户既有 Excel 未处理。

## 构建与安装

**最终安装验收通过**：`installation-result.json` 的 `passed=true`；实际安装 EXE 运行结果同时为 `passed=true`、`installed_validation=true`、`source_only=false`。

| 项目 | 最终结果 |
|---|---|
| 构建身份 | `8dc50dbfbf414826bd1884e56d50e8a8`，版本 0.4.14，完整内部配置，保留模板、不含面积模型 |
| 新安装 | 成功；安装目录 `C:\Program Files\FDM Template Test 8dc50dbfbf41` |
| 文件核对 | 安装前后核对 4,821 个文件；除明确替换的可编辑模板外，与构建产物字节一致 |
| 覆盖安装 | 先备份原模板，替换为用户 `sheet.xlt`，再运行相同安装器；用户模板 SHA 保持不变 |
| 程序自检 | 实际安装 EXE 的 `--self-check --json` 返回 `ok=true` |
| 完整交互 | 选择阶段 42 项、重启阶段 7 项检查通过；包含真实主窗口、生产 worker 和桌面 Excel |
| 原环境 | 正式软件注册项、用户偏好文件及原有 Excel 进程身份均保持；未关闭或保存用户工作簿 |
| 测试收尾 | 两阶段最终同步完成，测试工作簿残留均为 0，无清理错误 |

首次进入含量试验时，没有自定义配置，仅保留一个无效的历史个人模板夹具。实际打开来源为安装目录 `runtime/content-templates/sheet.xlt`，worker SHA 与用户源 `56f6f151…d90417` 完全相同；12 页、活动页“特种毛(报出)”和新增报表页的抽样格式与独立 Excel 基准一致。这项独立证明直接替换默认文件生效。

随后通过实际报告页按钮选择另一份模板副本，确认当前数据 D25=18.25 写入；执行显式重算和真实导出后，新增页及格式仍保留。把无效模板覆盖到同一个选定路径，替换失败后再入账 19.75：旧工作簿和代次保持，D26=19.75，独立更换失败标志及原因持续存在。恢复有效模板后，两条记录进入新簿并清除失败提示。第二个安装版进程重启后读取已保存选择，正确显示新增页。

独立安装器与正式包使用同一程序 payload，仅安装标识、名称、目录及注册行为隔离。正式安装脚本和产品版本保持原样。此次验证覆盖真实安装、同版本覆盖安装及模板保留；没有改写机器上正式产品的文件关联，也不将它表述为正式 AppId 升级验证。

正式测试安装包（399,741,898 字节）：

`fiber-diameter-measurement-setup-0.4.14-no-area-models.exe`

SHA-256：`98f8ab781dcb19bf6e8d417a5a3535dbe136656206b5914064f0816067a0aa8d`。

实际执行的隔离安装器 SHA-256：`535b47eb2c98f11b6fe543e7d6efd1772aa0f6177ba3576e5974f21e99b5b747`。两个安装器的安装身份差异见本地 `installer-identity.diff`；安装后的程序文件已经逐项核对。

源码快照为 HEAD 加本次经过审查的任务差异及内部运行资源，不包含其他未提交 DSX 修改。独立快照无 `.git`，构建 manifest 如实记录 unknown／dirty；其基线、逐文件 SHA-256 和差异由 `.fdm_snapshot.json` 单独证明，不伪造提交身份。

首次构建外层 PowerShell 将 PyInstaller 正常 stderr 日志误作异常，未生成发布清单与安装器。已改为文件重定向并依据进程退出码判断，随后完整重新构建。官方 Windows 版本包含文件生成器将 `version.auto.iss` 从 LF 写成 CRLF；规范化摘要与原输入完全一致、版本仍为 0.4.14，其他 5766 项输入字节不变。保留初始清单，并在构建清单记录这一个生成文件转换前后的摘要，不跳过其余来源检查。

最后两次验收补充的文件变更也逐项记录转换前后的大小和 SHA，最终来源清单 SHA 为 `8e4079eb3d20dcd1b7238d20a970b0bf14d8f3ea5fc8baf7d4922b7e0e4cd46b`。安装 helper 的空注册表序列化问题在实际安装前修复；安装器尚未关闭时的一次预检只写出状态快照，已单独保留，未执行安装。最终四阶段均重新执行成功。

## 本地证据

不加入 Git 的原始输出位于仓库 `build/`：

- `template-redesign-implementation-20261111/pytest-mac-final-v5.xml`
- `template-redesign-implementation-20261111/windows/pytest.xml`
- `template-redesign-implementation-20261111/windows/worker-contract.json`
- `template-redesign-implementation-20261111/windows/native-flow-v4/`
- `template-redesign-implementation-20261111/source-snapshot.json`
- `template-redesign-implementation-20261111/final-v5/SHA256SUMS.txt`
- `template-redesign-implementation-20261111/final-v5/evidence/refinement-v5/pytest-final-v5.xml`
- `template-redesign-implementation-20261111/final-v5/evidence/installed-validation-v5/installation-result.json`
- `template-redesign-implementation-20261111/final-v5/evidence/installed-validation-v5/installed-template-flow/`
- `template-redesign-implementation-20261111/final-v5/evidence/source/.fdm_snapshot.json`
- `template-redesign-implementation-20261111/final-v5/evidence/source/build/template-isolated-package-v5/`
- `template-worker-diagnostics-20261111/release-self-check.json`

截图、模板副本、Excel 文件、完整日志和安装产物仅本地保存。现场另一台电脑的原故障没有被远程直接复现；本轮验证针对已识别的源码路径与实际构建。

操作说明见 [模板选择与验证入口](../../trial-template-selection.md)。原 CU 静态证据见 [CU 模板加载复核](../../research/cu-template-workflow-20261111.md)。
