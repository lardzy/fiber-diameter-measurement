# 试验 Excel 工作进程验证

日期：2026-11-11。Windows 11 测试虚拟机，系统报告版本 `10.0.26300.0`，桌面 Excel COM 版本 `16.0`。本记录验证软件及模板，不代表真实相机或实际样品计量验收。

## 复现准备

在已具备 FDM 源码依赖的环境，从仓库根目录生成合成输入。目标文件必须不存在，脚本不会覆盖历史证据；这一步不启动 Excel，也不读写模板和用户配置：

```powershell
python scripts/validation/prepare_live_excel_worker_inputs.py --output C:\tmp\fdm-excel-probe\requests.json
Copy-Item src/fdm/live_experiment/excel_live.ps1 C:\tmp\fdm-excel-probe\excel_live.ps1
```

将三份模板的**副本**放入该独立目录，分别命名为 `sheet.xlt`、`sheet11.xlt`、`sheet_dia.xlt`。这些名字只是验证器的固定输入名，不限制产品中模板的文件名。然后在 Windows 桌面 Excel 环境执行：

```powershell
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File scripts/validation/verify_live_excel_worker.ps1 -Root C:\tmp\fdm-excel-probe
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File scripts/validation/verify_live_excel_interruptions.ps1 -Root C:\tmp\fdm-excel-probe
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File scripts/validation/inspect_trial_excel_formulas.ps1 -Root C:\tmp\fdm-excel-probe
```

异常验证会操作专用测试 Excel 的 F2／Esc、弹出测试窗口，并终止自身创建且已核对身份的测试 Excel，须在可用的交互桌面运行。脚本的 JSON 输出留在 `-Root`；重复复现使用新目录以保留旧证据。合成输入固定为 2026-11-11、三条直径 `12.5／15.25／18.75`、两个类别。它使用当前模型及投影接口生成，因此随接口完善可包含比原始记录更多的诊断字段，历史 JSON 不回写。

## 实际验证结果

`scripts/validation/verify_live_excel_worker.ps1` 对独立 STA 工作进程执行 **61 项检查，全部通过**。使用工作进程返回的 PID 和该 PID 的 `EXCEL7` 原生窗口接口进行核对；不使用 `GetActiveObject`，不借用或关闭其他 Excel 实例。

- 三份模板均成功创建未保存工作簿、写入数值、计算计数及结果：原含量模板、用户提供的 `sheet11.xlt`、直径模板。
- 追加只写新记录；删除后旧尾部清空；数值入账和计算完成分别回报。
- 输入区在批次外无密码保护；非受管报告单元格可编辑；解除关联撤回新增保护，工作簿留给用户保存。
- 保存／另存后仍跟随同一个工作簿对象，文件保存状态与同步状态分别返回。
- 同时关联含量与直径工作簿，后台含量同步保留用户正在查看的另一工作簿、工作表及滚动位置。
- 旧输入修订被拒绝。用户关闭工作簿后返回关闭状态，不自动重开；显式的新关联代次可以重建完整数据。
- 模板宏失败不返回同步成功；仍保留工作簿关联，完整重新同步能够在同一工作簿恢复。
- 普通 `.xls` 文件也通过 `Workbooks.Add(模板路径)` 创建新的未保存工作簿；源文件哈希不变。
- 工作进程正常退出后不关闭交给用户的报告。验证脚本随后仅关闭自己创建的测试副本。

完整 JSON 记录保存在本地 `build/trial-excel-20261111/worker/worker-validation.json`。模板本体、测试生成的 Excel 文件和媒体不纳入 Git。上述 61 项最后运行的工作进程 SHA-256：`f175af254c63f5a18e592f853d858ab1175767402e740d00c646263d32ec919b`。

`scripts/validation/verify_live_excel_interruptions.ps1` 另有 **30 项断言通过**，包含请求身份核对。脚本在专用 Excel 中真实按 F2 编辑非受管单元格：原生焦点从 `EXCEL7` 网格切换到 `EXCEL6` 文本编辑窗口，出现新的宽 1 像素、高 41 像素的插入光标，证明已进入编辑状态。同步返回 `excel_busy` 且关联保留；Esc 后 `Excel.Ready=True`、光标返回原网格，同一工作簿恢复完整同步，计数正确。

脚本还实际弹出 `Application.InputBox`，观察到模态窗口后发起同步，同样验证忙碌与恢复。随后按 PID 和启动时间核对并终止该脚本创建的 Excel，确认返回 `workbook_closed`，新代次显式重连可以新建实例并恢复全部输入。数据在 `worker-interruptions.json`。这次测试只终止自己的可丢弃测试实例，不属于产品中的退出行为。

三种模板的一次性导出也分别成功，无警告，记录见同目录 `oneshot-summary.json`。直径例子中 FDM 原始均值与 Excel 受管原始区均为 `15.5`，模板分组均值为 `15.75`，两者没有混为同一指标。

## 三份模板的公式证据

`scripts/validation/inspect_trial_excel_formulas.ps1` 使用独立 Excel 实例读取模板副本的公式、命名区域、表结构和统计覆盖范围，读取前后核对源模板哈希。

| 模板 | SHA-256 | 核实的覆盖范围 |
| --- | --- | --- |
| 原 `sheet.xlt` | `2ecc9cdd777ee37b8555192ce612bceb205d252f390b296775d6063973d108a0` | 原始统计 25～1524 行，共 1500 条；分组统计 25～1512 行，共 1488 条 |
| 用户 `sheet11.xlt` | `56f6f1510932d1d1a33b34b6831671b17f5ab4366d26dea320a371f705d90417` | 表结构及上述原始／分组覆盖范围与原模板相同，没有扩大 |
| `sheet_dia.xlt` | `c532498d6123550df080c9d3d55b6ce252f175112897bce654af8767fa7eefce` | 原始计数及多数报告使用 25～3000 行，共 2976 条；TWC 与 ASTM 部分公式使用 25～2012 行，共 1988 条 |

含量模板的直接公式示例为 `=COUNT(D$25:D$1524)`。直径模板的 `C8` 为 `=COUNT(C25:C3000)`；`C9` 为 `=IF(C8>0,IF(分组统计!D7="",0,分组统计!D7),0)`，因此 `C9` 是**分组均值**，不能用它判定原始输入平均值错误。工作进程另外用 `WorksheetFunction.Average(raw_range)` 核对受管原始区域，与 FDM 原始均值比较；分组均值作为独立信息返回。模板仍按自身规则生成报告。

完整公式证据保存在本地 `build/trial-excel-20261111/worker/template-formulas.json`。

## 在真实 Excel 中发现并修复的问题

1. Windows PowerShell 5.1 将没有 BOM 的中文脚本按旧编码读取。生产脚本采用 UTF-8 BOM，协议仍为无 BOM 的 UTF-8 JSON 行。
2. `UserInterfaceOnly=True` 时，外部 COM 的部分 `ClearContents` 仍被 Excel 拒绝。批次在同一 STA 中暂时解除本进程添加的输入保护，写入控制参数后、运行模板宏前重新保护；异常路径也恢复保护和 Excel 显示／计算设置。不设置 `Application.Interactive=False`，避免超时终止进程后留下不可操作的 Excel。
3. CU 分组宏内部依赖 `ActiveWorkbook`。即便调用带工作簿名的宏，也必须先激活其所属工作簿，完成后恢复原来的窗口和工作表；否则同时打开直径工作簿时会进入 VBA 错误路径。
4. 跨工作簿的 `Activate()` 可能返回布尔值。已显式丢弃返回值，避免它混入 PowerShell 函数结果并使 JSON 应答字段变空。
5. 已关闭的工作簿 COM 对象可能返回空名称，而不是抛出异常；两种情况均识别为关闭。
6. 真实模态窗口也会使部分 COM 属性返回空。空名称现在结合进程身份、`Excel.Ready` 和工作簿集合的可读性判断；忙碌时不丢弃关联。

## 验证边界

首次完整重建与常规增量分别记录，不能用首次清理全部原始输入范围的耗时代表连续测量性能。原始写入回执发生在重新计算之前，也不等同于用户已经看到重绘后的报表；端到端延迟和 P95 以协调器的专门性能记录为准。

本轮已实测宏错误、工作簿关闭、多工作簿恢复、真实 F2 编辑、模态忙碌和 Excel 进程退出恢复。不同 Office 版本、用户修改过的模板及设备环境仍需各自验证。
