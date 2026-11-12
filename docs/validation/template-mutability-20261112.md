# 所选模板允许修改与同名替换

日期：2026-11-12。版本保持 0.4.14。

## 调整行为

所选模板直接按配置的源路径交给 Excel；没有重新引入模板缓存、固定目录副本或默认模板回退。

- 删除实时 Excel 打开前的预期 SHA 比较及打开后的源文件复验。
- 删除一次性模板编辑、检查及导出中的相同限制。
- 删除报告页因源文件与已关联工作簿的旧指纹不同而拒绝导出的限制。导出读取该关联来源路径上当前的模板内容。
- Python 读取诊断指纹期间发生保存／替换，或无法完成哈希读取时，不再拒绝模板选择和打开。诊断分别记录 `changed_during_read`／`unavailable`，SHA 留空；不把可能混合的读取内容冒充已加载工作簿的身份。
- 已打开的工作簿继续使用，不因源模板修改自动重建或改写人工报告内容。重新选择、重新创建时读取当前文件。
- 打开 Excel 后遇到忙碌，继续已有候选工作簿，不再次读取、验证源文件，也不重复创建工作簿。

仍保留实际运行所需检查：文件存在、Excel 实际能打开、输入映射可用、写入确认与关联代次。缺失或无效模板仍明确报告失败。原输出报告暂存和发布机制保留，导出不覆盖源模板。

哈希现在仅为可缺失的诊断数据。两个可用指纹不同可以提示源文件已修改，不阻断既有同步；缺少指纹不再被误判为源文件已变化。

## 本机自动回归

macOS / Qt offscreen：**200 passed，2 skipped，6.74 s**。两项跳过是仅 Windows 执行的 PowerShell 集成检查。

覆盖读取期间真实文件内容改变、诊断读取失败、缺文件／目录拒绝、含量及直径模板的同名替换后导出、已有工作簿保持关联、无指纹时正常打开和继续同步，以及模板选择、报表信息维护、忙碌恢复与窗口呈现等既有回归。

```sh
PYTHONPATH=src QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q \
  tests/test_trial_template_sources.py \
  tests/test_report_template_selection.py \
  tests/test_excel_live.py \
  tests/test_excel_template_runtime.py \
  tests/test_live_excel_output_boundaries.py \
  tests/test_excel_presentation.py \
  tests/test_trial_excel_lifecycle.py \
  tests/test_cu_template_contract.py \
  tests/test_trial_template_flow_validation.py
```

原始日志：本地忽略目录 `build/template-mutability-regression-20261112.log`。

## Windows 桌面 Excel 验证

现有 Windows 11 虚拟机、桌面 Excel 16.0，独立目录中的合成 XLT 副本：**28 项断言通过**。验证脚本为 `scripts/validation/verify_template_mutability.ps1`，没有构建或安装新包。

- 刻意携带过期 SHA，实时打开仍成功；从实际工作簿读取到 D25=18.25，新增报表页保留。
- 修改同一源路径的模板后，旧工作簿继续写入 D26=21.5，并保持自己的报表内容。
- 显式重新创建后，工作簿读到同路径上更新后的报表页内容。
- 一次性检查、导出及编辑均接受过期 SHA；导出文件重新打开后，D25=22.75，新增报表页及当前模板内容保留。
- 缺源文件返回 `template_missing`；缺少映射输入页返回 `template_mapping`，没有把这些真实故障作为模板指纹变化放行。

另在 Windows 执行生产 PowerShell 函数与请求分支的契约检查：**23 项通过**，覆盖哈希采集失败降级、同名变化、缺文件、忙碌候选复用和旧关联保留。执行脚本为 `scripts/validation/verify_template_worker_contract.ps1`。

已核对 Windows 执行的两份生产脚本与本次源码 SHA-256 一致。测试开始前没有 Excel 进程；退出后再次确认没有遗留 Excel 进程。

Windows 结果与模板副本位于本地忽略目录 `build/template-mutability-20261112/`，其中 `windows-excel-result.json`、`windows-contract-result.json`、`windows-cleanup.json` 分别记录上述流程、契约和退出检查。此结果证明当前脚本与桌面 Excel 的流程，不等于对用户另一台电脑的旧安装包完成更新，也不声称复现了此前源文件被改写的具体原因。

本轮不增加版本，不构建安装包；现有安装版须更新相应代码后才会采用新行为。截图、模板副本、Excel 文件和原始日志不加入 Git，其他未提交工作保留。
