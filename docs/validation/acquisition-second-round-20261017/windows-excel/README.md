# 模板报告运行时验证（2026-10-17）

结论：本轮最终模板源在 Windows 隔离目录通过运行时、真实 Excel 导出、内容保留、失败保护和取消清理验证。生产源及验证脚本哈希见 `source-fingerprints.json`；汇总见 `summary.json`。

| 检查 | 结果 |
| --- | --- |
| Windows 运行时与模板契约 | 43 passed，包含真实 PowerShell 请求解析、父进程 Restricted、owner 不匹配保护，以及 GPO/关闭失败的模拟测试 |
| 三模板 × 0/1/1488/1500/2000 条 | 15 次真实 Excel 导出成功；每次关闭诊断均 `ok=true` |
| 独立只读核对 | 15 份原始直径精确匹配；864 个输入区公式、VBA、工作表数量、形状数量、打印区及编号前导零保持 |
| 实际进程级策略 | 父进程 `Restricted` 时，应用子进程显式 `Bypass`，模板标准读取成功；恢复父环境后仍为原 `RemoteSigned` |
| 实际导出失败 | 用无效标准触发 COM 工作流失败，已有成功报告 SHA-256 不变 |
| 实际取消 | 等本次 Excel PID/创建时间写入 owner 后取消；仅本次实例被终止，已有报告 SHA-256 不变 |
| 结束状态 | 没有遗留 Excel 进程；用户、机器及 GPO 策略值与开始完全一致 |

环境是 Parallels Windows 11 的交互用户 `B1EA\lishuyang`，PowerShell `5.1.26100.9711`，Excel `16.0`。父进程不带 `PSExecutionPolicyPreference`，CurrentUser 为 `RemoteSigned`，MachinePolicy/UserPolicy 未配置；原控制台和 ANSI 编码均为 936，工作进程明确输出 UTF-8（65001）。见 `environment.json`、`environment-after.json` 及各次导出的 diagnostics。

1500 条的分组容量提醒、2000 条的模板计数范围提醒按既有契约返回 `review_required`；这些报告成功保存，并且全部记录写入，没有静默截断，也没有用本地统计替换模板公式。

使用现有 `scripts/validation/validate_cu_template_excel.py` 和 `verify_cu_template_reports.ps1`。一次性异常探针源码保留为 `exception_probe.py`，在隔离源码根目录执行；它只修改当前进程的策略环境变量，不执行全局或用户级 `Set-ExecutionPolicy`。所有模板均复制自先前验证副本，没有改写原模板、原目录或用户工作簿。

隔离目录为 `C:\tmp\fdm-cu-live-second-round-20261017`。临时模板、报告和请求位于其 `.tmp\windows-excel-20261017\matrix`，未提交工作簿。界面诊断控件的 macOS 模拟截图位于仓库 `.tmp/acquisition-second-round-20261017/template-diagnostics-macos.png`；已检查中文可读、展开/复制/导出控件可达。

这是隔离源码的 Excel 验证，不代替冻结安装包验收。组织 GPO 拒绝和 COM Close/Quit 异常使用模拟测试；没有调整真实组织策略或强制造成真实 Excel 关闭故障。真实导出矩阵使用 `show=False`；系统关联自动打开报告及其失败保留成功状态已有单元测试，未在该矩阵中驱动可见 Excel 窗口。未涉及 Microview 硬件。

证据：`runtime-tests.log`、`matrix.log`、`export-results.json`、`preservation.log`、`preservation-results.json`、`exception-probes.log`、`exception-probes.json`。测试时允许后台构建并行，本轮耗时不作为性能基准。

纳入版本管理的文本日志统一使用 LF 换行并移除末尾多余空行，测试输出内容保持不变。
