# CU 实时测量 Windows 内部发行验证 · final-r2

本报告只描述 **final-r2 阶段验证快照，不是最终用户交付版本**。后续已另行安排 Save As 后证据路径重定位修正（R3），不回写或覆盖本次证据。旧 R1 的 905 文件清单、安装包、自检和正式性能 JSON 均保留，不能用它们替代 R2 运行证据。

## 来源与构建

- 独立 Windows 目录：`C:\tmp\fdm-cu-live-final-r2-20261015`。
- 952 文件源码清单 SHA256：`9ca1c13ea22b23e0f5025e727933fd671d8b375f65813a94e276cb402ab0bde2`；原生源码 SHA256：`4ec56fd735aa4c8edcdcfe2905bc488f3e8cc1534c2363a0d28e4cee2cfc13c8`。
- 两个 profile 均使用原有 `scripts/build_windows_onedir.py`，**clean=true**，没有禁用 PyInstaller 分析或使用缓存绕过。
- `core` 和 `full --exclude-area-models` 均通过实际冻结自检；CU 检查 revision 2，11/11，含 deferred build/apply、组距 undo/redo 的真实数值计算。
- 两个 onedir 和安装包解出 EXE 中的 7 个关键模块均对实际 PYZ 做递归代码对象比较，仅忽略 co_filename；字节码、常量、嵌套代码、异常/行表及参数均与本次源码一致。
- 性能环境仅补齐缺失的 `psutil==7.2.2`，所有其他 distribution 版本保持原样；旧 R1 数据不回写。

## 正确性

Windows 专项 **671 passed、7 subtests**。人工金标 **246/246**，192 个正宽度位级相等；运动金标 **111/111**；自动金标 **177 组：175 正常域完全一致、2 项经参照验证的容量扩展，0 差异、0 错误**。全部证据绑定新的 native source SHA。

## 同一安装包验证

- 内部 full、无面积识别模型；四个私有含量模板保留；未使用 public 模式。
- 安装包：[fiber-diameter-measurement-setup-0.4.7-no-area-models.exe](../../build/cu-live-measurement/windows-release/final-r2/fiber-diameter-measurement-setup-0.4.7-no-area-models.exe)，398,823,789 bytes。
- SHA256：`b47ff02a1fc1fd860af7a25f941e925534ac55927d26cd68cd741bc4b504640b`。
- 用已核验的任务目录 innounp 2.67.11 解开**上述同包**；逐一核对 4768 个清单文件大小与 SHA256，自检及 PYZ 再次通过。
- 真实解包程序从空的隔离配置启动，验证主窗口 → 试验菜单 → 实时试验窗口及 11 个可见关键控件；先关闭试验窗口，再读取本次空项目的未保存对话框并选择放弃；自然退出 0，无强制清理、无残留进程。
- 未执行安装器注册/快捷方式写入流程；已有固定 AppId 安装记录前后完全一致。

## 证据及边界

完整机器可读证据见 [cu_windows_release_final_r2_20261015.json](cu_windows_release_final_r2_20261015.json)，原始文件在 `build/cu-live-windows-final-r2/`。验证工具为单独 overlay，不改 952 文件冻结清单。

真实 CU 有效设备流、Microview 板卡采集仍未验收。本报告不将源代码/受控窗口/解包运行等同于真实硬件验收，也不冒充已经完成后续 R2 正式运动/UI 性能与连续使用压测。
