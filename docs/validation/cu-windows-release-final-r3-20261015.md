# CU 实时测量 Windows 内部发行验证 · final-r3

本报告描述 **final-r3 冻结包构建与正确性验证**。本轮纳入逐行 PNG 编码、工作区预算及 Save As 后证据路径重定位修正。旧 R1/R2 清单、安装包、自检和正式性能 JSON 均保留，不能替代 R3 证据；后续正式性能与持续使用验证另行记录。

## 来源与构建

- 独立 Windows 目录：`C:\tmp\fdm-cu-live-final-r3-20261015`。
- 980 文件源码清单 SHA256：`dbe309b7dfcfd4f731581c583acb5b1d4718be018c3b4e850ccf242947135ef5`；原生源码 SHA256：`4ec56fd735aa4c8edcdcfe2905bc488f3e8cc1534c2363a0d28e4cee2cfc13c8`。
- 两个 profile 均使用原有 `scripts/build_windows_onedir.py`，**clean=true**，没有禁用 PyInstaller 分析或使用缓存绕过。
- `core` 和 `full --exclude-area-models` 均通过实际冻结自检；CU 检查 revision 3，16/16，含 deferred build/apply、组距 undo/redo、GRAY/BGR/BGRA 实际流式 PNG 与 SHA/像素、全部租约释放、保存后证据路径及 undo/redo 恢复。
- 两个 onedir 和安装包解出 EXE 中的 11 个关键模块均对实际 PYZ 做递归代码对象比较，仅忽略 co_filename；字节码、常量、嵌套代码、异常/行表及参数均与本次源码一致。
- R3 环境确认 `psutil==7.2.2` 已存在，全部 distribution 版本保持原样；首次补齐见 R2 记录，旧 R1 数据不回写。

## 正确性

Windows 专项 **815 passed、73 subtests**，包含全部 5 项 Save As 真实路径回归。人工金标 **246/246**，192 个正宽度位级相等；运动金标 **111/111**；自动金标 **177 组：175 正常域完全一致、2 项经参照验证的容量扩展，0 差异、0 错误**。全部证据绑定新的 native source SHA。

macOS R3 全量为 **4606 passed、1650 subtests、5 个已复现基线失败**，不是全绿；名称与 R2 基线逐项一致。该全量结果晚于源码快照，作为验证证据 overlay 记录，不改变 980 文件源码清单。

## 同一安装包验证

- 内部 full、无面积识别模型；四个私有含量模板保留；未使用 public 模式。
- 安装包：[fiber-diameter-measurement-setup-0.4.7-no-area-models.exe](../../build/cu-live-measurement/windows-release/final-r3/fiber-diameter-measurement-setup-0.4.7-no-area-models.exe)，398,844,521 bytes。
- SHA256：`e341089d859e2db05213dba2c10b16be40150e531fa2dbb6a86381fea3533058`。
- 用已核验的任务目录 innounp 2.67.11 解开**上述同包**；逐一核对 4768 个清单文件大小与 SHA256，自检及 PYZ 再次通过。
- 真实解包程序从空的隔离配置启动，验证主窗口 → 试验菜单 → 实时试验窗口及 11 个可见关键控件；先关闭试验窗口，再读取本次空项目的未保存对话框并选择放弃；自然退出 0，无强制清理、无残留进程。
- 另用同一冻结源码运行原有 Windows Qt/QTest 布局探针，1280×720、200% DPI 下九类按钮与键盘模式均可见，两击测量/右键后选/操作者改名检查通过；截图及 JSON 位于 `build/cu-live-windows-final-r3/ui-probe-final-r3/`。这项明确属于合成来源的源码 Qt 验证，不冒充安装产物测量。可查看[1280×720 布局截图](cu-live-measurement-2026-10-15/ui-windows-1280x720-r3.png)及其[来源身份](cu-live-measurement-2026-10-15/ui-windows-1280x720-r3.identity.json)。
- 未执行安装器注册/快捷方式写入流程；已有固定 AppId 安装记录前后完全一致。

## 证据及边界

完整机器可读证据见 [cu_windows_release_final_r3_20261015.json](cu_windows_release_final_r3_20261015.json)，原始文件在 `build/cu-live-windows-final-r3/`。验证工具为单独 overlay，不改 980 文件冻结清单。

真实 CU 有效设备流、Microview 板卡采集仍未验收。本报告不将源代码/受控窗口/解包运行等同于真实硬件验收，也不冒充已经完成后续 R3 正式运动/UI 性能与连续使用压测。
