# 第五轮 Windows 候选包流程

产品版本保持 **0.4.10**。独立构建目录为 `C:\tmp\fdm-cu-live-fifth-round-20261020`，保留第四轮源码、结果、候选和用户原安装。内部包使用 `--exclude-area-models`，保留 CU 模板与 Microview 组件。

## 实际基线与隔离

2026-10-20 首次只读检查，Parallels 的 `Windows 11` 为 **stopped**。这与第四轮结束时的 running 不同，后续以本轮实查为准：测试需要启动 VM，完成后优先恢复 stopped；若期间发现新启动的用户交互或独立任务，不强制关机，单独报告。不能照搬历史状态。

本轮已按实际基线启动 VM，保留启动后发现的独立 Python 后台进程；不读取或输出其命令行。构建前将原为开启的相机共享暂时关闭，结束后恢复。阶段结果见同目录 `windows/`，本流程说明本身不代表任何验收通过。没有 Microview 实体相机；模拟图像、Windows ARM64 上的 AMD64 应用、离线实体设备结果分别记录。

## 构建及验证顺序

1. 等本机生产文件冻结后，使用 `sync_acquisition_fifth_round_windows.py` 将完整源快照和内部运行资源传到新的隔离目录，校验 SHA-256。原 Python 环境仅提供已有解释器和依赖，不覆盖其源码、配置或依赖。
2. `build_acquisition_fifth_round.ps1 -TestsOnly` 执行候选、混合输入、证据、主题、显示与 Excel 定向回归。Windows 原生 Qt 视觉脚本使用 `--base-device-scale 2`，以截图实际物理尺寸确认缩放；不把 3200 像素截图误记为 1600。
3. 同一冻结快照构建 `0.4.10-no-area-models`；安装器已存在时拒绝覆盖。`verify_fifth_round_installer.py` 提取并核验载荷、模板、Microview DLL、源清单、构建 ID 与冻结自检。
4. 只备份固定 FDM 注册项和旧安装关键文件哈希。安装到本轮 `dist/windows/installed-candidate`，参数为 `/NOICONS /TASKS=""`，不使用旧安装目录，不创建整机快照。
5. 暂时关闭并记忆 VM 相机共享，防止测试使用主机摄像头。普通用户通过 Explorer → 本轮隔离配置辅助文件 → 安装 exe 验证启动与退出。按真实入口检查四主题下混合默认、相册、证据复核、高级存档选项及常驻 Excel 导出。
6. 记录夹具使用本轮短路径，避免证据文件名和临时后缀超过 Windows MAX_PATH。只对本任务窗口及 PID 操作，不干预原有 TextileExecutionBridge 等独立后台任务，不输出敏感命令行。
7. 回传日志、截图、安装包及源/发布清单，均经 Parallels 私有接口与 SHA-256 校验。安装后仅还原本次改变、且仍与安装后状态一致的限定 FDM 注册项；不运行旧卸载器，不整机回滚。验证旧安装哈希与策略未改变，相机共享恢复，然后根据本轮实际基线结束 VM。

## 验收边界

Windows 真实 Microview 的快速移动、缓慢漂移、菜单展开、连续采集 30 分钟、物理高 DPI／跨显示器和相机原帧阈值仍需离线验收。冻结安装版没有相机时只验证真实 UI、存档与报告；不把夹具导航写成真实采集通过。
