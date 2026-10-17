# 实际安装版最终 GUI 验收：6 / 6 通过

本轮使用普通当前用户启动实际安装到隔离路径的 **0.4.10** 程序。VM 自动摄像头共享由主任务临时关闭；六组各自使用独立配置，固定 `selected_capture_device_id=microview:0`。不构成 Microview 真机采集、算法精度或吞吐验收。

- 安装路径：`C:\tmp\fdm-cu-live-second-round-20261017\dist\windows\installed-candidate\FiberDiameterMeasurement.exe`
- Build ID：`d10251c2976d4018a92123a211cfe13f`
- 安装器 SHA256：`81efe93e5ba9fad47dd189cf6269d9c2c28242a4972bfb5782c743442c1f7840`
- 探针 SHA256：`83b6555ebaea5cfcaae577f5b0246bacdf6ec8e615e5114da7ffddf0afc461c1`
- 完整汇总：[summary.json](summary.json)。每组 stdout 同目录；Windows 图像、UIA 树和项目副本位于 `build\second-round\frozen-gui\20261017-031203Z-1cb25c82`。

| 组 | 客户区 | 结果 |
|---|---:|---|
| dark / light / navy / warm，各一组新项目 | 1366 × 690 | 4 / 4 通过 |
| light，新项目 | 1280 × 670 | 通过 |
| dark，0.4.9 旧项目副本 | 1366 × 690 | 通过 |

验证包含主入口重复点击 20 次、含量/直径/实时采集切换、侧栏操作/记录/报告切换时固定样品字段的位置、常用控件实际可达性，以及空项目不误报保存。195 个客户区尺寸检查均保持请求尺寸；OS DPI 为 192，探针 `QT_SCALE_FACTOR=0.5`，不是跨显示器 DPI 验收。

旧项目从补计基线 40 开始，先读取“样品名称”编辑框的 ValuePattern，精确确认 `save-lifecycle-probe`，再经同一侧栏的竖向滚动条 RangeValuePattern 滚动到补计按钮。每次调用前检查按钮完整处于局部 viewport 内；20 轮逐次核对磁盘补计数为 41–60，并包含重复保存点击。旧项目源文件没有修改。

六个探针主进程 PID 为 7064、1696、1636、6356、11464、7652；结果均为 `graceful_exit=true`、`forced_cleanup=false`、`process_exited_after_cleanup=true`。验收结束后 GUI 已交还主任务做 Explorer 启动与快照恢复。

前一轮 `20261017-025842Z-36027492` 是中间验证：曾自动访问 VM 映射 OpenCV 摄像头，且旧名称查找断言过时。其相机 PNG 不进入交付；最终图像只使用本目录对应的 Windows run。验证脚本的更改不涉及生产源码。
