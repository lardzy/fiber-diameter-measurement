# 中间验证记录，不作为最终验收

五个 clean 组通过布局与退出检查；legacy 组在保存前因旧探针仍按静态标签查找样品名称而停止。失败截图证实编辑框显示正确，后续探针改为严格校验“样品名称”编辑框的 UIA ValuePattern 值。

此轮自动选择了 Parallels 映射的 OpenCV 摄像头。原探针的 `real_hardware: false` 字段不准确，不应用来证明无相机访问，也不构成 Microview 硬件验收。所有归属于探针的程序进程均已退出。

该轮含摄像画面的 PNG 只保留本机审计，不随交付分发；Windows 原始目录的 PNG 也应排除交付归档。最终验收需要在不可用设备配置及 VM 摄像头隔离后另起六组运行。

Qt 实际 UIA 证据：QScrollArea 没有 ScrollPattern，局部 QScrollBar 提供 RangeValuePattern；被 viewport 裁剪的补计按钮仍可能报告 IsOffscreen=false。后续探针增加局部滚动条回退及实际边界检查，不降低可见性要求。
