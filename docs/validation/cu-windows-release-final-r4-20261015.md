# CU 实时测量 Windows 内部发行验证 · final-r4

R4 的生产差异仅为 `native/kernels.cpp` 峰值扫描中两处有限值检查改用已有严格 IEEE 位校验；计算顺序、转换和规则保持。Python、界面、PNG 与统计源码逐字节同 R3。R1/R2/R3 原目录、包、清单及 JSON 保留。

## 本轮实际验证

- 独立目录 `C:\tmp\fdm-cu-live-final-r4-20261015`，1008 个文件；源清单 SHA `f53b73e983dc723f84be2795f36c86d2dd4c40d9c1cc3d4fedb61e814d009347`，原生源 SHA `b22a7b134aadbafc1fe91e570c027f62bbdd335620d58c279f89d524eb9b44dc`。
- Windows 重编两个扩展后专项 **482 passed、145 subtests**；人工 246/246（192 个正宽度位级相等）、运动 111/111、全图 177 组（175 一致、2 容量扩展、零差异/错误）。
- 复用的全量依据明确是 **R3 的 4606 passed、1650 subtests、5 个原基线失败**，未称为 R4 新跑全量。R4 新增覆盖见本轮 XML 和金标。
- core 与 full 均通过原一键脚本 **clean=true** 构建，实际 revision 3 的 16 项冻结自检、两个扩展源摘要和 11 模块实际 PYZ 对照通过。
- 内部 full 无面积识别模型，保留 4 私有含量模板，未使用 public 模式。

## 同包交付

安装包：[fiber-diameter-measurement-setup-0.4.7-no-area-models.exe](../../build/cu-live-measurement/windows-release/final-r4/fiber-diameter-measurement-setup-0.4.7-no-area-models.exe)；398,816,084 bytes；SHA256 `6faadecbd90c94a6cb3252dc68fb629e481e6a5408e7d12d4f4fb138b9eaca9a`。

上述同包解出后逐一验证 4768 个文件大小和 SHA，再运行 16 项自检、实际 PYZ 对照和真实 GUI 主窗→试验菜单→实时入口→关键控件探针。隔离空项目自然退出 0，无强制清理、无残留。既有 AppId 安装记录保持不变；未运行注册安装/快捷方式/卸载流程。

另存的 [1280×720、200% DPI 布局图](cu-live-measurement-2026-10-15/ui-windows-1280x720-r4.png)来自本次冻结源码的合成 Qt/QTest 探针，**不是安装程序测量截图或真实采集**；[身份文件](cu-live-measurement-2026-10-15/ui-windows-1280x720-r4.identity.json)与安装产物 GUI 证据分别记录。

## 性能和设备边界

旧/新扩展同进程交替 peak 微基准已由独占窗口验证，原始两平台结果单列。其收益不能替代完整运动/UI 正式采样与一小时持续试验，后者报告另记。真实 CU 有效视频和 Microview 板卡尚待设备验收。

完整证据：[cu_windows_release_final_r4_20261015.json](cu_windows_release_final_r4_20261015.json)。验证工具是单独 overlay，不改冻结源清单。
