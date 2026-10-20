# 第四轮视觉与真实导航回归

2026-10-19，macOS、Qt offscreen：**16 / 16 场景通过，生产源文件指纹一致**。版本维持 0.4.10。

使用真实 MainWindow 导航、类别编辑对话框、全图分析子进程、独立对齐检查、候选点击入账和异步证据写入。相机输入是固定的 768×576 合成纤维帧，以 35 ms 定时持续送帧；没有注入候选或测量结果，也没有绕过对齐门禁。背景任务等待由真实 Qt 事件循环驱动，避免 `QTest.qWait` 在本机 PySide 版本中占用 GIL 干扰线程耗时。

| 四种主题分别覆盖 | 完整可见类别 | 结果 |
|---|---:|---|
| 1600×900，100% | 9 | 通过，目标至少 6 类 |
| 1600×900，125% | 4 | 通过，目标至少 4 类 |
| 1600×900，150% | 1 | 控件可达，类别及右侧内容需要局部滚动 |
| 1366×768，100% | 5 | 控件可达，剩余类别局部滚动 |

尺寸为 Qt 客户区的物理栅格尺寸，不包含系统标题栏和任务栏。缩放由 `QT_SCALE_FACTOR` 实现，不替代物理显示器移动或 Windows 字体布局检查。

验证器同时记录 `actual_client_raster` 与每张截图的真实尺寸，要求与请求物理尺寸相差不超过 1 像素（逻辑尺寸取整）。原生 Windows 的系统 DPI 已为 200% 时，必须传 `--base-device-scale 2`，使工具的目标缩放不会与系统 DPR 再次相乘。macOS offscreen 保持默认值 1。

主场景执行 200 次三页真实导航，其他 15 个场景各 6 次，共 290 次。每个场景共享采集仅启动一次，两级导航位置保持不变，未发现多余顶层控件。实际截图增加源像素存在性检查，避免仅凭控件显隐或几何判断画面已正确显示。

需要展示候选的截图会等待当前显示帧通过真实核验，再立即绘制和截图，并断言有可见的 C 编号。送帧继续运行，不注入门禁或候选状态。这样避免把正常的暂不可确认状态截作候选展示图；这不证明候选在重负载下始终可见，也不提供点击可用时间比例。

每个场景保留 11 张主窗口截图及一张证据窗口截图，共 192 张原始截图；本目录精选 16 张图片（含单独光学图标注样本）和两段录像，保留全部 16 份 `result.json`、汇总及哈希清单。完整原始输出位于 `.tmp/acquisition-fourth-round-20261019/visual/`。

已检查的交互包括：空类别入口、真实分析生成 9 个候选、点击接受一条后其他候选保留稳定编号、候选透明度调整、隐藏后不可点击、手工模式继续落点、Esc 取消草稿、局部证据写入与复核、固定小窗、关闭叠加查看纯图、稳定周期自动分析一次，以及返回三页后应用偏好保持。

精选图已按主题和缩放检查：线段与编号分离、候选可对应、深浅主题文字可读、工具不重叠、保存和 Excel 导出入口常驻。150% 场景只有一类完整可见，这属于当前有限逻辑高度下的明确限制，不能表述为达到 100% / 125% 的类别数量目标。

最终补充的独立快捷键回归包含 15 个真实 Qt 交互用例：主窗口已有图片历史时，含量试验在采集中、停止后及记录浮窗内的 Ctrl+Z、Ctrl+Y、Ctrl+Shift+Z 都只操作当前试验；核验期间不改动历史，文本、数字输入和可编辑下拉框保留原生撤销重做。连同旧全屏及候选画布测试共 33 项通过；另有 94 项图片历史、比例尺和试验历史回归通过。该补充包含在本次最终矩阵的主窗口源文件指纹中。

本次最终矩阵也包含四条几何缓存、增量标注布局及慢核验时保持实时预览的回退。标注部分新增回归持续检查新线、端点、草稿、悬停优先级及删除释放空间，相关 143 项通过；局部性能对照见[标注缓存与增量布局](../annotation-cache-profile/README.md)。这些优化不放宽精确帧核验，候选不可确认时仍不能绘制或点击。

## 复查入口

- [矩阵汇总](matrix-summary.json)与[产物哈希](artifact-manifest.json)。各子目录 `result.json` 记录控件几何、源文件指纹、实际窗口事件、交互和截图哈希。
- [浅色：部分接受](light-1600x900-1/content-partial-accepted.png)、[深色](dark-1600x900-1/content-partial-accepted.png)、[深蓝](navy-1600x900-1/content-partial-accepted.png)、[暖灰](warm-1600x900-1/content-partial-accepted.png)。
- [125% 类别布局](light-1600x900-1p25/content-partial-accepted.png)、[150% 有限空间](light-1600x900-1p5/content-ready.png)、[1366×768](warm-1366x768-1/content-partial-accepted.png)。
- [固定复核小窗](light-1600x900-1/evidence-pinned.png)、[隐藏证据标注](light-1600x900-1/evidence-clean-image.png)、[独立窗口](light-1600x900-1/evidence-window.png)。
- [100% 操作录像](light-1600x900-1/candidate-evidence-navigation.mp4)、[125% 操作录像](light-1600x900-1p25/candidate-evidence-navigation.mp4)。录像为 8 fps 的 Qt 控件采样及应用所属复核窗合成，**不是桌面录屏**；不证明系统窗口堆叠、原生 SDK 窗口或快速闪窗一定无问题。窗口事件检查与录像共同用于本机 Qt 回归。
- [既有光学样本标注](optical-screenshot-fixture-labels.png)使用工作区原有样本 `.tmp/纤维直径测量测试/光学4.jpg`，不是用户本轮附带的两张截图。原图 1469×978，取 `(350, 201, 768, 576)` 中心区域、不缩放；叠加该裁块的离线传统法 3 条候选，以及人工添加的 `#12` 测量几何，仅检查标注呈现，不是本轮真实相机或完整主窗口的分析结果。原 JPEG 的 SHA-256 为 `cd2572530e5742d09675e34c69c2ad42b678680e63683f7d9d45a9e19d158699`；已与 `.tmp/analysis-plan-20261019/latency/optical4-768x576.npy` 逐像素核对。样本的原始相机来源未知。`canvas-isolated-performance.json` 是早期独立画布绘制测试，不能当作完整采集性能。

`result.json` 中的耗时只用于确认本轮真实流程可运行，单场景分析仅两次、点击仅一次，不是性能统计验收。持续采集与存档的 P50 / P95 证据见上级 `candidate-performance` 目录。

复现命令：

```sh
.venv/bin/python scripts/validation/validate_acquisition_fourth_round.py --matrix --workers 2 --cycles 200 --video
```

Windows 单场景入口：

```powershell
python scripts/validation/validate_acquisition_fourth_round.py --theme light --size 1600x900 --scale 1 --base-device-scale 2 --platform windows --cycles 200 --output <output-directory>
```

上例假定原生 Qt DPR 为 2；系统 100% 时使用默认值 1。`--video` 需要 ffmpeg。分析子进程需要操作系统共享内存权限。Windows 安装包、真实 Microview 设备、物理多显示器、离线报告导出和设备长时采集另行验收；本目录不覆盖这些项目。
