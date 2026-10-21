# Windows 虚拟机验证

Parallels Windows 11，768×576 合成相机输入；没有 Microview。源码验证使用实际 Qt Windows 平台、导航、全图分析子进程、严格对齐、画布事件、账本与 JPEG。Qt 注入鼠标事件不是实体鼠标操作。

## 连续候选点选：入账正确，严格无隐藏未通过

正式轻量追踪关闭逐方法、逐帧和指针快照，没有视频或 cProfile，运行期间没有其他重任务。初始截图后等待候选真实恢复，才开始连续点选。

- 12 个批次，100 条接受、100 张 JPEG 保存，产品源码未变化。
- 第 42 次点击附近发生一次 335.494 ms 的剩余候选暂停，原批次及剩余 3 条候选保留。
- 主线程 generation 2 GC 耗时 **26.281 ms**；对应核验完成耗时 **40.916 ms**，超过原预算 **27.962 ms**，GUI 消费延迟约 **3.479 ms**。
- 这次是实际就绪超预算后的安全暂停，不能算“正常负载无隐藏”通过。没有修改产品 GC 策略、核验阈值或恢复要求。

见 [轻量结果](source-clicks100-light.json) 与 [归因摘要](source-clicks100-light-summary.json)。脚本哈希 `a1961d4d44c77570453aaa15dcc85b599bf9ae5adf2f232d823514ed9d2ccc72`；原始结果 SHA256 `b19804653bda3eeb15f93264fa22d05b566a720a2bcd4cc2715ba9dd7304fd69`。

四次早期失败与逐步诊断原始 JSON 收在 [诊断归档](source-clicks-diagnostics.zip)，[索引](source-clicks-diagnostic-index.json) 记录各次脚本哈希及隐藏事件。初期截图累计阻塞约 157.85 ms，独立解释了一个设置阶段暂停；不能据此抹去随后仍出现的业务阶段 GC 暂停。第一次旧脚本未在断言前保存全部暂停详情，归因依赖后续可复查诊断，不倒推为已明确。

## 同机性能对照

两组依次运行，各 30 秒负载加 3 秒尾部恢复，33 ms 输入间隔、每秒一条实测，局部 JPEG 与相册持续打开。固定候选样例用于测试 Qt、严格核验、入账和存档；不包含真实全图算法计算，也不属于相机验收。

| 指标 | 基线 | 候选核验 |
|---|---:|---:|
| FPS | 30.2636 | 30.3026 |
| P95 帧龄 | 9.142 ms | 19.096 ms |
| P95 增量 | — | 9.953 ms（小于 33 ms） |

FPS 和帧龄门槛通过；不能将其作为连续点击“无隐藏”的替代指标。完整数据见 [基线](performance-final-baseline.json)、[核验](performance-final-guard.json)、[比较](performance-final-comparison.json)。

虚拟机中此 benchmark 快照早于新增 JSON 源码指纹字段，原始结果没有该字段。两组使用同一个未变更的隔离源码，后续构建校验对照源码清单逐项核对产品文件；不宣称这两份结果各自含有执行前后源码哈希。

## 测试与安装版边界

首轮选定源码回归 911 项及 50 子测试通过。新增记录发布测试在 Windows 粗粒度时钟下暴露夹具假设：多条记录创建时间相同，产品按既定的时间及稳定 ID 规则排序，夹具却要求插入顺序。只修改夹具为递增 1 ms，并新增同时间戳排序回归；定向最终 7 项通过，旧 5 失败日志保留。

首点后静置／1,000 次无按键移动来自源码 Qt 事件回归。安装版 UIA 检查主要覆盖真实窗口导航、菜单、默认混合选项、存档设置、相册与 Excel 导出，不宣称其完整覆盖 1,000 次原生鼠标移动。

## 已安装候选版

构建 ID `862b9fc24cf248af817271cb159fbbbb`，版本仍为 `0.4.10`；候选包路径和完整哈希见[总报告](../README.md)。

- [实际安装](actual-installation.json)：限定目录静默安装成功，安装需要管理上下文；与下述普通用户运行验证分别记录。
- [普通用户自检](installed-selfcheck-result.json)：`elevated=false`，实际子进程校验同图、噪声、漂移、引用更新及关闭均通过，已安装可执行文件与校验过的载荷一致。
- [Explorer 启动](explorer-result.json)：普通用户，实际父进程链为 Explorer → 隔离 cmd 帮助程序 → 已安装应用；正常关闭、未强制清理。
- [9 组原生 UIA](installed-gui-summary.json)：空白浅色、四主题有记录、1366×768、1067×600、1600×900@125%、Excel 常驻入口导出，全部通过。源 Qt 测试和真实 Windows UIA 不混为同一类型。
- [原注册状态恢复](registration-restored.json)与[虚拟机恢复](vm-after.json)：均通过，原应用文件哈希保持，虚拟机已优雅停止、自动共享相机恢复开启。

最初普通用户自检的启动器将短 PowerShell 文件转为 `-Command`，Parallels 丢失引号导致解析失败，应用当时根本未启动，见[原始启动器失败](launcher-quoting-failure.log)。第六轮适配器已改为可信文件始终落盘并通过 `-File` 执行；同一已安装应用随后自检与 Explorer 验证通过，没有重构建产品。原第五轮脚本未修改。另一次证据 ZIP 工具缺少显式加载 .NET 压缩程序集，补齐后归档成功；此为一次性归档脚本问题，不是产品失败。

## 已有截图复核

[含量试验及相册图标入口](screenshots/03-content.png)、[高级存档设置](screenshots/03a-advanced-storage-options.png)、[原生相册窗口](screenshots/04a-evidence-album-window-native.png)、[相册最大化](screenshots/04b-evidence-album-window-maximized-native.png)、[历史证据复核](screenshots/04c-evidence-review.png)。

截图复用了名为“第五轮视觉验证”的测试样品；这是夹具样品名称，实际运行身份为上述第六轮构建。没有为更换样品标题重跑验证。图标与菜单箭头没有重叠，常驻保存／Excel 和类别区可见，相册卡片文字可读。其余原始截图、UIA 树和日志保留在 Windows 完整证据 ZIP，避免向报告目录重复展开数百文件。
