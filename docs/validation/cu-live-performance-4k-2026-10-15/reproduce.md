# 复现本轮检查

Windows 使用独立副本 `C:\tmp\fdm-cu-live-opt4k-20261015`，借用原项目已安装的 Python 环境；没有改动 Downloads 项目的源码或 Git 状态。最终修改的关键文件以 `optimized-source.json` 为准，原生接口与源码摘要在每份性能 JSON 中记录。依赖版本固定为报告记录值，不通过升级依赖改善对照结果。

在 PowerShell 中设置本次副本和 Windows Qt。以下新结果写入 `build/recheck-4k`，不覆盖正式原始数据。性能命令必须逐条运行，并等待主机／虚拟机的其它测试和构建结束。

验收期间保持主机和虚拟机显示唤醒，并记录显示比例、窗口是否前台／最小化。2026-10-15 的主机息屏期间，本次 VM 绘制速度降到约 9 FPS，亮屏后明显回升；这支持显示状态与性能相关，尚未确认内部节流机制。息屏轮次必须独立标注，不能与正常交互的采样直接混算。已运行的长程主机使用 `caffeinate -di -t 4500` 绑定到验证命令，Windows 验证调度线程使用 `SetThreadExecutionState(0x80000003)`，在 `finally` 中以 `0x80000000` 释放。二者是临时测试条件，不修改电源设置或产品行为；用户主动锁屏、合盖或休眠仍需另行记录，不能假定断言阻止了这些操作。

用户随后将实际验收规模调整为通常不超过 2,000 根，详情见 [范围调整](scope-update-2000.md)。下面显式采用十分钟、至少 2,000 条目标；原工具的一小时默认值仍可用于主动选择的扩大测试，不再是此次交付门槛。

```powershell
Set-Location 'C:\tmp\fdm-cu-live-opt4k-20261015'
$fdmPython = 'C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe'
$env:PYTHONPATH = "$PWD\src;$PWD;$PWD\validation_vendor"
$env:PYTHONUTF8 = '1'
$env:QT_QPA_PLATFORM = 'windows'
& $fdmPython scripts/benchmark_cu_live_ui.py --sizes 2048 4096 --motion-input stable --output build/recheck-4k/ui-stable.json
& $fdmPython scripts/benchmark_cu_live_ui.py --sizes 2048 4096 --motion-input lsb-noise --output build/recheck-4k/ui-dynamic.json
& $fdmPython scripts/benchmark_cu_live_ui.py --sizes 2048 4096 --motion-input stable --background-automatic multiscale --output build/recheck-4k/ui-joint-stable.json
& $fdmPython scripts/benchmark_cu_live_ui.py --sizes 2048 4096 --motion-input lsb-noise --background-automatic multiscale --output build/recheck-4k/ui-joint-dynamic.json
& $fdmPython scripts/validation/benchmark_cu_png_stream_pair.py --reference build/opt4k/reference/png_stream_r4.py --sizes 2048 4096 --output build/recheck-4k/png-pair.json
& $fdmPython scripts/validation/soak_cu_workspace.py --size 4096 --duration-seconds 600 --minimum-accepted 2000 --stop-file build/recheck-4k/stop.request --output build/recheck-4k/workspace-2000.json
```

短程默认 5 次预热、30 次正式尝试，并各跑原图关闭／开启。噪声短程使用两个固定来源图交替，再加入逐帧唯一标记，防止运动缓存误把不同帧当成同图；不是持续随机数生成性能测试。

持续检查使用两种固定像素场景，每十秒移动，原图开启、每秒最多 4 次手势。它覆盖真实 Qt、换场与账本增长，不等同于连续传感器噪声测试。每个原图文件独立核对 SHA256，重复载荷只解码一次，并逐条核对场景归属。`passed=true` 仅表示脚本完整性检查通过，必须另读 GUI／帧率／清场指标及独立审计。

需要提前停止时，在另一终端创建指定 `stop.request` 文件；脚本最多约一秒后结束采样，完成排空、证据核验和账本保存。未满足完整时长时仍以失败退出，并在 `counters.stop_requested` 中记录原因。重新运行请使用新的空输出目录和不存在的停止请求文件。

原有构建入口继续可用，缺失／过期原生扩展由既有构建检查自动重建：

```powershell
& $fdmPython scripts/build_windows_onedir.py --profile core
& $fdmPython scripts/build_windows_onedir.py --profile full --exclude-area-models
& $fdmPython scripts/build_windows_installer.py --reuse-onedir --exclude-area-models
```

重建 full 会替换同一个 onedir 输出；需要保留 core 时先复制到另一个目录。不要用安装器输出路径或旧自检 JSON 代替包内实际代码核验。先前 0.4.7 核对 12 个嵌入 PYZ 模块，最终 0.4.8 加入 `fdm.version` 后核对 13 个模块与最终源文件，另记录原生构建摘要及 17 项实时功能自检。没有注册式安装、覆盖升级或卸载用户现有安装。

恢复前一实现时，应在独立检出目录使用 `5049422` 并重建原生扩展与安装包。不要在保留用户暂存工作的当前工作区直接 reset，也不要把旧 PNG 模块单独覆盖到新冻结包中。已有 PNG 仍是标准无损 PNG；算法、统计和项目格式没有因为本轮性能改动而迁移。
