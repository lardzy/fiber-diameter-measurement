# 维护阶段复核与浏览器缓存建议

2026-10-22。此次仅修改 `tools/Cleanup.ps1` 的maintenance段，未操作虚拟机，未执行缓存清理。浏览器容量来自此前来宾扫描，不代表当前可释放量。

## DISM状态分支已修正

| 步骤/退出码 | 行为 |
|---|---|
| before Analyze = 0 | 允许执行StartComponentCleanup。 |
| before Analyze = 3010 | 记录成功但需重启；跳过cleanup和after Analyze，原因`before_analysis_requires_restart`。 |
| before Analyze其他非0 | 记录失败；跳过后两步，原因`before_analysis_failed`。 |
| cleanup = 0 | 允许after Analyze。 |
| cleanup = 3010 | 记录成功但需重启；**不再紧接after Analyze**，原因`cleanup_requires_restart`。 |
| cleanup其他非0 | 记录失败并跳过after Analyze，原因`cleanup_failed`。 |
| after Analyze = 3010 | 记录需重启，无后续DISM调用。 |

最终结果仍分为`completed`、`completed_restart_required`和`completed_with_failures`；即使前面的uv/npm失败、DISM同时要求重启，也保留独立`failed_actions`和`restart_required`字段。无ResetBase，不发起重启。native进程的stdout/stderr仍分开落盘再合并，退出码取自Process，避免stderr被PowerShell的ErrorAction Stop误判。

静态检查确认after Analyze位于`$cleanup.exit_code -eq 0`分支内；PowerShell解析和实际Windows运行仍由根代理验证。after被跳过时没有`system-after.txt`是预期结果，收集器应读取结果JSON中的跳过原因，不因缺文件补发分析。

## 可以单独列入候选的精确缓存目录

以下8个浏览器目录互不嵌套，共222,030,279逻辑字节（211.745 MiB）；Cache已经包含Cache_Data，不能重复计量子目录。

| 精确路径 | 快照逻辑字节 |
|---|---:|
| `C:\Users\lishuyang\AppData\Local\Google\Chrome\User Data\Default\Cache` | 134439874 |
| `C:\Users\lishuyang\AppData\Local\Google\Chrome\User Data\Default\Code Cache` | 36382733 |
| `C:\Users\lishuyang\AppData\Local\Google\Chrome\User Data\Default\GPUCache` | 1642864 |
| `C:\Users\lishuyang\AppData\Local\Microsoft\Edge\User Data\Default\Cache` | 39805951 |
| `C:\Users\lishuyang\AppData\Local\Microsoft\Edge\User Data\Default\Code Cache` | 3025548 |
| `C:\Users\lishuyang\AppData\Local\Microsoft\Edge\User Data\Default\GPUCache` | 557424 |
| `C:\Users\lishuyang\AppData\Local\Microsoft\Edge\User Data\Guest Profile\Cache` | 6175741 |
| `C:\Users\lishuyang\AppData\Local\Microsoft\Edge\User Data\Guest Profile\Code Cache` | 144 |

第9个独立候选是 `C:\Program Files (x86)\Google\GoogleUpdater\crx_cache`：515,327,209字节、4文件（491.454 MiB）。Chromium官方Updater规范明确该目录保存已下载的应用更新载荷；它不同于Updater安装程序和prefs/注册状态。删除后将失去这些本地更新下载副本，今后需要时应重新下载，不是卸载Chrome。[Chromium Updater规范](https://chromium.googlesource.com/chromium/src/%2B/main/docs/updater/functional_spec.md)

9项合计737,357,488逻辑字节（703.199 MiB）。本建议不扩展到Chrome/Edge的`component_crx_cache`、浏览器整个安装目录或其他未分类目录。

## 实施建议：进程未运行才处理，状态不明就跳过

1. 使用上述精确绝对路径生成独立缓存清单，不使用`*cache*`递归搜索作为删除规则。重新枚举当前文件，拒绝目录及祖先/后代reparse point。保留缓存根目录和ACL，仅删除核实的根内缓存文件及空子目录，避免意外扩大到Profile父目录。
2. 浏览器组：检查`chrome.exe`、`msedge.exe`及其所属应用路径下的Crashpad/后台进程；无法读取进程信息时不能当成未运行。打开窗口不是充分条件，Chrome后台应用及Edge Startup Boost也算在用。出现相关进程就跳过对应浏览器组，不关闭窗口、不结束进程。
3. GoogleUpdater组：检查位于GoogleUpdater/旧Google Update安装路径下的进程，以及`updater.exe`、`GoogleUpdater.exe`、`GoogleUpdate.exe`的归属；检查相关Windows服务是否Running以及Google更新计划任务是否Running。名称随版本变化，宜同时按可执行路径判定，不依赖唯一服务名。存在运行中的Updater或Chrome则跳过crx_cache，不停服务、不禁任务、不触发更新。
4. 在每个精确目录删除前再查进程/任务状态、核对文件清单；文件在扫描后变化、锁定或访问失败时停止该项，记录`skipped_busy`或`changed_or_locked`。不得接管权限、修改ACL或用强制重试掩盖失败。任务可能自动启动，所以“先前没有进程”不是永久保证；不控制服务/任务时应保守处理这种竞态。
5. 保留浏览器配置及网站持久存储：`Local State`、`Preferences`、`Secure Preferences`、`Bookmarks`、`History`、`Login Data`、`Web Data`、`Cookies`、`Network`、`Sessions`、`Local Storage`、`Session Storage`、`IndexedDB`、`Service Worker/CacheStorage`、`Storage`、`Extensions`和所有profile父目录均不在候选内。可以记录必要配置/书签文件前后SHA，但不读取正文进入日志，也无需备份登录数据到本次研究目录。
6. 输出逐目录删除/跳过/部分失败、文件数及逻辑字节；后续核对配置基线未变、浏览器安装仍在。不要为了验证自动启动浏览器或联网。浏览器HTTP/代码/GPU缓存被清除后可能需要重新获取资源、重新编译，离线访问此前缓存页面可能受影响；用户数据保留不等于所有离线页面仍可打开。

上述持久数据排除原则与Chrome/Edge官方说明一致：缓存和Cookie、站点存储、密码、历史等是不同的数据类别；本建议仅处理明确的缓存叶目录。[Chrome数据类别说明](https://support.google.com/chrome/answer/2392709?hl=en)，[Edge数据类别说明](https://support.microsoft.com/en-us/edge/view-and-delete-browser-history-in-microsoft-edge)

本轮没有把这9项写入执行脚本。根代理可根据已获授权的清理范围决定是否增加；如果当前浏览器/updater在用，直接跳过这约0.69 GiB候选即可，不影响既有FDM清理结果。
