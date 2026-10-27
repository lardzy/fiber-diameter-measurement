# 应用、缓存与用户配置空间分析

本报告只分析已经取回的目录和大文件元数据；没有再次访问虚拟机，没有打开原始业务文件内容，没有执行清理。扫描时刻为 2026-10-22 22:05（北京时间）。数值以 GiB=2³⁰、MiB=2²⁰ 计。

最值得进一步评估的是包管理缓存、浏览器缓存与更新下载残留。但本次没有足够证据给出“可以释放多少”的确定数字：硬链接、仍在使用的包、离线需求及工程保留要求都需要纳入。候选 JSON 的 `releasable_bytes_estimate` 均保留为 null。

## 分项结论

| 位置/类别 | 逻辑体积 | 判定与后续处理边界 |
|---|---:|---|
| Local\uv\cache | 2.221 GiB | 确为uv缓存，但与现存.venv共享硬链接，不能把全量当成可回收空间。 |
| Roaming\uv\python | 63.01 MiB | Python 3.12.13解释器，保留。 |
| Local\npm-cache | 172.37 MiB | 下载缓存，可作为将来由npm管理的候选。 |
| Roaming\npm | 191.95 MiB | 全局CLI工具，包括@openai、@moonshot-ai；不是下载缓存。 |
| GoogleUpdater\crx_cache | 0.480 GiB | 更新下载缓存；需核对更新状态，不能删整个Updater。 |
| Chrome Default Cache / Code Cache | 128.21 / 34.70 MiB | 分别可识别；不纳入密码、站点数据、账号及扩展。 |
| Edge Default Cache | 37.96 MiB | 单独缓存项，不等于整个Edge User Data。 |
| Local\Temp | 0.977 GiB | 混有工程构建与诊断材料，逐项核对；不是整目录可删。 |
| .codex | 1.070 GiB | 程序、会话、数据库、插件混合；整体保留。 |
| .docker | 352.26 MiB | 几乎都是Scout模块程序，不是镜像层清单。 |
| WindowsApps | 5.995 GiB | 已安装应用和共享运行库；只考虑将来用户选择后的正规卸载。 |
| ProgramData\Package Cache | 1.837 GiB | 安装修复载荷，不能当普通下载缓存清空。 |
| ProgramData\Microsoft\VisualStudio\Packages | 1.159 GiB | 离线开发环境的安装/维修包，优先保留。 |
| ProgramData\Zemax | 0.506 GiB | Samples、Objects、模板等应用资源，保留。 |
| Office\SolutionPackages | 649.63 MiB | 多组PackageResources，单凭目录名不能判可重建缓存；保留待归属核对。 |

表中父子目录有交叉，不能将各行直接相加。精确路径、字节数、来源及类别见 [cache-app-candidates.json](cache-app-candidates.json)。

## uv：缓存、解释器与现有虚拟环境应分开

`C:\Users\lishuyang\AppData\Local\uv` 的全部约 2274.69 MiB 都在 `cache` 下：`archive-v0` 2211.91 MiB，`simple-v22` 42.27 MiB，`simple-v20` 15.53 MiB，`builds-v0` 4.39 MiB，`wheels-v6` 0.31 MiB。另一个位置 `AppData\Roaming\uv\python` 是解释器安装，共 63.01 MiB，不属于该缓存。

大文件清单中15项uv缓存文件有11项硬链接数大于1。以下两例可在现有证据内直接按同一卷+文件ID相连：

| 文件 | 逻辑体积 | 同一文件ID的缓存外路径 |
|---|---:|---|
| NumPy OpenBLAS，ID `dcdfde70:00010000000260ff` | 19.46 MiB | `Downloads\fiber-diameter-measurement\.venv\Lib\site-packages\numpy.libs` |
| PySide6 opengl32sw.dll，ID `dcdfde70:00010000000282dd` | 19.68 MiB | `Downloads\fiber-diameter-measurement\.venv\Lib\site-packages\PySide6` |

因此，删除缓存路径并不意味着这些文件的分配空间消失；仍存在的虚拟环境硬链接会保留它们。扫描按“首次遇到的路径”归属的约2.298 GiB分配量，也不是缓存可释放量。

uv官方提供 `uv cache prune` / `uv cache clean` 管理缓存，并明确不要直接修改缓存目录；清理还影响离线重建所需的本地包。若之后授权，优先评估工具管理的修剪方式，保留解释器和现有项目环境；本轮没有运行这些命令。[uv缓存文档](https://docs.astral.sh/uv/concepts/cache/)

## 临时文件、更新下载和npm

用户Temp中可见WinGet安装载荷196.48 MiB、Excel诊断144.08 MiB、`final-entry-build-debug` 127.90 MiB，以及四个Clash Verge更新器目录合计约181.53 MiB；根目录直系文件还有219.40 MiB。应把工程构建目录交由本次工程保留规则审查，诊断目录先看是否仍有排障用途，安装残留也要确认相关安装完成。没有枚举到内容等价的证据，因此不把它们判作重复备份。

npm缓存中的 `_cacache\tmp` 约62.95 MiB，下载内容约108.91 MiB。`npm cache verify` 会校验并回收未使用数据，因此它也是会修改缓存的维护动作，不在本次只读审计中执行；整库清理仅在后续确需腾空间时考虑。全局`Roaming\npm\node_modules`应保留。[npm缓存文档](https://docs.npmjs.com/cli/v11/commands/npm-cache/)

GoogleUpdater的`crx_cache`主要由一个474.41 MiB文件构成；官方说明此位置存更新下载。其归类为缓存有依据，但本轮未确认Updater空闲或受支持的独立清理入口，所以仅列候选，不给出直接删除指令，也不把`prefs.json`、服务程序等纳入。[Chromium Updater规范](https://chromium.googlesource.com/chromium/src/+/main/docs/updater/functional_spec.md)

## .codex、.docker和应用安装材料

`.codex` 的主要组成是 `packages\standalone` 355.70 MiB、`.sandbox-bin` 293.02 MiB、`logs_2.sqlite` 273.74 MiB、`sessions` 65.07 MiB、`.tmp\plugins` 65.25 MiB。真正名为`cache`的目录只有9.27 MiB。会话历史不能作为缓存删除；数据库未读取内容，不能仅凭文件名判定可丢弃或在运行时手动压缩。两个`codex.exe`虽同为293.02 MiB，但file_id不同且未做内容哈希，不能直接宣称重复可删。

`.docker\modules\tmp`有176.19 MiB的`docker-scout.exe`，`modules\scout\backup`有176.07 MiB的另一个程序；尺寸和file_id不同。路径名暗示模块暂存/回退用途，尚不能证明不再使用。没有证据把这352.26 MiB说成Docker镜像/容器缓存，不能用`docker system prune`的预期收益解释。

WindowsApps的约5.995 GiB路径逻辑量，对应约5.001 GiB首次唯一分配归属；包含4275个重复文件ID路径。最大应用目录有Photos约1063.63 MiB、ScreenSketch约578.20 MiB、Paint约419.15 MiB、YourPhone约408.06 MiB。此列表不等于“可卸载清单”，也未核对Appx当前注册依赖；多架构/多版本WindowsAppRuntime不能凭名称手删。

Visual Studio官方说明包缓存用于无网环境下维修，禁用/移动要通过安装器与其配置管理。当前虚拟机承担构建任务，应优先保留相关载荷。ProgramData\Package Cache中最大的`v24.1.00`目录约1.054 GiB，与已安装Zemax版本相近只是线索，尚未读取MSI元数据，不能据此精确归属或删除。[Visual Studio包缓存文档](https://learn.microsoft.com/en-us/visualstudio/install/disable-or-move-the-package-cache?view=visualstudio)

## 证据与路径检查

本报告依据 `guest/transfer/scan/directories.csv`、`large-files.csv`、`summary.json`、`top100-files.json` 及环境清单；候选JSON保存这四个扫描文件的SHA-256。只使用元数据，没有读取来宾原文件内容。

不存在反斜杠传输损坏：`json.loads(summary)["root"]` 的字符码为 `[67,58,92]`；JSON Top 100的所有100条路径与CSV大文件清单精确匹配。JSON文本的 `\\` 是合法转义，工具再次把输出编码成JSON后视觉上会再多一层。候选JSON仅用 `ntpath.normpath` 生成派生路径；原始CSV/JSON没有被编辑。

下一步如需精确评估回收量，应在既定保留规则和授权下再查实际使用/安装状态及硬链接剩余归属；当前只形成候选，不执行删除。
