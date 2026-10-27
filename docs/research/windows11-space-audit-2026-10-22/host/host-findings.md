# Windows 11 虚拟机：宿主侧空间盘点

盘点时间：2026-10-22T13:52:40.513966+00:00（UTC，客户端时区为 Asia/Shanghai）。范围仅 `/Users/lishuyang/Parallels/Windows 11.pvm`，没有遍历其它虚拟机。只读检查文件元数据、选定 XML 字段和快照列表；未删除、压缩或改配置，也未由本代理启停虚拟机。

## 占用结论

停止状态首次 `du` 为 **124.02 GiB**。主代理随后启动 Windows 做来宾扫描，后续元数据盘点约 **124.04 GiB**。这是非原子、运行中会变化的观测，十几 MiB 的变化不是新的独立大文件。

| 部分 | 实际分配 | 逻辑长度 | 解释 |
|---|---:|---:|---|
| 唯一基础 `.hds` | 126999.00 MiB | 126999.00 MiB | 整个 Windows 文件系统容器，不能当缓存删除 |
| 4份日志 | 14.09 MiB | 12.65 MiB | 单独清理收益很小；本次先保留诊断证据 |
| 运行期 `.mem` / `.mem.sh` | 3.00 MiB | 64.00 MiB | 启动后出现；不是已确认可删的挂起镜像 |
| 配置、NVRAM、VM.app等 | 0.66 MiB | 0.63 MiB | 启动/配置资料，不列为清理候选 |
| 磁盘描述及辅助文件 | 0.03 MiB | 0.02 MiB | 磁盘必要结构，不按大小判断可删 |

基础磁盘占分配字节的 **99.986%**。`.pvm`、`harddisk.hdd` 和 `.hds` 是父子包含关系，不能把 Finder 或 du 的三行相加。表中分配量是文件系统报告的 `st_blocks × 512` / du，不是 APFS 克隆共享块的独占空间计算。

## 容量、快照与挂起状态

- DiskDescriptor 的 `Disk_size=536870912`、逻辑扇区512字节，对应 **256 GiB** 虚拟容量。当前约124 GiB宿主占用与256 GiB容量是不同指标；不能把两者之差当作可回收量。
- descriptor 只有一个 `Image Type=Compressed`，对应唯一 `.hds`；唯一 Shot 的 ParentGUID 全零，是基础链节点。`Snapshots.xml`为空，`prlctl snapshot-list`也只有表头。**未发现可通过删除旧快照回收的大额空间。**
- `Autoprotect.Enabled=0`，`TotalSnapshots=3`为配置策略值，不是已有3个快照；LinkedSnapshotUuid为空。
- 停止状态初次列表没有挂起内存大文件。后续主代理启动后出现 `.mem` 0字节和 `.mem.sh` 64 MiB逻辑长度/3 MiB实际分配，不应在运行时手工删除。
- 全局 `AutoCompress.Enabled=0`，磁盘字段 `AutoCompressEnabled=1`、`OnlineCompactMode=1`，另有 `ReclaimDiskSpace=0`。本次不改变它们，不能只据这些字段断言后台压缩已执行或将回收多少空间。

## 哪些可删、哪些不能直接删

- **宿主 `.pvm` 内没有发现值得优先清理的独立大安装包、旧快照或挂起内存镜像。** 大头都在 Windows 虚拟磁盘里，清理优先级应以主代理的来宾目录扫描为准。
- 4份日志合计仅 **14.09 MiB** 实际分配，其中旧压缩 `parallels.log.1.gz` 约1.68 MiB；即使都可丢弃诊断历史，收益也不足0.02 GiB。没有为这点收益执行删除。
- `.hds` 是包含系统、应用、项目和数据的磁盘本体；不能删除或裁短。DiskDescriptor、`.drh`、NVRAM、配置文件、VM.app也不作为手工清理候选。
- 来宾 Windows 删除临时文件、无用安装包或缓存，先释放的是来宾卷的空间；宿主磁盘文件不保证立即缩小。Parallels 提供独立的 Reclaim/磁盘回收机制，界面会显示可回收空间。实际回收量需清理后重新观测，不能承诺与删除逻辑字节一比一。参见 [Parallels 官方回收说明](https://kb.parallels.com/en/114654) 和 [官方 Mac 空间管理说明](https://kb.parallels.com/en/123553)。
- 后续如获授权清理，建议分别记录“来宾可用空间增加”和“.pvm宿主分配量减少”，并保留清理前后时间点；回收动作本轮未执行。

## 证据

- [host-inventory.json](host-inventory.json)：24个文件的逻辑/分配量、分类汇总、选定配置、磁盘链、快照命令结果及源XML身份。
- [du.txt](du.txt)：后续宿主层级du原始输出。
- 本地CLI包装器在sandbox内因ps限制失败，直接二进制第一次因服务访问限制失败；获只读权限后快照查询成功。不按错误提示重启、修复或重初始化 Parallels。
- 运行中盘点不能当作关机后的最终尺寸，完整清理候选需与来宾扫描合并。

## 单独列示：宿主额外 Windows ARM 安装介质

仅检查已在本机光驱配置中引用的 `/Users/lishuyang/Downloads/26100.4349.250607-1500.ge_release_svc_refresh_CLIENTCONSUMER_RET_A64FRE_zh-cn.iso`，未扩展扫描下载目录。该路径当前不存在，没有可计算的介质占用。

当前 VM 的对应 CdRom 配置仍引用这一完整路径，`Connected=0`，即已断开；精确路径 stat 返回 FileNotFoundError（errno=2）。这是**已断开光驱的残留路径引用，当前没有该 ISO 可删，也没有额外可回收容量**。若以后恢复该路径的安装介质，它属于 `.pvm` 之外的宿主文件，应单列占用并在保留其它可用恢复介质后才考虑清理，不并入虚拟磁盘回收量。本轮未删除或修改配置或介质。
