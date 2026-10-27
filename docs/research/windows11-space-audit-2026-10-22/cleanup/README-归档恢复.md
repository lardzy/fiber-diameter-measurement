# 本轮保护归档的查找与恢复说明

这份说明适用于2026-10-22至23日Windows虚拟机空间清理。归档已在宿主独立验证，包含75,073个原路径对应的4,202份唯一内容；同内容只保存一次。它用于取回本轮选择保留的源码、配置、模板、诊断和其它资料，**不是全部旧程序目录的完整备份**。

## 应一起保留的文件

| 文件 | 用途 | SHA-256 |
|---|---|---|
| [preserved.zip](preserved.zip) | 4,202份唯一内容及内部manifest；45,740,655字节 | `7fc20c93ef385888f30f0647f0e428edbefaf647883db7ed48b679b9c5c9830b` |
| [archive-mappings.json](archive-mappings.json) | 75,073个原路径到代表ZIP成员的完整映射 | `1afe705148bcc0ea79873669df63bef523bfc8c75b4345047e5dd1bc1897f9a0` |
| [plan.json](plan.json) | 32个操作目标、文件身份、归档理由、保留/跳过信息 | `aeb98bd97ebf7a210c0afda8c1ceae50da5460012496b06f52b375be56bb0876` |
| [archive-verification.json](archive-verification.json) | 宿主对ZIP内容及映射的验证结论 | 与上述三份实际文件配套保留 |

此外保留`preflight-result.json`、`finish-preflight-result.json`和`protected-before.json`，用于核对范围、基线和执行阶段。`plan.json`里的哈希只证明文件身份；没有出现在归档映射中的内容不能仅凭plan恢复。验证记录中的`all_original_paths_reconstructable`语义是“全部映射原路径内容可重建”，不是“所有待删文件都已备份”。

## 先验证文件没有损坏

在上述文件所在目录打开PowerShell，可只读核对三个核心文件：

```powershell
Get-FileHash -Algorithm SHA256 -Path .\preserved.zip, .\archive-mappings.json, .\plan.json
```

将输出与表中值逐一比较。不同则先查找正确副本，不继续恢复。ZIP能打开或CRC检查通过不能替代SHA-256核对。

## 按原路径找到实际存储的内容

1. 在JSON查看器中打开`archive-mappings.json`，在`entries`内按完整`original_full_path`查找需要的文件；也可先用`original_root`和`original_relative_path`缩小范围。原路径是Windows路径，JSON文本中的`\\`是反斜杠转义，解析后为单个反斜杠。
2. 记录该条的`archive_entry`、`length`、`sha256`。`archive_entry`才是ZIP里真正应取出的成员名，使用正斜杠和`C/...`形式。不要假定ZIP成员一定与原路径同名。
3. `representative_full_path`说明是哪一个原文件充当这份相同内容的代表。多个不同原路径可能指向同一个`archive_entry`；这是按SHA-256与长度去重的正常结果。
4. 用ZIP工具只提取该成员到新的临时目录，核对长度及SHA-256与该条映射一致。然后按`original_full_path`或选定的新位置取回所需副本；不要直接覆盖现有工程、标定或配置。
5. 如需恢复多个原路径，逐条按映射建立各自文件。直接把ZIP全部解压只会得到代表路径树，**不会自动重建全部75,073个原路径**。

实例：映射中`...\et_xmlfile-2.0.0.dist-info\INSTALLER`的两字节内容，与`...\click-8.4.1.dist-info\INSTALLER`相同；它的`archive_entry`指向后者。恢复前者时应取出后者对应ZIP成员，验证哈希后按前者的目标路径保存。不要因为归档里没有`et_xmlfile/.../INSTALLER`同名成员就判为丢失。

## 范围与限制

- ZIP有4,203个成员，其中一个是内部`manifest.json`；原路径映射以独立的`archive-mappings.json`为准，不把内部manifest误当完整路径恢复清单。
- 映射记录`original_last_write_utc_ticks`，可作为原修改时间证据；归档未承诺重建NTFS文件ID、硬链接关系、ACL、备用数据流或全部文件系统元数据。此处证明的是映射所列普通文件的内容。
- 13个重解析点只记录、未跟随，不在本次普通文件哈希覆盖内。不得利用归档路径自动追入共享目录或覆盖其目标。
- 4份有差异的runtime整树保留，2个未完成全量比较的传输ZIP也保留。它们不需要从本归档“恢复”，也不能因为存在本归档就删除。
- 另有30个模型副本（合计2,764,074,236字节）与主工作区runtime中9个保留模型文件的SHA-256一致，因此未重复归档。原路径、保留来源与哈希见`plan.json`各目标的`model_duplicates`；需要取回时从对应`retained`来源复制并核对哈希。
- 此说明没有执行恢复动作。实际删除成功/失败和最终保留结果应结合[清理结果](清理结果.md)及对应执行回执判断，不能仅从plan推断。
