# Git 收录与本地交付说明

2026-10-27整理。Git保存研究报告、接口与现场规程、C#收集器源码及构建脚本、独立算法参考实现、合成输入与验证输出、原厂文件身份和选定静态分析证据。压缩的REA证据与算法fixture是结论和校验的依据，纳入版本控制；没有包含原厂DLL或EXE。

以下内容通过本目录的`.gitignore`留在本地，忽略不会删除文件：

| 本地内容 | Git中保留的对应资料 |
|---|---|
| `collector/release/DsxEvidenceCollector-*/`和发布ZIP | 源码、构建/打包脚本、发布验证摘要及ZIP的`.sha256` |
| `collector/validation/run-*/`原始模拟运行树 | `collector/validation/latest/`的最终结果、构建身份与界面证据 |
| `collector/validation/windows-result.zip`和传输`.part` | 已整理的最终验证资料 |

收集器的[使用说明](collector/README.md)、[验证摘要](collector/验证摘要.md)和研究报告中的发布包链接是本地交付入口。新克隆不含这些二进制；须取得相应版本的发布ZIP并核对SHA-256，或在Windows上先执行`build.ps1`、`validate.ps1`，再按`package_release.py`所需的运行目录打包。不要把只取得源码视为已取得可现场运行的程序。

`COMMITTED-SHA256SUMS.txt`仅覆盖本次纳入Git的文件，不包含清单自身。原有各模块SHA清单、构建身份和历史验证结果保持原字节，`.gitattributes`禁用换行归一化，避免Windows编码或CRLF改变哈希。

历史`verification.json`与`identity/fdm-source.json`记录分析当时的FDM版本。此后产品已有正常提交；`tools/verify_deliverables.py`会对比当前产品源码并覆盖该历史结果，不应在整理Git时直接运行或改写基准来消除差异。算法证据可只读核对：

```sh
python3 docs/research/dsx1000-integration-2026-10-22/algorithms-2026-10-23/build_research_index.py --check
```

`run_all_references.py`会生成合成验证结果；需要重放时，在研究目录副本运行并将新结果另存，保留既有交付记录。模拟通过不等于原厂数值等价或实机到位、新帧、完整Z栈链路已经验收。
