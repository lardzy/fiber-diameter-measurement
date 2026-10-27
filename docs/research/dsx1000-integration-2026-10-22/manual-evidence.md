# 随附手册与原厂副本身份

日期：2026-10-22。只读检查；没有运行原厂应用、安装器或设备命令。

## 软件副本

- 根目录共清点 3,037 个普通文件，排除 AppleDouble `._*` 与符号链接。
- `OLYMPUS/DSX-BSW/` 是本轮主体，共 527 个文件，全部计算完整 SHA-256。
- `OLYMPUS/LEXT-OLS50-SW/` 属于另一产品的分析软件，不作为 DSX 控制证据。
- `backup/DSX10-BSW_Ver010204/` 是备份发行材料；它还含 LEXT 分析组件、显卡和运行库安装器，不能将每个文件都当作 DSX 设备控制组件。
- `OlympusMicro/Log/` 中本次只有日志工具 EXE/DLL 和配置，没有可用于核验运动时序的现场采集日志。
- 另对备份中三份名称含 DSX 的 MSI 计算完整哈希。共 530 份全文件摘要；其余备份文件只有路径和大小清单，不声称完成内容核验。

详见 [清单](identity/inventory.json)、[CSV](identity/file-identities.csv)、[REA 目录概览](identity/rea-directory-summary.json)。REA 目录摘要的身份范围是已读取的元数据与字节区间，不等同于所有文件的完整内容摘要；后者以清单的 `hash_scope=whole-file` 为准。

## 主程序身份

| 程序 | 程序集版本 | 声明运行库 | 托管架构 |
|---|---|---|---|
| DSX-BSW-CTRL | 1.2.4.1217 | .NET Framework 4.6.2 | AnyCPU |
| DSX-BSW-APP | 1.2.4.44 | .NET Framework 4.7.2 | AnyCPU |

完整哈希、MVID、CLI 标志和证据保存在 [软件版本记录](identity/software-versions.json)。PE 外层标记 x86 不等于程序集只能运行 32 位；该副本另含 x64 原生库，实际进程与全部依赖位数仍需设备端核对。

- CTRL 身份：`ev_cfca8102d146a1e8f664cfd22185767d43a61489a3b2aeccfc46212e5955e500`。
- APP 身份：`ev_c874ac163fe1d8050a84f597da6093e7ea9983f085ba05d21ead3abc38182abd`。
- REA 目录概览父证据：`ev_8377653c45d854ad75f894205a27039426cdb9a9c67c7352158f05635bef5d01`。

以上是所给副本身份，尚未与正在使用的设备电脑逐项比对。

## 手册阅读

原件：`OLYMPUS/DSX-BSW/Common/Help/DSX1000-Application-M-E.pdf`，完整文件哈希见清单。文本提取使用本机 Poppler；PDF 第 122 页另渲染并目视核对。下列页码同时列出物理 PDF 页和纸面页，避免封面偏移。

| 位置 | 手册事实 | 对接入分析的意义 |
|---|---|---|
| PDF 119 / 纸面 112，9.3.1 | 3D 图由沿 Z 移动时取得的多张图生成；采集会先移动到范围下限 | 原厂已具有 Z 扫描编排；不能假设以当前 Z 为不变起点 |
| PDF 120 / 纸面 113 | 扫描范围由 Pitch × Steps 形成，文档描述最多 1,000 张，Pitch 与物镜/倍率相关 | 需要保留该版本参数约束；文档上限不是本次已验证的内存/性能能力 |
| PDF 122 / 纸面 115 | 采集期间不能从应用操作样品台，也不要手动或通过手柄改变位置 | 原厂采集是具有控制权边界的过程；FDM 不应并行发另一套 XY/Z 命令 |
| PDF 122 / 纸面 115 | 快速与精细模式在低对比接缝和锐边失焦上可能不同 | 不能仅凭模式名用最快模式替代保留质量的逐层采集 |
| PDF 122 / 纸面 115 | 中止采集后，会用已经取得的图生成并显示 3D 图；可设置完成后恢复 Live | 原厂“生成部分合成结果”不等于已经提供每层原图给外部客户端 |

手册说明的是原厂界面能力。它没有在本次所选章节证明外部 API、全部原始焦层导出或电机物理到位语义；这些结论以代码证据与后续现场验证分别处理。

## 可复现的身份清点

在仓库根目录运行（仅向 `--output` 写分析产物）：

```sh
python docs/research/dsx1000-integration-2026-10-22/tools/build_inventory.py \
  --root '/Volumes/LARD/2.中转/OLYMPUS' \
  --output docs/research/dsx1000-integration-2026-10-22/identity
```

不复制或再分发原厂 DLL、安装器和完整手册；报告保留哈希、必要代码证据和分析结论。
