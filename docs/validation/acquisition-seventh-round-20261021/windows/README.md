# Windows 第七轮验证

版本 0.4.10；内部构建排除面积模型，保留报告模板和 Microview 运行资源。

先行包构建 ID：`0b4a03782cd94d1dbc23225ad5ace87f`。
安装器 SHA256：`3298883840527ba419a2c32dfb3db1fab11923078e6f5c5d81862799c52dce59`，398,842,744 字节。

## 交付前快检

| 检查 | 结果 | 证据 |
|---|---|---|
| Windows 选择性单元回归 | 52 passed，12.06 s | `quick/windows-tests.log`、`quick/test-selection.json` |
| 真实导航、进程分析、3 次点选、3 张 JPEG、换场保留 | 通过；同批余项未隐藏，独立对齐任务计数 0 | `quick/source-quick/result.json` |
| 安装器解包及完整性 | 4,818 文件校验；436 产品及运行资源源文件匹配 | `quick/package/installer-identity.json` |
| 解包程序自检 | 通过；独立诊断探针保留，实时候选不启用它 | `quick/package/extracted-selfcheck.json` |
| 隔离实际安装 | 通过；SYSTEM 安装到候选专属目录，无桌面快捷方式，不覆盖原安装文件 | `quick/actual-installation.json` |
| 普通用户安装版自检 | `B1EA\\lishuyang`，非管理员，退出成功 | `quick/installed-selfcheck/result.json` |
| Explorer 启动 | Explorer → 隔离配置 cmd → 安装程序；非管理员，主窗口显示，正常退出 0 | `quick/explorer-start/result.json` |
| 回传与源快照核对 | 安装器/证据 ZIP SHA256 相符；436 产品及资源源文件与当前工作区一致 | `dist/internal/acquisition-seventh-round-20261021/candidate-manifest.json` |

快检脚本版本 `1c4e106ef4a2c6cab6b39d888e6e2365688f007a7e8e637f156b2d58825864ae`。快检为 3 秒静态输入，不替代后续 60 秒静态、100 次连续点选或实体相机验收。安装包来自未提交快照，`source_commit=unknown`；Git 基线 `6278af2bb86f70f880332bfa84342a6436783813` 只用于追溯。

## 环境与边界

本轮实测开始时 Parallels Windows 11 **正在运行**，相机共享 **on**；不是此前记录的停机状态。原应用为用户 Downloads 中的 0.4.7，相关注册信息及原文件哈希先备份后安装候选。完整回归结束后恢复所改注册信息并保留候选隔离目录，不进行整机回滚。环境恢复证据另存。

本轮没有连接实体 Microview。Windows 虚拟机、模拟图像和独立自检不代表真实相机帧率、稳定判定或现场交互验收通过。

交付时先行快检通过，随后 Windows 计划内功能、安装版、Excel 及性能回归完成。详见 [交付后回归](FULL_REGRESSION.md)，包含验证器首轮失败、虚拟机资源不足、补跑及最后无遮挡截图。快检 manifest 和原安装器保持交付时原样。实体 Microview 仍待验收。
