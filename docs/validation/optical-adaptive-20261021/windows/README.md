# Windows 安装版验证

构建 `daa32894eef74113a27e758c76746ebe`，版本 **0.4.10**；实际安装包 SHA256 `ee7ec2619ce1252074789305f7ab4d739d5642b461cf08f93c2e89cc99ae9c8c`。运行环境为 Parallels Windows 11 ARM、x64 Python/应用，未连接 Microview。

| 检查 | 结果 | 证据 |
|---|---|---|
| Windows 源测试 | 267 通过 | `windows-tests.log` |
| 实际导航及新算法点选 | 通过；合成采集源 | `source-quick/result.json` |
| 安装包与资源 | 4,818 文件完整性及 441 生产源文件哈希核验通过；仅排除 area-models | `package/installer-identity.json` |
| 解包自检 | 新算法正例、空图、真实工作子进程及其他自检通过 | `package/extracted-selfcheck.json` |
| 实际安装 | SYSTEM 安装到候选独立目录，原安装文件不覆盖 | `actual-installation.json` |
| 普通用户自检 | 非管理员，实际安装版新算法工作进程通过 | `installed-selfcheck/result.json`、`installed-selfcheck/selfcheck.json` |
| Explorer 启动 | 隔离配置、非管理员启动，主窗口出现，正常退出 0 | `explorer-start/result.json` |
| 九组 UI 与业务 | 全部通过：空态、四主题、1366×768、1067×600、1600×900@125%、实际 Excel 导出 | `gui-summary.json` |
| Excel | 52 项输入、144 项直径、12 项计数、486 项公式核验；VBA 与打印区域保留 | 对应 `excel-verification.json` |
| 恢复测试环境 | 仅恢复发生变化的 FDM 自有注册项；原安装四个文件哈希不变，候选目录保留 | `installation-backup/restored.json` |

四主题 GUI 案例真实操作记录停靠、浮窗、相册、历史证据、缩略图与导航；项目文件前后哈希一致。已目视复核传回的 1366×768 浅色操作台与深色相册截图，未见新的文字遮挡。新光学设置窗口的四主题像素检查见上一层 UI 报告，其来源是 macOS offscreen。

临时关闭的宿主摄像头共享已恢复为开启。安装测试未运行旧版卸载程序、未回滚整个虚拟机、未改写其他应用配置。

边界：Windows 源导航测试受到虚拟显示器最大窗口尺寸约束，其通过仅用于交互/业务，不用来证明 1600×900 几何；实际安装版 GUI 案例另行检查指定客户区尺寸。1067×600 是小窗口测试，不等同于完整 150% DPI 认证。真实相机、4K、跨显示器移动以及真实成像参数变化仍待验收。
