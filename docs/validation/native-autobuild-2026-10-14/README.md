# 打包前原生扩展自动重建验证（2026-10-14）

本次在 `56da75c` 基础上，将独立进程检查、按需编译、编译后复检接入原有 onedir 构建入口。默认安装器调用同一入口；`.bat` 和现有命令参数不变。

## 验证结果

| 检查 | macOS ARM64 | Parallels Windows 11 / x64 Python 3.12.13 |
| --- | --- | --- |
| 打包、安装器、发布自检及新增自动构建回归 | 110 tests、123 subtests 通过 | 110 tests、123 subtests 通过 |
| 缺失扩展后真实编译 | 通过 | 通过 |
| 有效扩展跳过编译，文件摘要和修改时间不变 | 通过 | 通过 |
| 修改 C++ 源码后自动重建 | 通过 | 通过 |
| 仅存在旧 ABI 扩展时自动重建 | 通过 | 通过 |
| 损坏扩展后自动重建 | 通过 | 通过 |
| 缺少 Python 构建工具时临时安装并编译 | 通过 | 通过 |
| 打包主进程没有导入扩展，egg-info 字节不变 | 通过 | 通过 |

回归还覆盖编译器失败、超时、DLL 被占用、编译后及替换后复检失败；这些情况下返回失败，并保护原扩展／既有打包目录。macOS 的缺工具用例使用单独的 Python 环境，结束后该运行环境仍未安装 setuptools 和 pybind11。

Windows 在独立检出副本中，先移开该副本的扩展，再直接执行原有入口：

```powershell
scripts\build_windows_onedir.bat --profile core --exclude-content-templates --strict-asset-hashes
```

该命令完成了临时构建依赖安装、MSVC 编译、原生复检、三个 PyInstaller 可执行文件的生成及打包程序自检。最终报告为 `ok=true`、`errors=[]`、`warnings=[]`；`functional_checks.native_kernels` 四项均为 true。

本次产物位于虚拟机的：

```text
C:\Users\lishuyang\Downloads\fiber-diameter-measurement\build\native-autobuild-validation\packaging-source\dist\windows\FiberDiameterMeasurement
```

原有发布目录未被覆盖。此次没有重新制作或安装 full/public 安装器；安装器默认调用 onedir 的路径和复用已有目录的自检路径已纳入上述回归。`--reuse-onedir` 继续验证已有冻结包，不在其中单独替换扩展。

## 环境与证据边界

初次 Windows 回归的失败来自 `offscreen` 模式找不到字体，以及检出资源的 CRLF 换行与固定清单不符。正式复验使用 Windows 默认图形平台，仅在独立验证副本中，将 27 个资源恢复为与清单大小、SHA256 完全一致的字节。没有放宽资源校验或测试断言，也没有更改原工作区资源和全局 Git 换行配置。

完整运行日志、源码摘要、两平台五种状态的结果、缺工具结果和打包程序自检保存在本机 `build/native-autobuild-validation/`；Windows 原始记录保存在虚拟机对应目录。这些构建行为验证不替代原生计算性能验收、真实样本计量或新一轮正式发行验收。
