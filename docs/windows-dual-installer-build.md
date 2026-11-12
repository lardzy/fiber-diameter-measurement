# Windows 双安装包构建

在 Windows 仓库中运行一次：

```powershell
.\scripts\build_windows_both_installers.bat
```

BAT 使用仓库 `.venv`，从其他当前目录调用也会切到仓库；参数继续向下传递。等价 Python 命令为 `python scripts/build_windows_installer.py --both-area-variants`。

## 产物和区别

版本直接读取 `src/fdm/version.py`，脚本不增加版本。当前为 0.4.14。

| 产物 | 面积识别 | 内部报表模板 |
|---|---|---|
| `dist/installer/fiber-diameter-measurement-setup-0.4.14.exe` | 包含模型并启用 | 保留 |
| `dist/installer/fiber-diameter-measurement-setup-0.4.14-no-area-models.exe` | 不包含六个 `.pth` 模型，对应功能关闭 | 保留 |

现有两种 full profile 使用相同 Python／原生依赖。无面积模型版仍保留普通直径、图像测量、相机、数字切片和魔棒等功能所需代码与依赖。它不是 core profile，也不移除报表模板。

完整 onedir 保留在 `dist/windows/FiberDiameterMeasurement/`。第二种变体在本轮专用临时目录中派生：优先硬链接不可修改的公共文件，文件系统不支持时复制；移除面积权重，独立生成 build-id 和发布清单。校验公共文件、功能清单、模型排除及两份包内自检后，分别执行 Inno 压缩。临时变体结束时清理，不删除其它历史安装器。

每种安装器直接生成到独立文件名；旧流程先共用默认文件名再重命名，会覆盖先生成的完整安装器，本次一起修正。重新构建同版本同变体时仍更新其对应输出，其他变体及其他版本保留。

最终报告 `dist/installer/area-variants-build.json` 包含源码提交、工作区是否有未提交改动、公共程序 build-id、各输出绝对路径与 SHA-256，以及程序构建、自检、验证、变体派生和 Inno 压缩耗时。两个 `*-release-manifest.json` 保留各自的完整文件清单。第一种成功、第二种失败时报告不会声称双包成功，已成功安装器仍保留。

## 缓存和重复构建

- 默认：干净构建一次程序，然后生成两份包。
- `--both-area-variants --no-clean`：用于开发试测，保留 PyInstaller 分析缓存，由 PyInstaller 重新检查源码／资源输入；清空本轮 onedir 输出后重新收集，防止不同变体留下旧模型。发布清单标为非干净构建，自检和资源验证仍执行。
- `--both-area-variants --reuse-onedir`：只复用已经确认的完整 onedir，不重新编译。其 manifest、版本、源码提交、资源和自检必须通过现有验证；不能使用已排除面积模型的 onedir 作为公共输入。内部模式会提示 dirty 工作区，不能把复用旧 onedir 当成编译最新未提交源码。
- `--strict-release` 继续要求干净源及干净构建，不能与 `--no-clean` 组合。

单安装器原入口保持兼容。双包入口不能再传 `--exclude-area-models` 或 `--public-release`，避免含义冲突；如明确传 `--exclude-content-templates`，则两份都不含内部模板，输出分别为 `-no-content-templates` 与 `-public`。

## 耗时与验证边界

本次只调整脚本并做模拟编译器、实际文件清单及缓存行为回归，未生成新的完整安装包，也未测得优化后的实际总耗时。

2026-11-12 回归结果：**100 项通过、123 项子测试通过**。包括双包流程只调用一次程序构建／两次 Inno 编译、两种输出互不覆盖、六个面积权重移除、普通功能与模板字节保持、源 build-id 与清单不变、硬链接失败时复制回退、复用目录中新增模板保留、无效共享包拒绝，以及增量缓存保留但旧输出目录清理。日志保存在本地 `build/packaging-shared-final-unrestricted.log`。

初次扩展自检在受限执行环境中有 4 项失败，具体为测试子进程不能创建 `/psm_*` 共享内存。允许本地子进程共享内存后，相同完整测试组全部通过；没有修改产品自检或略过失败断言。

既有 Windows 虚拟机的一次无面积模型构建记录中，PyInstaller 约 421 秒，Inno 压缩约 155 秒；另一次相同 payload 的 Inno 压缩约 157 秒。这是既有单次样本，不代表用户电脑的一小时耗时分布。新流程消除第二次 Python／原生程序收集构建，两份独立安装器仍各需一次压缩；正式压缩参数没有降低。

本地日志依据：`build/template-redesign-implementation-20261111/final-v5/evidence/source/build/template-isolated-package-v5/official-build.{stdout,stderr}.log`。

回归命令：

```powershell
python -m pytest -q tests/test_build_windows_variants.py tests/test_build_windows_installer.py tests/test_build_windows_onedir.py tests/test_build_support.py tests/test_release_self_check.py
```

2026-11-12 本地结果：**100 passed、123 subtests passed**。覆盖仅构建一次程序、两次独立压缩、共享文件字节一致、模型排除、追加可编辑模板、复制回退、保留旧变体产物、缓存参数和发布自检。首次受限环境的共享内存权限导致四项自检失败；在允许共享内存的环境中原样重跑全部通过，没有跳过自检或修改其断言。原始日志保存在本地 `build/packaging-shared-final-unrestricted.log`。

实际 Windows 双包构建后，请依据报告分别检查两份安装器及包内清单，并确认无模型版不能启用面积识别、模板页与普通测量功能正常。单元测试不替代真实安装验证。
