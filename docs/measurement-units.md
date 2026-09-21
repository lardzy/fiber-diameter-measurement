# 测量单位

## 选择单位

“图内标尺标定”和“新增／编辑标定预设”支持以下单位，默认仍为微米：

| 单位 | 符号 | 相当于多少毫米 |
| --- | --- | --- |
| 纳米 | nm | 0.000001 mm |
| 微米 | μm | 0.001 mm |
| 毫米 | mm | 1 mm |
| 厘米 | cm | 10 mm |
| 米 | m | 1000 mm |

输入标尺的真实长度，并选择与该数值对应的单位。例如，100 像素的标尺代表 500 nm 时，输入 `500` 并选择 `nm`。更换下拉框单位后，输入数字保持原值，需要按真实标尺填写。

普通图片和数字切片均使用这套标定。项目统一比例尺和后续应用的标定预设也保留所选单位。长度、直径、比例尺按该单位显示；面积自动使用平方单位，如 `nm²`、`cm²`、`m²`。显示小数位数沿用“测量与显示”中的设置。

## 保存与导出

- 项目、标定预设、比例尺 JSON、CSV 和 Excel 保留标定单位与对应数值。
- 含多种单位的项目按单位分别统计，不直接合并数值。
- 前后轮廓对比会将以上公制标定统一换算成毫米，面积为平方毫米。
- 微米在新标定的文件中沿用历史代码 `um`；旧文件的 `um`、`µm`、`μm` 均兼容，编辑预设保留原有写法。CU 标尺导入仍按其格式使用微米。
- 既有项目结构不变，无需迁移。旧预设中的其它自定义单位仍会保留，轮廓对比对未知单位继续采用未标定处理。

## 打包自检

`--self-check --json` 的 `functional_checks.measurement_units` 分别报告 `nm`、`um`、`mm`、`cm`、`m` 的长度、逆向换算与面积检查结果。任一单位失败会使 `core_measurement` 和总结果失败，现有 Windows 打包脚本会阻止自检通过。

本次新增模块通过正常 Python 导入随包收集，没有新增依赖或模型资源。

## 开发验证记录（2026-09-21，macOS）

| 回归范围 | 结果 |
| --- | --- |
| 新单位输入、项目与预设保存、长度／面积换算、CSV／Excel／比例尺 JSON、统计、CU 导入 | 185 项通过，25 项子测试通过 |
| 新单位的数字切片预览、DPR 1.5、缩放、两种文字显示模式、不同焦面原始视窗导出及未标定场景 | 22 项通过 |
| 原有标定、预设与比例尺 UI 流程 | 18 项通过，3 项子测试通过 |
| 发行自检、纳米错误换算拦截、打包配置和 Windows 构建脚本 | 83 项通过，94 项子测试通过 |

可重现命令：

```bash
uv run --no-sync pytest tests/test_length_units.py tests/test_models_project_io.py tests/test_export_service.py tests/test_measurement_statistics.py tests/test_capture_and_cu_scale.py -q
uv run --no-sync pytest tests/test_digital_slide_annotation_display.py -q -k 'calibrated or native_export'
uv run --no-sync pytest tests/test_ui_canvas_and_export.py -q -k 'calibration or preset or scale_overlay'
uv run --no-sync pytest tests/test_release_self_check.py tests/test_build_support.py tests/test_build_windows_onedir.py tests/test_build_windows_installer.py -q
```

轮廓对比来源读取已覆盖五种单位、两种旧微米符号及未知自定义单位，并检查关闭来源图片后仍按冻结的标定读取。已有数字切片预览模式矩阵也通过回归。

以上为开发环境检查；Windows 实际打包、安装后运行尚待 Windows 机器验证。
