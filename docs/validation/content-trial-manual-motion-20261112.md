# 含量试验：视场变化时保留手动测量起点

日期：2026-11-12。软件版本保持 0.4.14。

## 问题与修复范围

含量试验中先点击第一点，准备点击第二点时，如果视场检测判定画面移动，原先会结束旧视场并清空所有未完成任务。控制器中的首点、画布手势和工作区的失败处理都会参与取消，导致用户必须重新取点。

本次只对含量试验的自动运动检测保留未完成的手动取点。原视场的候选、已提交计算、待确认结果和旧测量标记仍按原规则失效；直径试验的现有处理不变。

## 用户操作与帧语义

- 两点方式：第一点后发生视场变化，起点保持原图像坐标，预览继续更新。画面稳定后可以继续点击第二点。
- 仍在变化或新视场尚未绘制时点击第二点，不提交结果，也不清空第一点，显示等待提示。恢复稳定后提示继续落下终点。
- 拖动方式：已开始的拖动可跨过视场变化，稳定后松手完成。若不稳定时已松手，保留第一点，恢复后再点终点完成。
- 不自动冻结画面，也不把第一点当作跟踪目标。第一点保留的是图像坐标，没有估算样品的物理位移。
- 只有跨过自动视场变化的测量，才在完成时使用当前已实际绘制的帧。手动计算、贴边计算和图片留存都使用这同一帧；没有跨过视场变化的测量继续使用原有首点帧规则。
- 测量身份、编号及首点时确定的图片策略保留。Esc、主动新视场、切换来源／尺寸／标定／操作员／方向，以及设备断开仍取消旧操作。

## 实现

1. 控制器在 `MOTION` 边界区分保留的手动草稿与需要失效的后台工作。仅更新草稿及其手工指针的视场代次，不更新旧计算任务的代次。
2. 画布保留对应的首点、预览和已开始拖动的指针引用；稳定等待单独反馈，不走普通拒绝终点后的取消分支。
3. 完成时验证当前视场、已绘制帧及来源上下文。既有入账屏障、候选核验和迟到结果校验保持不变。
4. 帧引用继续由原预算管理，反复变化不累积草稿；关闭时释放全部引用。

## 验证结果

环境：macOS，本项目 `.venv`，Qt offscreen；实际发送 Qt 按下、移动、松开和 Esc 事件。包含真实视场检测算法对移动条带的检测，以及受控后台完成顺序。没有连接现场相机，也没有宣称完成 Windows 实机验收。

联合回归：**313 passed，20.18 s**。`git diff --check` 通过。

重点验证：

| 场景 | 结果 |
|---|---|
| 手动／混合模式，两点取点跨过运动 | 首点保留，仅在有效第二点后产生一条结果 |
| 稳定判定完成但新画面还未 paint | 第二次点击不清首点、不提交；绘制后可完成 |
| 原条带宽度约 15 px，新条带约 25 px | 贴边结果采用新画面得到约 25 μm；图片留存接收到同一新帧 |
| 首点后更改图片存档开关 | 本条仍沿用首点时固定的策略 |
| 按住拖动经过运动／运动期间松手 | 保留操作，稳定后完成，指针引用释放 |
| 连续八次运动 | 同一测量身份保留，帧预算不累积；关闭后引用归零 |
| 旧后台结果已完成但未入账，同时新首点存在 | 新首点保留，旧结果取消，旧候选清空 |
| Esc、上下文切换和断开 | 草稿与引用按原规则释放 |
| 直径试验、普通稳定首帧锁定、候选交互 | 保持原行为 |

执行命令：

```sh
PYTHONPATH=src QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q \
  tests/test_live_manual_motion_continuation.py \
  tests/test_live_experiment_motion_pipeline.py \
  tests/test_live_explicit_manual_gesture.py \
  tests/test_live_mixed_presented_frame.py \
  tests/test_live_measurement_mixed_canvas.py \
  tests/test_live_measurement_candidate_canvas.py \
  tests/test_live_experiment_content_workspace.py \
  tests/test_live_experiment_runtime.py \
  tests/test_live_experiment_post_classification.py \
  tests/test_live_experiment_canvas_navigation.py \
  tests/test_live_experiment_transition_confirmation.py \
  tests/test_live_experiment_cu_transition.py \
  tests/test_trial_preferences_runtime.py \
  tests/test_trial_assistance_ui.py
```

日志保留在本地 `build/trial-manual-motion-regression-20261112.log`。本次不提交截图、录像或构建产物，不改软件版本，不打包、不推送。
