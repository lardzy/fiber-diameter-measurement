# 关闭与项目切换复核

日期：2026-10-16。平台：macOS / Qt offscreen。使用真实 MainWindow 导航、项目保存协调器和确定性模拟采集 backend；未打开真实相机。

## 复现与修复

1. 进入含量试验并填写样品信息，关闭窗口或打开其他项目，在未保存确认中取消。原实现提前销毁试验页，取消后导航退回实时采集，缓存为空且工具栏仍沿用试验页隐藏状态。现在确认前只暂停来源、取消未完成请求并提交编辑；取消或保存失败恢复原页面和原采集订阅。已接受的本场记录、field_id、记录表、视场元数据及报告页选择保持。
2. 离开采集触发异步停止，立即重新进入；旧停止回调完成并已排队自动重启，此时接受关闭窗口。原实现处理队列后在隐藏窗口中再次打开 backend（starts 1→2）。现在接受关闭或项目重置会撤销本次进入与重启意图，关闭后 starts 保持 1。
3. 正式销毁已确认/保存的项目页时，不再新增视场结束元数据；普通独立关闭和显式结束视场仍采用原有语义。该调整仅涉及生命周期控制，没有修改测量计算。

## 回归

新增 `tests/test_live_experiment_transition_confirmation.py` 覆盖取消关闭后重试、取消打开项目后重试、后台保存成功/失败、保存期间聚焦编辑后再次取消，以及已排队重启的关闭时序。取消用例均包含已接受的本场记录；保存成功比较内存与项目文件中的视场元数据，确认销毁没有产生新的未保存变更。

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -x -q \
  tests/test_live_experiment_transition_confirmation.py \
  tests/test_live_experiment_project_lifecycle.py \
  tests/test_acquisition_workflow_redesign.py \
  tests/test_acquisition_workbench.py \
  tests/test_live_experiment_content_workspace.py \
  tests/test_live_experiment_save_rebase.py \
  tests/test_project_background_save.py --tb=line
141 passed in 30.91s
```

完整输出：`lifecycle-tests.log`。Windows 安装包与真实 SDK 生命周期仍由对应 Windows 验证证据覆盖，本次模拟回归不代替硬件验收。
