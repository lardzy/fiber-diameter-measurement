# 首次图片无效

installed-window.png 未通过实际视觉检查，图片内容是其他 Windows 窗口。该次捕获未核验前台窗口，且未采用 DPI-aware 坐标。保留原图与原始进程/退出结果，不将其作为 FDM 可见性证据。

有效截图见 ../explorer-visual-recheck/installed-window-printwindow.png 及 result.json；同一安装 EXE 已再次以普通用户启动、校验前台 HWND，并正常关闭。
