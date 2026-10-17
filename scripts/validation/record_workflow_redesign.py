"""Record real Qt navigation with fake capture, never the user's desktop.

This complements the all-Show-event observer: a 20 fps movie alone cannot
prove that no sub-frame transient window appeared. The JSON keeps both the
step timeline and every top-level Show event. Requires ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", default="1366x768", help="offscreen client size WIDTHxHEIGHT")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from validate_workflow_redesign import WorkflowHarness, dimensions, settle, source_fingerprints
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter
    from PySide6.QtTest import QTest

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for the optional process recording")
    width, height = dimensions(args.size)
    if width % 2 or height % 2:
        raise ValueError("H.264 recording requires an even client width and height")
    path = args.output / "workflow-offscreen.mp4"
    timeline, samples = [], []
    recording_error = []
    profile = Path(tempfile.mkdtemp(prefix="isolated-profile-", dir=args.output))
    before = source_fingerprints()
    began = time.perf_counter()
    with (args.output / "ffmpeg.log").open("w") as log, WorkflowHarness(profile, size=(width, height)) as harness:
        encoder = subprocess.Popen([ffmpeg, "-y", "-loglevel", "warning", "-f", "rawvideo",
            "-pixel_format", "rgba", "-video_size", f"{width}x{height + 32}", "-framerate", "20", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        caption = "测量分析"
        timer = QTimer(harness.window)
        timer.setInterval(50)

        def capture():
            try:
                if harness.window._capture_manager.is_preview_active():
                    # A physical camera is never constructed; feed one complete,
                    # owned synthetic frame through the real capture manager.
                    harness.backend.tick()
                harness.assert_size()
                image = QImage(width, height + 32, QImage.Format.Format_RGBA8888)
                image.fill(QColor("#171b20"))
                painter = QPainter(image)
                painter.drawPixmap(0, 32, harness.window.grab())
                painter.setPen(QColor("#ffd28a"))
                painter.setFont(QFont("PingFang SC", 11))
                painter.drawText(12, 22, f"OFFSCREEN · 模拟相机 · 20 fps · {caption}")
                painter.end()
                encoder.stdin.write(image.constBits())
                samples.append(round(time.perf_counter() - began, 4))
            except Exception as exc:
                timer.stop()
                recording_error.append(repr(exc))

        timer.timeout.connect(capture)
        timer.start()

        def step(label, action, duration=700):
            nonlocal caption
            caption = label
            timeline.append({"label": label, "elapsed_seconds": round(time.perf_counter() - began, 4),
                             "video_frame": len(samples)})
            action()
            settle()
            capture()
            QTest.qWait(duration)
            assert not recording_error, recording_error
            assert not harness.slot_errors, harness.slot_errors

        try:
            step("测量分析：原有面板", lambda: None)
            step("进入采集：自动连接设备", harness.enter)
            step("含量试验", lambda: harness.select(1))
            step("打开本场记录：图像区域不变", lambda: harness.hub.pages.currentWidget().workflow_tabs.setCurrentIndex(1))
            step("返回操作", lambda: harness.hub.pages.currentWidget().workflow_tabs.setCurrentIndex(0))
            step("直径试验", lambda: harness.select(2))
            step("回实时采集：导航保留", lambda: harness.select(0))
            for index in (1, 2, 0) * 4:
                step(f"快速连续切换：{('实时采集', '含量试验', '直径试验')[index]}",
                     lambda index=index: harness.select(index), 120)
            step("退出测量：停止采集并恢复面板", harness.leave)
            step("再次进入：自动恢复设备", harness.enter)
            step("停止采集", lambda: harness.hub.capture_button.click())
            step("重启采集", lambda: harness.hub.capture_button.click())
            step("模拟断线：页面内提示", harness.backend.fail)
            step("重试：恢复完整帧", lambda: harness.hub.capture_button.click())
            step("含量页选择 137%", lambda: (harness.select(1),
                harness.hub.pages.currentWidget().display_binding.set_mode("percent", 137)))
            step("直径页共享显示比例", lambda: harness.select(2))
            step("显示改为适合窗口", lambda: harness.hub.pages.currentWidget().display_binding.set_mode("fit"))
            step("返回测量：空试验未产生待保存改动", harness.leave)
            harness.assert_clean()
            assert not harness.observer.unexpected(), harness.observer.unexpected()
        finally:
            timer.stop()
            encoder.stdin.close()
            returncode = encoder.wait(timeout=30)
        assert returncode == 0 and samples, "ffmpeg failed; see ffmpeg.log"
        report = {"passed": True, "real_hardware": False, "platform": "offscreen", "fps": 20,
            "client_size": [width, height], "video_size": [width, height + 32], "font": harness.font_report(),
            "frames": len(samples), "video_seconds": len(samples) / 20,
            "wall_seconds": round(time.perf_counter() - began, 3), "video": path.name,
            "source_fingerprints": before, "source_changed_during_run": before != source_fingerprints(),
            "timeline": timeline, "frame_wall_timestamps": samples,
            "top_level_show_events": harness.observer.events,
            "unexpected_windows": harness.observer.unexpected(), "slot_errors": harness.slot_errors,
            "limitations": ["Qt offscreen rendering and deterministic fake frames; no real SDK or desktop capture.",
                "Video is 20 fps. The independent Show-event observer checks windows shorter than a video frame.",
                "Encoding is constant-frame-rate; frame_wall_timestamps preserve observed timing."]}
        (args.output / "recording.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"video": str(path), "frames": len(samples), "passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
