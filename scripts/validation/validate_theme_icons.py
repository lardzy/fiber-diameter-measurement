"""Render a component gallery for each palette; this is not full-workflow acceptance."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1)
    args = parser.parse_args()
    os.environ["QT_SCALE_FACTOR"] = str(args.scale)
    from PySide6.QtCore import QSize, Qt
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
        QLineEdit, QPushButton, QTabBar, QToolButton, QVBoxLayout, QWidget,
    )
    from fdm.ui.icons import themed_icon
    from fdm.ui.theme import apply_application_theme, semantic_colors

    app = QApplication.instance() or QApplication([])
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"boundary": "Qt component gallery, no device or complete application workflow", "palettes": []}
    for mode, title in (("light", "实验室浅色"), ("dark", "石墨深色"), ("navy", "深蓝"), ("warm", "暖灰")):
        apply_application_theme(app, mode)
        widget = QWidget()
        widget.setAutoFillBackground(True)
        widget.setWindowTitle(title)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(24, 20, 24, 20)
        label = QLabel(title + " · 主题与图标组件核对")
        label.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(label)
        tabs = QTabBar()
        tabs.setExpanding(False)
        for name, text in (("measurement", "测量分析"), ("acquisition", "采集与试验"), ("digital_slide", "数字切片")):
            tabs.addTab(themed_icon(name), text)
        tabs.setCurrentIndex(1)
        layout.addWidget(tabs)
        controls = QHBoxLayout()
        sample = QLineEdit("样品 2026-001")
        sample.setAccessibleName("样品名称")
        controls.addWidget(sample)
        ruler = QComboBox()
        ruler.addItems(["10× · 0.632496 μm/px", "20× · 0.316248 μm/px"])
        controls.addWidget(ruler)
        start = QPushButton(themed_icon("start_capture", color=semantic_colors().on_accent), "开始采集")
        start.setProperty("emphasis", "primary")
        controls.addWidget(start)
        layout.addLayout(controls)
        statuses = QHBoxLayout()
        for tone, text in (("info", "正在连接"), ("success", "可以测量"), ("warning", "请确认标尺"), ("error", "设备已断开")):
            frame = QFrame()
            frame.setProperty("statusTone", tone)
            row = QHBoxLayout(frame)
            hint = QLabel(text)
            hint.setProperty("statusTone", tone)
            row.addWidget(hint)
            statuses.addWidget(frame)
        layout.addLayout(statuses)
        grid = QGridLayout()
        entries = (("content_trial", "含量试验"), ("diameter_trial", "直径试验"),
                   ("capture_frame", "抓拍图像"), ("records", "记录复核"), ("report", "生成报告"),
                   ("save_project", "保存项目"), ("next_field", "新视场"), ("undo", "撤销"),
                   ("settings", "设备设置"), ("help", "操作指引"))
        for index, (name, text) in enumerate(entries):
            button = QToolButton()
            button.setText(text)
            button.setIcon(themed_icon(name))
            button.setIconSize(QSize(20, 20))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            grid.addWidget(button, index // 5, index % 5)
        layout.addLayout(grid)
        disabled = QPushButton(themed_icon("export"), "暂无测量记录，报告不可用")
        disabled.setEnabled(False)
        layout.addWidget(disabled)
        widget.resize(900, 390)
        widget.show()
        app.processEvents()
        output = args.output / f"components-{mode}@{args.scale:g}x.png"
        widget.grab().save(str(output))
        report["palettes"].append({"mode": mode, "logical_size": [widget.width(), widget.height()],
                                   "dpr": widget.devicePixelRatioF(), "screenshot": output.name})
        widget.close()
        widget.deleteLater()
        app.processEvents()
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
