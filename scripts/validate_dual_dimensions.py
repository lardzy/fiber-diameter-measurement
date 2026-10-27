"""Render Chinese CAD-dimension fixtures through FDM's actual overlay renderers.

The PNGs are static software examples, not instrument images or accuracy tests.
No application settings, user documents or hardware are accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", default="windows" if sys.platform == "win32" else "offscreen")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        parser.error("use a fresh output directory to preserve prior evidence")
    os.environ["QT_QPA_PLATFORM"] = args.platform
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen
    from PySide6.QtWidgets import QApplication
    from fdm.geometry import Line, Point
    from fdm.measurement_annotation import MeasurementAnnotationLayout
    from fdm.metrology import MeasurementDefinition, MeasurementFeature
    from fdm.models import ImageDocument, Measurement
    from fdm.settings import AppSettings
    from fdm.ui.professional_rendering import draw_professional_measurements
    from fdm.ui.rendering import draw_measurements, resolve_measurement_annotation

    app = QApplication.instance() or QApplication([])
    families = set(QFontDatabase.families())
    family = next((name for name in ("PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "Heiti SC") if name in families), app.font().family())
    font = QFont(family); font.setPixelSize(21)
    settings = AppSettings(measurement_label_font_family=family, measurement_label_font_size=21,
                           measurement_label_color="#F4CD7B")
    report = {"schema": 1, "system": platform.platform(), "font": family,
              "scope": "实际 FDM 绘制器生成的合成二维尺寸示例；无设备和真实精度结论",
              "checks": [], "images": [], "source_sha256": {}}

    def title(painter, text, rect, size=24, color="#EAF2F5"):
        selected = QFont(family); selected.setPixelSize(size)
        painter.setFont(selected); painter.setPen(QColor(color))
        painter.drawText(QRectF(*rect), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap, text)
        painter.setFont(font)

    def page(name, heading, subtitle, size=(1760, 1000)):
        image = QImage(*size, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor("#0C1520"))
        painter = QPainter(image); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(font)
        title(painter, heading, (60, 20, 1635, 56), 34)
        title(painter, subtitle, (60, 74, 1640, 44), 19, "#A1B4C6")
        return name, image, painter

    def card(painter, rect, heading, description):
        painter.setPen(QPen(QColor("#2A4051"), 1))
        painter.setBrush(QColor("#122331")); painter.drawRoundedRect(QRectF(*rect), 14, 14)
        x, y, width, _height = rect
        title(painter, heading, (x + 24, y + 12, width - 48, 40), 23)
        title(painter, description, (x + 24, y + 49, width - 48, 62), 17, "#9FB7C8")

    def nodes(painter, points, names="ABCD", offsets=None):
        painter.setBrush(QColor("#F2C66D")); painter.setPen(QPen(QColor("#F2C66D"), 1))
        for index, (name, point) in enumerate(zip(names, points)):
            painter.drawEllipse(QPointF(point.x, point.y), 4, 4)
            dx, dy = offsets[index] if offsets else (-8, 10)
            title(painter, name, (point.x + dx, point.y + dy, 65, 32), 22, "#F4CD7B")

    def document(name):
        item = ImageDocument(name, "", (1760, 1100)); item.initialize_runtime_state()
        return item

    def add_chain(doc, points, offsets, *, legacy=False):
        for i, (start, end, offset) in enumerate(zip(points, points[1:], offsets)):
            layout = MeasurementAnnotationLayout(leader_style="dual", placement_mode="manual", offset_px=offset)
            if legacy:
                item = Measurement(f"{doc.id}-m{i}", doc.id, None, "manual", line_px=Line(start, end), annotation_layout=layout)
                item.recalculate(None); doc.measurements.append(item)
            else:
                feature = MeasurementFeature(f"{doc.id}-f{i}", doc.id, "line", (start, end))
                definition = MeasurementDefinition(f"{doc.id}-p{i}", doc.id, "distance", (feature.id,), annotation_layout=layout)
                doc.professional_features.append(feature); doc.professional_measurements.append(definition)

    def render(painter, doc, expected, *, legacy=False):
        mapper = lambda p: QPointF(p.x, p.y)
        painter.setFont(font)
        if legacy:
            draw_measurements(painter, doc, mapper, settings, line_width=1.7, endpoint_radius=2.5, use_sprite_cache=False)
            labels = [resolve_measurement_annotation(item, doc, settings, mapper) for item in doc.measurements]
        else:
            labels = list(draw_professional_measurements(painter, doc, mapper).values())
        for index, (label, pair) in enumerate(zip(labels, expected)):
            truth = tuple(mapper(p) for p in pair)
            assert label.source_points == truth, (doc.id, index, label.source_points, truth)
            assert len(label.witness_lines) == len(label.arrowheads) == 2
            report["checks"].append({"document": doc.id, "index": index + 1, "source_points": [[p.x(), p.y()] for p in label.source_points], "distinct_witnesses": True})

    def save(page_data):
        name, image, painter = page_data
        title(painter, "说明：黄色端点字母与示意引线仅用于本图解，不是软件自动生成的编号或测量字段。", (65, image.height() - 34, 1630, 26), 17, "#A1B4C6")
        painter.end()
        target = output / f"{name}.png"
        assert image.save(str(target))
        assert QImage(str(target)).convertToFormat(image.format()) == image
        report["images"].append({"file": target.name, "size": [image.width(), image.height()], "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})

    sheet = page("01-contiguous-old-and-professional", "连续共线尺寸：每一条都只关联自己的两端", "黄色 A / B / C / D 为真实取点；尺寸界线保持独立，文字移动不会改变取点。")
    painter = sheet[2]
    for legacy, y, caption in ((True, 420, "原有直径测量 · M1 / M2 / M3"), (False, 825, "专业测量 · P1 / P2 / P3")):
        card(painter, (60, y - 300, 1640, 365), caption, "相邻尺寸可以共用 B、C 节点；A—B、B—C、C—D 各有两条尺寸界线和各自的尺寸线。")
        points = [Point(x, y) for x in (260, 660, 1060, 1460)]
        doc = document("legacy-chain" if legacy else "professional-chain")
        add_chain(doc, points, [Point(0, offset) for offset in (-80, -140, -200)], legacy=legacy)
        render(painter, doc, list(zip(points, points[1:])), legacy=legacy); nodes(painter, points)
    title(painter, "同一套双端布局用于画布与叠加导出。标签位置保存在图像坐标中，测量值与原始图像保持不变。", (65, 923, 1630, 32), 19, "#A1B4C6")
    save(sheet)

    sheet = page("02-orientations-short-and-distant", "横、竖、斜与短尺寸：保持明确的端点归属", "尺寸线始终平行于实际测量段；短尺寸采用外置箭头，拖远文字时保留连接。", (1760, 1100))
    painter = sheet[2]
    card(painter, (60, 145, 500, 850), "竖向连续尺寸", "分别连接 A—B、B—C、C—D；左右偏移只改变标注布局。")
    points = [Point(410, y) for y in (340, 520, 700, 880)]
    doc = document("vertical"); add_chain(doc, points, [Point(-80, 0), Point(-140, 0), Point(-200, 0)])
    render(painter, doc, list(zip(points, points[1:]))); nodes(painter, points, offsets=[(15, -15)] * 4)
    card(painter, (590, 145, 550, 850), "斜向对齐尺寸", "界线垂直于测量段；箭头投影保持每一段自身的起止点。")
    points = [Point(660, 800), Point(760, 700), Point(860, 600), Point(960, 500)]
    doc = document("diagonal"); add_chain(doc, points, [Point(85, 45), Point(90, 65), Point(110, 85)])
    render(painter, doc, list(zip(points, points[1:]))); nodes(painter, points, offsets=[(-25, -40)] * 4)
    card(painter, (1170, 145, 530, 850), "短尺寸 / 文字拖远", "真实间距小于文字宽度时，尺寸线与箭头不会被文字完全盖住。")
    points = [Point(1400, 550), Point(1408, 550)]
    doc = document("short-8px"); add_chain(doc, points, [Point(0, -110)])
    render(painter, doc, [tuple(points)]); nodes(painter, points, names="AB", offsets=[(-45, 28), (35, 28)])
    painter.setPen(QPen(QColor("#F4CD7B"), .8))
    painter.drawLine(QPointF(1369, 582), QPointF(1400, 556))
    painter.drawLine(QPointF(1445, 582), QPointF(1408, 556))
    title(painter, "8 px 短间距 · 箭头外置", (1230, 620, 430, 35), 19, "#A1B4C6")
    points = [Point(1270, 870), Point(1480, 870)]
    doc = document("distant-label"); add_chain(doc, points, [Point(180, -130)])
    render(painter, doc, [tuple(points)]); nodes(painter, points, names="CD")
    title(painter, "文字移到区间外 · 两端归属仍保留", (1200, 938, 470, 35), 18, "#A1B4C6")
    save(sheet)

    sheet = page("03-radius-and-horizontal-component", "显示哪个指标，就标出哪个实际尺寸区间", "选择半径时使用圆心到边界；选择水平分量时使用原点到水平投影点。", (1760, 820))
    painter = sheet[2]
    card(painter, (60, 145, 790, 590), "圆：选择“半径”", "双端连接 C（圆心）与 E（圆边界），不再沿用完整直径的两端。")
    doc = document("radius")
    feature = MeasurementFeature("circle", doc.id, "circle", (Point(560, 470), Point(440, 590), Point(320, 470)))
    doc.professional_features.append(feature)
    doc.professional_measurements.append(MeasurementDefinition("radius", doc.id, "circle", (feature.id,), annotation_layout=MeasurementAnnotationLayout(leader_style="dual", metric_key="radius", offset_px=Point(70, -130))))
    render(painter, doc, [(Point(440, 470), Point(560, 470))]); nodes(painter, [Point(440, 470), Point(560, 470)], names="CE")
    card(painter, (890, 145, 810, 590), "斜线：选择“水平分量”", "尺寸端点为 A 与 Q。Q 是 B 在 A 所在水平线上的投影，灰色斜线仅示意原始取点关系。")
    a, b, q = Point(1020, 570), Point(1530, 320), Point(1530, 570)
    painter.setPen(QPen(QColor("#648293"), 1, Qt.PenStyle.DashLine)); painter.drawLine(QPointF(a.x, a.y), QPointF(b.x, b.y))
    doc = document("horizontal-component")
    feature = MeasurementFeature("line", doc.id, "line", (a, b)); doc.professional_features.append(feature)
    doc.professional_measurements.append(MeasurementDefinition("dx", doc.id, "distance", (feature.id,), annotation_layout=MeasurementAnnotationLayout(leader_style="dual", metric_key="dx", offset_px=Point(0, -120))))
    render(painter, doc, [(a, q)]); nodes(painter, [a, b, q], names="ABQ", offsets=[(-10, 12), (10, -22), (10, 12)])
    save(sheet)

    for relative in ("scripts/validate_dual_dimensions.py", "src/fdm/measurement_annotation.py", "src/fdm/ui/measurement_annotation_layout.py", "src/fdm/ui/rendering.py", "src/fdm/ui/professional_rendering.py"):
        report["source_sha256"][relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    report["passed"] = True
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "images": len(report["images"]), "endpoint_checks": len(report["checks"]), "passed": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
