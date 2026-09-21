"""Culling must keep logical widget pixels separate from physical image pixels."""
from __future__ import annotations

import math

import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter, QPicture
from PySide6.QtWidgets import QWidget

from fdm.ui.rendering import _is_visible_to_painter, _painter_visible_rect


class _VisibilityProbe(QWidget):
    def __init__(self, *, custom_view=False, rotate=False, clip=False, world=True):
        super().__init__()
        self.resize(640, 480)
        self.custom_view = custom_view
        self.rotate = rotate
        self.clip = clip
        self.world = world
        self.visible = None
        self.corner_visible = False

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            if self.custom_view:
                painter.setViewport(40, 30, 400, 200)
                painter.setWindow(-100, -50, 200, 100)
            painter.translate(20, 10)
            if self.rotate:
                painter.rotate(90)
            painter.scale(2, 4)
            painter.setWorldMatrixEnabled(self.world)
            if self.clip:
                painter.setClipRect(QRectF(7, 11, 30, 40))
            self.visible = _painter_visible_rect(painter)
            # This label is inside the bottom-right of the ordinary widget.
            self.corner_visible = _is_visible_to_painter(painter, QRectF(280, 100, 10, 10))
        finally:
            painter.end()


@pytest.mark.parametrize('target_dpr', [1, 1.25, 1.5, 2])
@pytest.mark.parametrize('custom_view,rotate,world,expected', [
    (False, False, True, (-10, -2.5, 320, 120)),
    (True, False, True, (-60, -15, 100, 25)),
    (True, True, True, (-30, -20, 50, 50)),
    (True, False, False, (-100, -50, 200, 100)),
])
def test_widget_culling_uses_logical_coordinates_when_rendered_to_image(
    desktop_application, target_dpr, custom_view, rotate, world, expected,
):
    widget = _VisibilityProbe(custom_view=custom_view, rotate=rotate, world=world)
    # The target DPR can differ from the widget's screen DPR; render() must
    # preserve all labels in the widget's logical viewport in either case.
    image = QImage(math.ceil(640 * target_dpr), math.ceil(480 * target_dpr),
                   QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(target_dpr)
    try:
        widget.render(image)
        assert widget.visible.getRect() == pytest.approx(expected)
        if not custom_view:
            assert widget.corner_visible
    finally:
        widget.close()
        widget.deleteLater()


@pytest.mark.parametrize('dpr', [1, 1.25, 1.5, 2])
def test_image_culling_converts_physical_viewport_to_logical_coordinates(dpr):
    image = QImage(math.ceil(640 * dpr), math.ceil(480 * dpr),
                   QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    painter = QPainter(image)
    try:
        painter.translate(20, 10)
        painter.scale(2, 4)
        assert _painter_visible_rect(painter).getRect() == pytest.approx((-10, -2.5, 320, 120))
        assert _is_visible_to_painter(painter, QRectF(280, 100, 10, 10))
        assert not _is_visible_to_painter(painter, QRectF(500, 300, 10, 10))
        painter.setClipRect(QRectF(7, 11, 30, 40))
        assert _painter_visible_rect(painter) == QRectF(7, 11, 30, 40)
    finally:
        painter.end()


def test_widget_explicit_clip_and_picture_without_viewport(desktop_application):
    widget = _VisibilityProbe(clip=True)
    image = QImage(1280, 960, QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(2)
    try:
        widget.render(image)
        assert widget.visible == QRectF(7, 11, 30, 40)
    finally:
        widget.close()
        widget.deleteLater()
    picture = QPicture()
    painter = QPainter(picture)
    try:
        assert _painter_visible_rect(painter) is None
        assert _is_visible_to_painter(painter, QRectF(1000, 2000, 100, 100))
    finally:
        painter.end()
