from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox

from fdm.raster import RasterPixelType, RasterPlane
from fdm.services.raster_derivation import (
    FrozenRasterRoi,
    RasterBounds,
    RasterDerivationError,
)
from fdm.services.raster_io import numpy_to_raster_plane
from fdm.ui.raster_derivation_dialogs import (
    Gray8RasterDocumentDescriptor,
    RasterChannelMergeDialog,
    RasterChannelMergeRequest,
    RasterCopyDerivationRequest,
    RasterCopyDialog,
    RasterCopyScope,
    RasterMaskOutsideMode,
)
from fdm.ui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox


class _FakeWheelEvent:
    def __init__(self) -> None:
        self.ignored = False
        self.accepted = False

    def ignore(self) -> None:
        self.ignored = True

    def accept(self) -> None:
        self.accepted = True


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _plane(array: np.ndarray) -> RasterPlane:
    return numpy_to_raster_plane(array)


def _roi(
    *,
    bounds: RasterBounds = RasterBounds(1, 1, 2, 2),
) -> FrozenRasterRoi:
    return FrozenRasterRoi.from_numpy(
        np.asarray([[True, False], [False, True]], dtype=np.bool_),
        bounds=bounds,
    )


def _descriptor(
    document_id: str,
    *,
    width: int = 4,
    height: int = 3,
    calibration: str = "calibration:5px-per-um",
    seed: str | None = None,
) -> Gray8RasterDocumentDescriptor:
    digest_seed = (seed or document_id).encode("utf-8").hex()
    digest = (digest_seed * 64)[:64].ljust(64, "0")
    return Gray8RasterDocumentDescriptor(
        document_id=document_id,
        display_name=f"图片 {document_id}",
        width=width,
        height=height,
        pixel_sha256=digest,
        calibration_signature=calibration,
    )


def test_copy_dialog_emits_immutable_full_image_request(
    app: QApplication,
) -> None:
    source = _plane(np.arange(12, dtype=np.uint8).reshape(3, 4))
    dialog = RasterCopyDialog(source, source_name="样品 A")
    requests: list[RasterCopyDerivationRequest] = []
    dialog.copyRequested.connect(requests.append)
    try:
        assert dialog.scopeCombo.count() == 1
        assert dialog.scopeCombo.currentData() == RasterCopyScope.FULL_IMAGE.value
        assert not dialog.outsideModeCombo.isVisible()

        dialog._emit_request()  # noqa: SLF001

        assert len(requests) == 1
        request = requests[0]
        assert request.source is source
        assert request.source_sha256 == source.sha256()
        assert request.scope is RasterCopyScope.FULL_IMAGE
        assert request.bounds is None
        assert request.roi is None
        with pytest.raises(FrozenInstanceError):
            request.scope = RasterCopyScope.ROI_BOUNDS  # type: ignore[misc]
    finally:
        dialog.close()
        app.processEvents()


def test_copy_dialog_exposes_bounds_and_mask_modes(
    app: QApplication,
) -> None:
    source = _plane(np.arange(48, dtype=np.uint8).reshape(4, 4, 3))
    roi = _roi()
    dialog = RasterCopyDialog(source, roi=roi)
    try:
        assert dialog.scopeCombo.count() == 3
        assert dialog.scopeCombo.currentData() == RasterCopyScope.ROI_BOUNDS.value
        bounds_request = dialog.request()
        assert bounds_request.scope is RasterCopyScope.ROI_BOUNDS
        assert bounds_request.bounds == roi.bounds
        assert bounds_request.roi is None

        dialog.scopeCombo.setCurrentIndex(
            dialog.scopeCombo.findData(RasterCopyScope.ROI_MASK.value)
        )
        assert dialog.outsideModeCombo.currentData() == (
            RasterMaskOutsideMode.TRANSPARENT.value
        )
        transparent_request = dialog.request()
        assert transparent_request.scope is RasterCopyScope.ROI_MASK
        assert transparent_request.roi is roi
        assert transparent_request.transparent_outside
        assert transparent_request.fill_value is None

        dialog.outsideModeCombo.setCurrentIndex(
            dialog.outsideModeCombo.findData(
                RasterMaskOutsideMode.FILL_VALUE.value
            )
        )
        for spin, value in zip(
            dialog.fillSpins,
            (10, 20, 30),
            strict=True,
        ):
            spin.setValue(value)
        fill_request = dialog.request()
        assert not fill_request.transparent_outside
        assert fill_request.fill_value == (10, 20, 30)
        assert dialog.fillGroup.isVisible() is False  # dialog is not shown
        assert not dialog.fillGroup.isHidden()
    finally:
        dialog.close()
        app.processEvents()


def test_copy_dialog_scientific_gray_uses_explicit_fill_only(
    app: QApplication,
) -> None:
    source = _plane(np.arange(16, dtype=np.uint16).reshape(4, 4))
    dialog = RasterCopyDialog(source, roi=_roi())
    try:
        assert dialog.outsideModeCombo.count() == 1
        assert dialog.outsideModeCombo.currentData() == (
            RasterMaskOutsideMode.FILL_VALUE.value
        )
        dialog.scopeCombo.setCurrentIndex(
            dialog.scopeCombo.findData(RasterCopyScope.ROI_MASK.value)
        )
        dialog.fillSpins[0].setValue(65_000)
        request = dialog.request()
        assert request.fill_value == 65_000
        assert not request.transparent_outside
    finally:
        dialog.close()
        app.processEvents()


def test_copy_request_rejects_stale_source_and_ambiguous_mask() -> None:
    source = _plane(np.ones((4, 4), dtype=np.uint8))
    roi = _roi()
    with pytest.raises(RasterDerivationError, match="SHA256"):
        RasterCopyDerivationRequest(
            source=source,
            source_sha256="0" * 64,
            scope=RasterCopyScope.FULL_IMAGE,
        )
    with pytest.raises(RasterDerivationError, match="必须且只能"):
        RasterCopyDerivationRequest(
            source=source,
            source_sha256=source.sha256(),
            scope=RasterCopyScope.ROI_MASK,
            bounds=roi.bounds,
            roi=roi,
        )
    gray16 = _plane(np.ones((4, 4), dtype=np.uint16))
    with pytest.raises(RasterDerivationError, match="不能隐式转换"):
        RasterCopyDerivationRequest(
            source=gray16,
            source_sha256=gray16.sha256(),
            scope=RasterCopyScope.ROI_MASK,
            bounds=roi.bounds,
            roi=roi,
            transparent_outside=True,
        )


def test_copy_dialog_all_combo_and_numeric_controls_ignore_wheel(
    app: QApplication,
) -> None:
    source = _plane(np.zeros((4, 4, 4), dtype=np.uint8))
    dialog = RasterCopyDialog(source, roi=_roi())
    try:
        combos = dialog.findChildren(QComboBox)
        spins = dialog.findChildren(QDoubleSpinBox)
        assert combos and spins
        assert all(isinstance(combo, NoWheelComboBox) for combo in combos)
        assert all(isinstance(spin, NoWheelDoubleSpinBox) for spin in spins)
        for editor in (*combos, *spins):
            before = (
                editor.currentIndex()
                if isinstance(editor, QComboBox)
                else editor.value()
            )
            event = _FakeWheelEvent()
            editor.wheelEvent(event)
            after = (
                editor.currentIndex()
                if isinstance(editor, QComboBox)
                else editor.value()
            )
            assert before == after
            assert event.ignored or event.accepted
    finally:
        dialog.close()
        app.processEvents()


def test_gray8_descriptor_rejects_invalid_contract() -> None:
    with pytest.raises(RasterDerivationError, match="GRAY8"):
        Gray8RasterDocumentDescriptor(
            document_id="bad",
            display_name="16 位",
            width=2,
            height=2,
            pixel_sha256="a" * 64,
            calibration_signature="uncalibrated",
            pixel_type=RasterPixelType.GRAY16,
        )
    with pytest.raises(RasterDerivationError, match="标定签名"):
        Gray8RasterDocumentDescriptor(
            document_id="bad",
            display_name="无签名",
            width=2,
            height=2,
            pixel_sha256="a" * 64,
            calibration_signature="",
        )
    with pytest.raises(RasterDerivationError, match="64 位"):
        Gray8RasterDocumentDescriptor(
            document_id="bad",
            display_name="坏摘要",
            width=2,
            height=2,
            pixel_sha256="bad",
            calibration_signature="uncalibrated",
        )


def test_merge_dialog_selects_first_compatible_distinct_triplet_and_emits(
    app: QApplication,
) -> None:
    documents = (
        _descriptor("wrong-size", width=8),
        _descriptor("red"),
        _descriptor("green"),
        _descriptor("blue"),
    )
    dialog = RasterChannelMergeDialog(documents)
    requests: list[RasterChannelMergeRequest] = []
    dialog.mergeRequested.connect(requests.append)
    try:
        assert all(
            isinstance(combo, NoWheelComboBox)
            for combo in (
                dialog.redCombo,
                dialog.greenCombo,
                dialog.blueCombo,
            )
        )
        assert dialog.request().document_ids == ("red", "green", "blue")
        assert dialog.createButton.isEnabled()
        assert "4×3 RGB8" in dialog.validationLabel.text()

        dialog._emit_request()  # noqa: SLF001

        assert len(requests) == 1
        assert requests[0].document_ids == ("red", "green", "blue")
        with pytest.raises(FrozenInstanceError):
            requests[0].red = documents[0]  # type: ignore[misc]
    finally:
        dialog.close()
        app.processEvents()


def test_merge_dialog_rejects_duplicate_or_incompatible_selection(
    app: QApplication,
) -> None:
    documents = (
        _descriptor("a"),
        _descriptor("b"),
        _descriptor("c", calibration="calibration:other"),
    )
    dialog = RasterChannelMergeDialog(documents)
    try:
        assert not dialog.createButton.isEnabled()
        assert "尺寸和标定" in dialog.validationLabel.text()

        dialog.greenCombo.setCurrentIndex(
            dialog.greenCombo.findData("a")
        )
        assert not dialog.createButton.isEnabled()
        assert "三个不同来源" in dialog.validationLabel.text()
        with pytest.raises(RasterDerivationError, match="三个不同来源"):
            dialog.request()
    finally:
        dialog.close()
        app.processEvents()


def test_merge_dialog_with_fewer_than_three_sources_is_safe(
    app: QApplication,
) -> None:
    dialog = RasterChannelMergeDialog(
        (_descriptor("a"), _descriptor("b"))
    )
    try:
        assert not dialog.createButton.isEnabled()
        assert "至少需要三张" in dialog.validationLabel.text()
    finally:
        dialog.close()
        app.processEvents()


def test_merge_dialog_combo_boxes_ignore_wheel(
    app: QApplication,
) -> None:
    dialog = RasterChannelMergeDialog(
        (_descriptor("a"), _descriptor("b"), _descriptor("c"))
    )
    try:
        for combo in (
            dialog.redCombo,
            dialog.greenCombo,
            dialog.blueCombo,
        ):
            before = combo.currentIndex()
            event = _FakeWheelEvent()
            combo.wheelEvent(event)
            assert combo.currentIndex() == before
            assert event.ignored or event.accepted
    finally:
        dialog.close()
        app.processEvents()


def test_merge_dialog_rejects_duplicate_document_descriptors(
    app: QApplication,
) -> None:
    duplicate = _descriptor("same")
    with pytest.raises(RasterDerivationError, match="ID 不能重复"):
        RasterChannelMergeDialog((duplicate, duplicate))
    app.processEvents()
