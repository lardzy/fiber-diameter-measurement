from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from fdm.models import ImageDocument
from fdm.services.digital_slide_cache import DigitalSlideSessionCache
from fdm.services.digital_slide_calibration import (
    CALIBRATION_AXIS_X,
    CALIBRATION_AXIS_Y,
    DigitalSlideCalibrationPair,
    DigitalSlideCalibrationSession,
)
from fdm.services.digital_slide_store import (
    DigitalSlideManifest,
    DigitalSlideOverviewAccumulator,
    DigitalSlideStore,
    DigitalSlideTile,
    DigitalSlideTileDescriptor,
    compress_slide_file,
)
from fdm.settings import (
    DIGITAL_SLIDE_PROFILE_FILE_KIND,
    DIGITAL_SLIDE_PROFILE_FILE_VERSION,
    AppSettings,
    DigitalSlideAcquisitionProfile,
    DigitalSlideAcquisitionProfileIO,
)
from fdm.ui.dialogs import DigitalSlideCompressionDialog, SettingsDialog
from fdm.ui.digital_slide_calibration import (
    CalibrationPairPreview,
    DigitalSlideCalibrationDialog,
)
from fdm.ui.digital_slide_canvas import DigitalSlideCanvas
from fdm.ui.main_window import (
    DIGITAL_SLIDE_GRID_ORIGIN_RECORDED,
    DIGITAL_SLIDE_RANGE_MODE_BOUNDARY,
    DIGITAL_SLIDE_RANGE_MODE_GRID,
    DigitalSlideZRangeRail,
    MainWindow,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _solid_image(width: int, height: int, color: str) -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    return image


def _gray_image(array: np.ndarray) -> QImage:
    contiguous = np.ascontiguousarray(array.astype(np.uint8))
    return QImage(
        contiguous.data,
        contiguous.shape[1],
        contiguous.shape[0],
        contiguous.strides[0],
        QImage.Format.Format_Grayscale8,
    ).copy()


def _profile_with_step(
    profile_id: str,
    name: str,
    step: int,
    *,
    fallback: AppSettings,
) -> DigitalSlideAcquisitionProfile:
    values = AppSettings._digital_slide_profile_values_from_settings(fallback)
    values["digital_slide_x_stage_step"] = int(step)
    return DigitalSlideAcquisitionProfile(profile_id, name, values)


def test_legacy_settings_migrate_to_one_default_digital_slide_profile() -> None:
    settings = AppSettings.from_dict(
        {
            "digital_slide_x_stage_step": -4321,
            "digital_slide_z_capture_step": 275,
            "digital_slide_dynamic_focus_overview_enabled": False,
        }
    )

    assert settings.digital_slide_active_profile_id == "default"
    assert len(settings.digital_slide_profiles) == 1
    profile = settings.digital_slide_profiles[0]
    assert profile.name == "默认配置"
    assert profile.values["digital_slide_x_stage_step"] == -4321
    assert profile.values["digital_slide_z_capture_step"] == 275
    assert settings.digital_slide_dynamic_focus_overview_enabled is False


def test_digital_slide_render_cache_setting_roundtrips_and_is_bounded(
    qapp: QApplication,
) -> None:
    del qapp
    assert AppSettings.from_dict(
        {"digital_slide_render_cache_gib": -3}
    ).digital_slide_render_cache_gib == 0
    assert AppSettings.from_dict(
        {"digital_slide_render_cache_gib": 99}
    ).digital_slide_render_cache_gib == 32

    settings = AppSettings.from_dict({"digital_slide_render_cache_gib": 7})
    assert settings.to_dict()["digital_slide_render_cache_gib"] == 7
    dialog = SettingsDialog(settings, document=None)
    try:
        assert dialog._digital_slide_render_cache_spin.value() == 7
        dialog._digital_slide_render_cache_spin.setValue(0)
        assert dialog.app_settings().digital_slide_render_cache_gib == 0
    finally:
        dialog.close()


def test_profile_roundtrip_selects_active_values_and_keeps_flat_compatibility() -> None:
    base = AppSettings().normalized_copy()
    first = _profile_with_step("lens-a", "镜头 A", 4000, fallback=base)
    second = _profile_with_step("lens-b", "镜头 B", -6200, fallback=base)
    payload = base.to_dict()
    payload["digital_slide_profiles"] = [first.to_dict(), second.to_dict()]
    payload["digital_slide_active_profile_id"] = "lens-b"
    payload["digital_slide_x_stage_step"] = 999

    loaded = AppSettings.from_dict(payload)

    assert loaded.digital_slide_active_profile_id == "lens-b"
    assert loaded.digital_slide_x_stage_step == -6200
    assert loaded.activate_digital_slide_profile("lens-a") is True
    assert loaded.digital_slide_x_stage_step == 4000
    roundtrip = AppSettings.from_dict(loaded.to_dict())
    assert roundtrip.digital_slide_active_profile_id == "lens-a"
    assert roundtrip.digital_slide_x_stage_step == 4000


def test_profile_json_is_versioned_and_import_does_not_overwrite(tmp_path: Path) -> None:
    base = AppSettings().normalized_copy()
    profile = _profile_with_step("source-id", "镜头 A", 3456, fallback=base)
    path = tmp_path / "lens.json"

    DigitalSlideAcquisitionProfileIO.save(profile, path, fallback=base)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == DIGITAL_SLIDE_PROFILE_FILE_KIND
    assert payload["version"] == DIGITAL_SLIDE_PROFILE_FILE_VERSION
    loaded = DigitalSlideAcquisitionProfileIO.load(path, fallback=base)
    assert loaded.profile_id == "source-id"
    assert loaded.values["digital_slide_x_stage_step"] == 3456


def test_settings_dialog_profile_edits_are_drafts_until_applied(
    qapp: QApplication,
) -> None:
    original = AppSettings().normalized_copy()
    original_step = original.digital_slide_x_stage_step
    dialog = SettingsDialog(original, document=None)
    try:
        with patch.object(
            dialog,
            "_prompt_digital_slide_profile_name",
            side_effect=["镜头 B", "镜头 B 副本", "镜头 C"],
        ):
            dialog._add_digital_slide_profile()
            lens_b_id = dialog._digital_slide_active_profile_id
            dialog._digital_slide_x_stage_step_spin.setValue(-7777)
            dialog._digital_slide_profile_combo.setCurrentIndex(0)
            assert dialog._digital_slide_x_stage_step_spin.value() == original_step
            lens_b_index = dialog._digital_slide_profile_combo.findData(lens_b_id)
            dialog._digital_slide_profile_combo.setCurrentIndex(lens_b_index)
            assert dialog._digital_slide_x_stage_step_spin.value() == -7777
            dialog._duplicate_digital_slide_profile()
            dialog._rename_digital_slide_profile()
        assert dialog._digital_slide_profile_combo.count() == 3
        assert original.digital_slide_x_stage_step == original_step
        assert len(original.digital_slide_profiles) == 1

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            dialog._delete_digital_slide_profile()
        assert dialog._digital_slide_profile_combo.count() == 2

        draft = dialog.app_settings()
        assert draft.digital_slide_active_profile_id != original.digital_slide_active_profile_id
        assert draft.digital_slide_x_stage_step == -7777
        dialog.reject()
        assert original.digital_slide_x_stage_step == original_step
        assert len(original.digital_slide_profiles) == 1
    finally:
        dialog.close()


def test_settings_dialog_profile_import_creates_new_id_and_unique_name(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    settings = AppSettings().normalized_copy()
    incoming = _profile_with_step("external-id", "默认配置", 8123, fallback=settings)
    source = tmp_path / "incoming.json"
    DigitalSlideAcquisitionProfileIO.save(incoming, source, fallback=settings)
    dialog = SettingsDialog(settings, document=None)
    try:
        with patch.object(
            dialog,
            "_store_current_digital_slide_profile",
            wraps=dialog._store_current_digital_slide_profile,
        ), patch(
            "fdm.ui.dialogs.QFileDialog.getOpenFileName",
            return_value=(str(source), "采集配置 JSON (*.json)"),
        ):
            dialog._import_digital_slide_profile()
        imported = next(
            profile
            for profile in dialog._digital_slide_profiles_draft
            if profile.profile_id == dialog._digital_slide_active_profile_id
        )
        assert imported.profile_id != "external-id"
        assert imported.name == "默认配置 (2)"
        assert imported.values["digital_slide_x_stage_step"] == 8123
    finally:
        dialog.close()


def test_profile_names_are_case_insensitively_unique_except_for_own_rename(
    qapp: QApplication,
) -> None:
    settings = AppSettings().normalized_copy()
    dialog = SettingsDialog(settings, document=None)
    try:
        current = dialog._digital_slide_profiles_draft[0]
        current.name = "Lens A"
        with patch(
            "fdm.ui.dialogs.QInputDialog.getText",
            return_value=("lens a", True),
        ), patch.object(QMessageBox, "information") as information:
            assert dialog._prompt_digital_slide_profile_name("新增采集配置", "") is None
        information.assert_called_once()

        with patch(
            "fdm.ui.dialogs.QInputDialog.getText",
            return_value=(current.name, True),
        ):
            assert dialog._prompt_digital_slide_profile_name(
                "重命名采集配置",
                current.name,
                exclude_profile_id=current.profile_id,
            ) == current.name
    finally:
        dialog.close()


def test_static_overview_accumulator_and_compression_store_only_middle_focus(
    tmp_path: Path,
) -> None:
    manifest = DigitalSlideManifest(1, 20, 12, 20, 12, [-10, 0, 10])
    accumulator = DigitalSlideOverviewAccumulator(manifest, focus_indices={1})
    for focus_index, color in enumerate(("#AA0000", "#00AA00", "#0000AA")):
        accumulator.add_tile(
            DigitalSlideTile(
                z_index=focus_index,
                x=0,
                y=0,
                width=20,
                height=12,
            ),
            _solid_image(20, 12, color),
        )
    assert set(accumulator.images()) == {1}

    source = tmp_path / "source.fdmslide"
    target = tmp_path / "target.fdmslide"
    store = DigitalSlideStore.create(source, manifest)
    for focus_index, color in enumerate(("#AA0000", "#00AA00", "#0000AA")):
        store.write_tile(
            DigitalSlideTile(
                z_index=focus_index,
                x=0,
                y=0,
                width=20,
                height=12,
            ),
            _solid_image(20, 12, color),
        )
    store.close()

    compress_slide_file(
        source,
        target,
        dynamic_focus_overview_enabled=False,
    )
    compressed = DigitalSlideStore(target)
    try:
        compressed.open()
        assert compressed.read_focus_overview(0).isNull()
        assert not compressed.read_focus_overview(1).isNull()
        assert compressed.read_focus_overview(2).isNull()
        generated = compressed.render_overview(z_index=0, maximum_edge=64)
        assert not generated.isNull()
        assert not compressed.read_focus_overview(0).isNull()
    finally:
        compressed.close()


def test_canvas_static_overview_targets_middle_focus_without_following_focus(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    slide_path = tmp_path / "static-overview.fdmslide"
    store = DigitalSlideStore.create(
        slide_path,
        DigitalSlideManifest(1, 40, 20, 20, 20, [-20, -10, 0, 10, 20]),
    )
    document = ImageDocument(
        id="static-overview",
        path=str(slide_path),
        image_size=(40, 20),
        document_kind="digital_slide",
    )
    document.initialize_runtime_state()
    canvas = DigitalSlideCanvas()
    try:
        canvas.set_dynamic_focus_overview_enabled(False)
        canvas.set_slide_document(document, store)
        assert canvas._overview_target_focus_index() == 2
        canvas.set_focus_index(4)
        assert canvas.focus_index() == 4
        assert canvas._overview_target_focus_index() == 2
        canvas.set_dynamic_focus_overview_enabled(True)
        assert canvas._overview_target_focus_index() == 4
    finally:
        canvas.shutdown()
        store.close()


def test_calibration_estimates_known_translation_and_cleans_network_cache(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(20260825)
    source_pixels = rng.integers(0, 255, size=(80, 160), dtype=np.uint8)
    slide_path = tmp_path / "calibration.fdmslide"
    store = DigitalSlideStore.create(
        slide_path,
        DigitalSlideManifest(1, 160, 80, 100, 80, [0]),
    )
    store.write_tile(
        DigitalSlideTile(
            z_index=0,
            x=0,
            y=0,
            width=100,
            height=80,
            stage_x=0,
            stage_y=0,
        ),
        _gray_image(source_pixels[:, :100]),
    )
    store.write_tile(
        DigitalSlideTile(
            z_index=0,
            x=60,
            y=0,
            width=100,
            height=80,
            stage_x=5000,
            stage_y=0,
        ),
        _gray_image(source_pixels[:, 60:160]),
    )
    store.close()

    temporary_parent = tmp_path / "network-cache-parent"
    cache = DigitalSlideSessionCache(
        temporary_parent=temporary_parent,
        network_path_predicate=lambda _path: True,
    )
    session = DigitalSlideCalibrationSession(cache=cache)
    session.open(slide_path)
    working_path = session.working_path
    assert working_path is not None and working_path != slide_path
    estimate = session.estimate(
        focus_index=0,
        axis=CALIBRATION_AXIS_X,
        target_frame_size=(50, 40),
        target_overlap_percent=40,
        current_stage_step=5000,
    )
    assert estimate.primary_stride_px == pytest.approx(30.0, abs=0.75)
    assert estimate.cross_axis_drift_px == pytest.approx(0.0, abs=0.75)
    assert estimate.pixels_per_step == pytest.approx(0.006, rel=0.08)
    assert estimate.suggested_stage_step == pytest.approx(5000, abs=150)
    session.close()
    assert not working_path.parent.exists()


def test_calibration_estimates_vertical_translation_and_preserves_step_sign(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(20260826)
    source_pixels = rng.integers(0, 255, size=(160, 80), dtype=np.uint8)
    slide_path = tmp_path / "calibration-y.fdmslide"
    store = DigitalSlideStore.create(
        slide_path,
        DigitalSlideManifest(1, 80, 160, 80, 100, [0]),
    )
    for y, stage_y in ((0, 0), (60, -4000)):
        store.write_tile(
            DigitalSlideTile(
                z_index=0,
                x=0,
                y=y,
                width=80,
                height=100,
                stage_x=0,
                stage_y=stage_y,
            ),
            _gray_image(source_pixels[y : y + 100, :]),
        )
    store.close()
    session = DigitalSlideCalibrationSession()
    try:
        session.open(slide_path)
        estimate = session.estimate(
            focus_index=0,
            axis="y",
            target_frame_size=(40, 50),
            target_overlap_percent=40,
            current_stage_step=-4000,
        )
        assert estimate.primary_stride_px == pytest.approx(30.0, abs=0.75)
        assert estimate.cross_axis_drift_px == pytest.approx(0.0, abs=0.75)
        assert estimate.suggested_stage_step == pytest.approx(-4000, abs=150)
    finally:
        session.close()


def test_calibration_rejects_low_texture_pairs(tmp_path: Path) -> None:
    slide_path = tmp_path / "low-texture.fdmslide"
    store = DigitalSlideStore.create(
        slide_path,
        DigitalSlideManifest(1, 160, 80, 100, 80, [0]),
    )
    for x in (0, 60):
        store.write_tile(
            DigitalSlideTile(
                z_index=0,
                x=x,
                y=0,
                width=100,
                height=80,
                stage_x=x * 100,
                stage_y=0,
            ),
            _solid_image(100, 80, "#808080"),
        )
    store.close()
    session = DigitalSlideCalibrationSession()
    try:
        session.open(slide_path)
        with pytest.raises(ValueError, match="低纹理|低置信度|低重叠"):
            session.estimate(
                focus_index=0,
                axis=CALIBRATION_AXIS_X,
                target_frame_size=(100, 80),
                target_overlap_percent=40,
                current_stage_step=6000,
            )
    finally:
        session.close()


def test_calibration_compares_forward_and_reverse_serpentine_passes(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(20260827)
    forward_pixels = rng.integers(0, 255, size=(80, 220), dtype=np.uint8)
    reverse_pixels = rng.integers(0, 255, size=(80, 228), dtype=np.uint8)
    slide_path = tmp_path / "serpentine.fdmslide"
    store = DigitalSlideStore.create(
        slide_path,
        DigitalSlideManifest(1, 220, 160, 100, 80, [0]),
    )
    for x in (0, 60, 120):
        store.write_tile(
            DigitalSlideTile(
                z_index=0,
                x=x,
                y=0,
                width=100,
                height=80,
                stage_x=x * 100,
                stage_y=0,
            ),
            _gray_image(forward_pixels[:, x : x + 100]),
        )
    # SQLite IDs retain the reverse write order of the second snake row.
    for logical_x, actual_x in ((120, 128), (60, 64), (0, 0)):
        store.write_tile(
            DigitalSlideTile(
                z_index=0,
                x=logical_x,
                y=80,
                width=100,
                height=80,
                stage_x=logical_x * 100,
                stage_y=5000,
            ),
            _gray_image(reverse_pixels[:, actual_x : actual_x + 100]),
        )
    store.close()
    session = DigitalSlideCalibrationSession()
    try:
        session.open(slide_path)
        estimate = session.estimate(
            focus_index=0,
            axis=CALIBRATION_AXIS_X,
            target_frame_size=(100, 80),
            target_overlap_percent=40,
            current_stage_step=6000,
        )
        assert estimate.accepted_count == 4
        assert estimate.directional_difference_px > 3.0
        assert any("回程间隙" in warning for warning in estimate.warnings)
        assert estimate.can_apply_stage_step is False
    finally:
        session.close()


def test_capture_preview_is_shared_with_serpentine_plan_and_checks_edges(
    qapp: QApplication,
) -> None:
    settings = AppSettings(
        digital_slide_x_stage_step=-100,
        digital_slide_y_stage_step=50,
        digital_slide_xy_soft_limit=1000,
        digital_slide_reverse_x_axis=True,
    ).normalized_copy()
    with patch("fdm.ui.main_window.AppSettingsIO.load", return_value=settings):
        window = MainWindow()
    try:
        window._digital_slide_range_mode = DIGITAL_SLIDE_RANGE_MODE_GRID
        window._digital_slide_grid_origin_mode = DIGITAL_SLIDE_GRID_ORIGIN_RECORDED
        window._digital_slide_grid_origin_stage = (800, -100)
        preview = window._build_digital_slide_plan_preview(
            settings=settings,
            cols=3,
            rows=2,
            focus_count=2,
            viewport_width_px=100,
            viewport_height_px=80,
            pixel_stride_x=75,
            pixel_stride_y=60,
        )
        assert preview.blockers == ()
        assert preview.image_count == 12
        assert preview.stage_target(2, 1) == (600, -50)
        plan = window._build_digital_slide_capture_plan(
            cols=3,
            rows=2,
            focus_levels=[-10, 10],
            pixel_stride_x=75,
            pixel_stride_y=60,
            plan_preview=preview,
            settings=settings,
        )
        spatial_targets = [
            (item["col"], item["row"], item["stage_x"], item["stage_y"])
            for item in plan[::2]
        ]
        assert spatial_targets == [
            (0, 0, 800, -100),
            (1, 0, 700, -100),
            (2, 0, 600, -100),
            (2, 1, 600, -50),
            (1, 1, 700, -50),
            (0, 1, 800, -50),
        ]

        limit_preview = window._build_digital_slide_plan_preview(
            settings=settings,
            cols=20,
            rows=1,
            focus_count=1,
            viewport_width_px=100,
            viewport_height_px=80,
            pixel_stride_x=75,
            pixel_stride_y=60,
        )
        assert any("软限位" in blocker for blocker in limit_preview.blockers)
        count_preview = window._build_digital_slide_plan_preview(
            settings=settings,
            cols=101,
            rows=100,
            focus_count=2,
            viewport_width_px=100,
            viewport_height_px=80,
            pixel_stride_x=75,
            pixel_stride_y=60,
        )
        assert any("20000" in blocker for blocker in count_preview.blockers)
    finally:
        window.close()


def test_boundary_preview_keeps_signed_stride_and_reports_overshoot(
    qapp: QApplication,
) -> None:
    settings = AppSettings(
        digital_slide_x_stage_step=-100,
        digital_slide_y_stage_step=50,
    ).normalized_copy()
    with patch("fdm.ui.main_window.AppSettingsIO.load", return_value=settings):
        window = MainWindow()
    try:
        window._digital_slide_range_mode = DIGITAL_SLIDE_RANGE_MODE_BOUNDARY
        window._digital_slide_region_edge_marks = {
            "left": 1000,
            "right": 1350,
            "top": 2000,
            "bottom": 2125,
        }
        window._digital_slide_region_bounds = dict(window._digital_slide_region_edge_marks)
        preview = window._build_digital_slide_plan_preview(
            settings=settings,
            cols=4,
            rows=3,
            focus_count=1,
            viewport_width_px=100,
            viewport_height_px=80,
            pixel_stride_x=80,
            pixel_stride_y=40,
        )
        assert preview.origin_stage_x == 1250
        assert preview.stage_delta_x == -100
        assert preview.stage_target(3, 0)[0] == 950
        assert preview.planned_map_bounds["left"] == 950
        assert preview.planned_map_bounds["right"] == 1350
        assert preview.overshoot["left"] == 50
        assert preview.overshoot["bottom"] == 25
        assert preview.warnings
    finally:
        window.close()


def test_workbench_setting_save_synchronizes_active_named_profile(
    qapp: QApplication,
) -> None:
    settings = AppSettings().normalized_copy()
    with patch("fdm.ui.main_window.AppSettingsIO.load", return_value=settings), patch(
        "fdm.ui.main_window.AppSettingsIO.save"
    ) as save:
        window = MainWindow()
        try:
            window._app_settings.digital_slide_capture_tile_codec = "jpeg"
            window._app_settings.digital_slide_capture_jpeg_quality = 83
            assert window._save_app_settings(context="test") is True
            active = next(
                profile
                for profile in window._app_settings.digital_slide_profiles
                if profile.profile_id
                == window._app_settings.digital_slide_active_profile_id
            )
            assert active.values["digital_slide_capture_tile_codec"] == "jpeg"
            assert active.values["digital_slide_capture_jpeg_quality"] == 83
            saved_settings = save.call_args.args[0]
            assert saved_settings.digital_slide_capture_jpeg_quality == 83
        finally:
            window.close()


def test_z_rail_drag_previews_then_emits_one_move_and_bounds_never_move(
    qapp: QApplication,
) -> None:
    rail = DigitalSlideZRangeRail()
    rail.resize(360, 230)
    rail.show()
    qapp.processEvents()
    moves: list[int] = []
    bounds: list[tuple[str, int]] = []
    rail.moveRequested.connect(moves.append)
    rail.boundCommitted.connect(lambda name, value: bounds.append((name, value)))
    rail.set_state(
        soft_limit=5000,
        current_z=0,
        lower_z=-1000,
        upper_z=1000,
        focus_step=100,
        movement_enabled=True,
        bounds_enabled=True,
    )
    rail_x, top, bottom, _rect = rail._rail_geometry()
    current_y = rail._value_to_y(0, top, bottom)
    target_y = rail._value_to_y(1200, top, bottom)
    QTest.mousePress(rail, Qt.MouseButton.LeftButton, pos=QPoint(rail_x, current_y))
    QTest.mouseMove(rail, QPoint(rail_x, target_y))
    assert moves == []
    QTest.mouseRelease(rail, Qt.MouseButton.LeftButton, pos=QPoint(rail_x, target_y))
    assert moves == [1200]

    upper_y = rail._value_to_y(1000, top, bottom)
    new_upper_y = rail._value_to_y(1600, top, bottom)
    QTest.mousePress(
        rail,
        Qt.MouseButton.LeftButton,
        pos=QPoint(rail_x + 10, upper_y),
    )
    QTest.mouseMove(rail, QPoint(rail_x + 10, new_upper_y))
    QTest.mouseRelease(
        rail,
        Qt.MouseButton.LeftButton,
        pos=QPoint(rail_x + 10, new_upper_y),
    )
    assert moves == [1200]
    assert bounds == [("upper", 1600)]
    rail.close()


def test_z_rail_zero_soft_limit_keeps_the_unbounded_command_range(
    qapp: QApplication,
) -> None:
    rail = DigitalSlideZRangeRail()
    rail.set_state(
        soft_limit=0,
        current_z=0,
        lower_z=-50_000,
        upper_z=50_000,
        focus_step=100,
        movement_enabled=True,
    )
    rail.resize(360, 230)
    _rail_x, top, bottom, _rect = rail._rail_geometry()
    assert abs(
        rail._value_to_y(50_000, top, bottom)
        - rail._value_to_y(-50_000, top, bottom)
    ) > 40
    rail.set_target_value(125_000)
    assert rail.target_value() == 125_000
    rail.close()


def test_calibration_short_layout_keeps_long_result_above_guarded_option(
    qapp: QApplication,
) -> None:
    dialog = DigitalSlideCalibrationDialog(AppSettings())
    try:
        dialog._result_label.setPlainText(
            "自动 8/10 对 | X 像素步距 60.00 px\n"
            "交叉轴漂移 +4.20 px | 尺寸 100x80 -> 80x64\n"
            "尺寸换算比例 X 0.8000，Y 0.8000\n"
            "换算 0.01200 px/step | 电机步距建议 8333 steps\n"
            "置信度 0.08\n"
            "注意：自动结果置信度不足；检测到回程间隙。"
        )
        dialog.show()
        dialog.resize(dialog.minimumSize())
        qapp.processEvents()

        assert (
            dialog._result_label.geometry().bottom()
            < dialog._apply_stage_checkbox.geometry().top()
        )
        assert dialog._preview.height() >= dialog._preview.minimumHeight()
    finally:
        dialog.close()


def test_calibration_selection_changes_preserve_manual_offsets(
    qapp: QApplication,
) -> None:
    def descriptor(
        tile_id: int,
        z_index: int,
        x: int,
        y: int,
    ) -> DigitalSlideTileDescriptor:
        return DigitalSlideTileDescriptor(
            tile_id=tile_id,
            z_index=z_index,
            x=x,
            y=y,
            width=100,
            height=80,
            stage_x=x * 100,
            stage_y=y * 100,
            focus_z=z_index * 10,
        )

    x0 = descriptor(1, 0, 0, 0)
    x1 = descriptor(2, 0, 60, 0)
    x2 = descriptor(3, 0, 120, 0)
    y1 = descriptor(4, 0, 0, 50)
    z1_x0 = descriptor(5, 1, 0, 0)
    z1_x1 = descriptor(6, 1, 62, 0)
    pairs = {
        (0, CALIBRATION_AXIS_X): [
            DigitalSlideCalibrationPair(x0, x1, CALIBRATION_AXIS_X),
            DigitalSlideCalibrationPair(x1, x2, CALIBRATION_AXIS_X),
        ],
        (0, CALIBRATION_AXIS_Y): [
            DigitalSlideCalibrationPair(x0, y1, CALIBRATION_AXIS_Y),
        ],
        (1, CALIBRATION_AXIS_X): [
            DigitalSlideCalibrationPair(z1_x0, z1_x1, CALIBRATION_AXIS_X),
        ],
        (1, CALIBRATION_AXIS_Y): [],
    }

    dialog = DigitalSlideCalibrationDialog(AppSettings())
    try:
        dialog._focus_combo.addItem("第 1 层", 0)
        dialog._focus_combo.addItem("第 2 层", 1)
        with (
            patch.object(
                dialog._session,
                "adjacent_pairs",
                side_effect=lambda focus, axis: pairs[(focus, axis)],
            ),
            patch.object(
                dialog._session,
                "read_pair",
                return_value=(
                    _solid_image(100, 80, "#A8B4C6"),
                    _solid_image(100, 80, "#52647C"),
                ),
            ),
        ):
            dialog._refresh_pairs()
            dialog._set_offsets(17, -9)

            dialog._pair_combo.setCurrentIndex(1)
            assert (
                dialog._offset_x_spin.value(),
                dialog._offset_y_spin.value(),
            ) == (17, -9)
            assert (dialog._preview._offset_x, dialog._preview._offset_y) == (17, -9)
            assert "手动当前视场对" in dialog._result_label.toPlainText()

            dialog._axis_combo.setCurrentIndex(
                dialog._axis_combo.findData(CALIBRATION_AXIS_Y)
            )
            assert (
                dialog._offset_x_spin.value(),
                dialog._offset_y_spin.value(),
            ) == (17, -9)
            assert (dialog._preview._offset_x, dialog._preview._offset_y) == (17, -9)

            # Even a temporary selection without an adjacent pair must not
            # destroy the adjustment before the user switches back.
            dialog._focus_combo.setCurrentIndex(1)
            assert dialog._current_pair is None
            assert (
                dialog._offset_x_spin.value(),
                dialog._offset_y_spin.value(),
            ) == (17, -9)
            dialog._axis_combo.setCurrentIndex(
                dialog._axis_combo.findData(CALIBRATION_AXIS_X)
            )
            assert (
                dialog._offset_x_spin.value(),
                dialog._offset_y_spin.value(),
            ) == (17, -9)
            assert (dialog._preview._offset_x, dialog._preview._offset_y) == (17, -9)
            assert "手动当前视场对" in dialog._result_label.toPlainText()
    finally:
        dialog.close()


def test_calibration_preview_separates_view_pan_from_offset_drag(
    qapp: QApplication,
) -> None:
    preview = CalibrationPairPreview()
    preview.resize(640, 320)
    preview.set_pair(
        _solid_image(320, 240, "#A8B4C6"),
        _solid_image(320, 240, "#52647C"),
        nominal_dx=260,
        nominal_dy=0,
    )
    preview.show()
    qapp.processEvents()
    preview.set_view_zoom(2.0)
    offsets: list[tuple[int, int]] = []
    preview.offsetChanged.connect(lambda x, y: offsets.append((x, y)))

    def drag(start: QPointF, end: QPointF) -> None:
        QApplication.sendEvent(
            preview,
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                start,
                start,
                start,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            preview,
            QMouseEvent(
                QEvent.Type.MouseMove,
                end,
                end,
                end,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            preview,
            QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                end,
                end,
                end,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )

    preview.set_pan_mode(True)
    center_before = QPointF(preview._view_center)
    drag(QPointF(300, 150), QPointF(340, 170))
    assert preview._view_center != center_before
    assert offsets == []
    assert (preview._offset_x, preview._offset_y) == (0, 0)

    preview.set_pan_mode(False)
    center_before_offset_drag = QPointF(preview._view_center)
    drag(QPointF(300, 150), QPointF(340, 170))
    assert offsets[-1] == (20, 10)
    assert preview._view_center == center_before_offset_drag
    preview.close()


def test_calibration_preview_supports_wheel_zoom_and_keyboard_view_shortcuts(
    qapp: QApplication,
) -> None:
    preview = CalibrationPairPreview()
    preview.resize(640, 320)
    preview.set_pair(
        _solid_image(320, 240, "#A8B4C6"),
        _solid_image(320, 240, "#52647C"),
        nominal_dx=260,
        nominal_dy=0,
    )
    preview.show()
    preview.setFocus()
    qapp.processEvents()

    fit_zoom = preview.view_zoom()
    QApplication.sendEvent(
        preview,
        QWheelEvent(
            QPointF(320, 160),
            QPointF(320, 160),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        ),
    )
    assert preview.view_mode() == "custom"
    assert preview.view_zoom() > fit_zoom

    QTest.keyClick(preview, Qt.Key.Key_1)
    assert preview.view_mode() == "actual"
    assert preview.view_zoom() == pytest.approx(1.0)
    QTest.keyClick(preview, Qt.Key.Key_0)
    assert preview.view_mode() == "fit"
    QTest.keyPress(preview, Qt.Key.Key_Space)
    assert preview._space_pan is True
    QTest.keyRelease(preview, Qt.Key.Key_Space)
    assert preview._space_pan is False
    preview.close()


def test_calibration_dialog_provides_complete_preview_controls_and_focus_mode(
    qapp: QApplication,
) -> None:
    dialog = DigitalSlideCalibrationDialog(AppSettings())
    try:
        dialog._preview.set_pair(
            _solid_image(320, 240, "#A8B4C6"),
            _solid_image(320, 240, "#52647C"),
            nominal_dx=260,
            nominal_dy=0,
        )
        dialog._update_preview_controls_enabled()
        dialog.resize(dialog.minimumSize())
        dialog.show()
        qapp.processEvents()

        QTest.mouseClick(dialog._actual_button, Qt.MouseButton.LeftButton)
        assert dialog._preview.view_mode() == "actual"
        assert dialog._zoom_spin.value() == 100
        QTest.mouseClick(dialog._zoom_in_button, Qt.MouseButton.LeftButton)
        assert dialog._preview.view_mode() == "custom"
        assert dialog._zoom_spin.value() == 125

        dialog._opacity_slider.setValue(73)
        assert dialog._preview.overlay_opacity() == pytest.approx(0.73)
        split_index = dialog._mode_combo.findData("split")
        dialog._mode_combo.setCurrentIndex(split_index)
        assert dialog._opacity_slider.isEnabled() is False
        assert dialog._split_slider.isEnabled() is True
        dialog._split_slider.setValue(35)
        assert dialog._preview.split_fraction() == pytest.approx(0.35)

        dialog._pan_button.setChecked(True)
        QTest.mouseClick(dialog._reset_view_button, Qt.MouseButton.LeftButton)
        assert dialog._preview.view_mode() == "fit"
        assert dialog._opacity_slider.value() == 52
        assert dialog._split_slider.value() == 50
        assert dialog._pan_button.isChecked() is False

        normal_preview_height = dialog._preview.height()
        dialog._focus_preview_button.setChecked(True)
        qapp.processEvents()
        assert dialog._preview_focus_mode is True
        assert dialog._heading.isHidden() is True
        assert dialog._selection_container.isHidden() is True
        assert dialog._preview.height() > normal_preview_height + 200
        assert dialog._focus_preview_button.text() == "返回校准"

        dialog._preview.escapeRequested.emit()
        qapp.processEvents()
        assert dialog._preview_focus_mode is False
        assert dialog._heading.isVisible() is True
        assert dialog._selection_container.isVisible() is True
        assert dialog._focus_preview_button.text() == "展开预览"
    finally:
        dialog.close()


def test_calibration_preview_toolbar_fits_minimum_dialog_width(
    qapp: QApplication,
) -> None:
    dialog = DigitalSlideCalibrationDialog(AppSettings())
    try:
        dialog.resize(dialog.minimumSize())
        dialog.show()
        qapp.processEvents()

        controls = dialog._preview_controls
        for widget in (
            dialog._zoom_out_button,
            dialog._zoom_spin,
            dialog._zoom_in_button,
            dialog._fit_button,
            dialog._actual_button,
            dialog._center_button,
            dialog._pan_button,
            dialog._reset_view_button,
            dialog._focus_preview_button,
            dialog._opacity_slider,
            dialog._split_slider,
            dialog._split_value_label,
        ):
            top_left = widget.mapTo(controls, QPoint())
            assert top_left.x() >= 0
            assert top_left.y() >= 0
            assert top_left.x() + widget.width() <= controls.width()
            assert top_left.y() + widget.height() <= controls.height()
        assert controls.geometry().bottom() < dialog._preview.geometry().top()
        assert dialog._preview.geometry().bottom() < dialog._button_box.geometry().top()
    finally:
        dialog.close()


def test_compression_minimum_layout_keeps_file_rows_separate(
    qapp: QApplication,
) -> None:
    dialog = DigitalSlideCompressionDialog(AppSettings())
    try:
        dialog.show()
        dialog.resize(dialog.minimumSize())
        qapp.processEvents()

        source_top = dialog._source_edit.mapTo(dialog, QPoint()).y()
        source_bottom = source_top + dialog._source_edit.height()
        target_top = dialog._target_edit.mapTo(dialog, QPoint()).y()
        assert source_bottom < target_top

        target_bottom = target_top + dialog._target_edit.height()
        codec_top = dialog._codec_combo.mapTo(dialog, QPoint()).y()
        assert target_bottom < codec_top
        assert dialog.height() >= 420
    finally:
        dialog.close()
