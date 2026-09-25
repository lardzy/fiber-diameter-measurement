from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fdm.geometry import Line, Point
from fdm.image_processing_models import DisplayTransform
from fdm.models import (
    Calibration, ImageDocument, Measurement, OverlayAnnotation, ProjectState,
    project_assets_root,
)
from fdm.project_io import ProjectIO
from fdm.services import raster_asset_reuse
from fdm.services.raster_asset_reuse import AssetFileStamp, RasterAssetReceipt
from fdm.services.raster_io import (
    RasterMetadata, numpy_to_raster_plane, read_raster_file,
    recommended_native_asset_suffix, write_native_raster_asset,
)
from fdm.ui.image_loader import ImageBatchLoaderWorker, ImageLoadRequest
from fdm.ui.project_session_controller import ProjectSessionController
from fdm.watermark import WatermarkSpec

from test_project_export_controllers import _ProjectHost


class RasterHost(_ProjectHost):
    def __init__(self, root, plane, metadata=None):
        super().__init__(root)
        self.plane = plane
        self.metadata = metadata
        self.warnings = []
        document = ImageDocument(
            id="asset", path="imports/image" + recommended_native_asset_suffix(plane.pixel_type),
            image_size=(plane.width, plane.height), source_type="project_asset",
            raster_pixel_type=plane.pixel_type,
        )
        document.initialize_runtime_state()
        self.project = ProjectState(version="test", documents=[document])

    def _project_asset_raster_for_save(self, document):
        return self.plane, self.metadata

    def _show_project_warning(self, title, message):
        self.warnings.append((title, message))

    def _document_display_name(self, document):
        return document.path


def setup_case(tmp_path, pixels=None, metadata=None):
    if pixels is None:
        pixels = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
    host = RasterHost(tmp_path, numpy_to_raster_plane(pixels), metadata)
    controller = ProjectSessionController(host)
    return host, controller, tmp_path / "project.fdmproj"


def stored_asset(host, project_path):
    return project_assets_root(project_path) / host.project.documents[0].path


def forbid_encoding():
    return patch(
        "fdm.ui.project_session_controller.write_native_raster_asset",
        side_effect=AssertionError("unchanged pixels must not be encoded"),
    )


@pytest.mark.parametrize("dtype,channels", [
    (np.uint8, 1), (np.uint16, 1), (np.float32, 1), (np.uint8, 3), (np.uint8, 4),
])
def test_repeat_save_reuses_pixels_without_encoding_reading_or_hashing(tmp_path, dtype, channels):
    shape = (8, 9) if channels == 1 else (8, 9, channels)
    pixels = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    host, controller, path = setup_case(tmp_path, pixels)
    assert controller.save_project(str(path))
    asset = stored_asset(host, path)
    initial = (asset.read_bytes(), asset.stat().st_mtime_ns)
    with (
        forbid_encoding(),
        patch("fdm.ui.project_session_controller._file_sha256", side_effect=AssertionError("file read")),
        patch.object(raster_asset_reuse, "file_sha256", side_effect=AssertionError("file read")),
        patch("fdm.raster.RasterPlane.sha256", side_effect=AssertionError("pixel scan")),
    ):
        assert controller.save_project(str(path))
        assert controller.save_project(str(path))
    assert (asset.read_bytes(), asset.stat().st_mtime_ns) == initial
    restored = read_raster_file(asset).require_success().plane
    assert restored.data == host.plane.data and restored.pixel_type == host.plane.pixel_type
    assert list(asset.parent.iterdir()) == [asset]


def test_measurement_annotation_calibration_watermark_and_display_changes_keep_asset(tmp_path):
    host, controller, path = setup_case(tmp_path)
    assert controller.save_project(str(path))
    document = host.project.documents[0]
    asset = stored_asset(host, path)
    stamp = AssetFileStamp.read(asset)
    document.measurements.append(Measurement(
        "line", document.id, None, "manual", line_px=Line(Point(1, 1), Point(5, 1)),
    ))
    document.overlay_annotations.append(OverlayAnnotation("text", document.id, "text", "标签"))
    document.calibration = Calibration("manual", 2, "um", "new scale")
    document.watermark = WatermarkSpec(enabled=True, text="水印")
    document.display_transform = DisplayTransform(black_point=10, white_point=200)
    document.mark_session_dirty()
    with forbid_encoding():
        assert controller.save_project(str(path))
    restored = ProjectIO.load(path).documents[0]
    assert restored.measurements[0].id == "line"
    assert restored.overlay_annotations[0].content == "标签"
    assert restored.calibration.pixels_per_unit == 2
    assert restored.watermark.text == "水印"
    assert restored.display_transform.black_point == 10
    assert AssetFileStamp.read(asset) == stamp


def test_verified_import_and_save_as_copy_existing_bytes(tmp_path):
    host, controller, path = setup_case(tmp_path)
    source = tmp_path / "session.png"
    assert write_native_raster_asset(host.plane, source)
    receipt = RasterAssetReceipt.from_verified_file(source, host.plane)
    controller.remember_raster_asset("asset", receipt)
    with forbid_encoding():
        assert controller.save_project(str(path))
        first = stored_asset(host, path)
        renamed = tmp_path / "别的目录" / "另存为.fdmproj"
        assert controller.save_project(str(renamed))
        second = stored_asset(host, renamed)
        assert controller.save_project(str(renamed))
    assert first.read_bytes() == second.read_bytes() == source.read_bytes()
    assert second.parent.parent != first.parent.parent
    assert read_raster_file(second).require_success().plane.data == host.plane.data


def test_pixel_and_encoded_metadata_changes_generate_new_assets(tmp_path):
    host, controller, path = setup_case(tmp_path)
    assert controller.save_project(str(path))
    first = stored_asset(host, path)
    host.plane = numpy_to_raster_plane(np.zeros((10, 10, 3), dtype=np.uint8))
    assert controller.save_project(str(path))
    second = stored_asset(host, path)
    assert second != first and not first.exists()
    assert read_raster_file(second).require_success().plane.data == host.plane.data
    host.metadata = RasterMetadata(dpi_x=123, dpi_y=234, icc_profile=b"test-profile")
    assert controller.save_project(str(path))
    third = stored_asset(host, path)
    assert third != second and not second.exists()
    restored = read_raster_file(third).require_success()
    assert restored.metadata.dpi_x == pytest.approx(123, abs=.1)
    assert restored.metadata.icc_profile == b"test-profile"
    with forbid_encoding():
        assert controller.save_project(str(path))


def test_missing_asset_is_recreated_from_authoritative_pixels(tmp_path):
    host, controller, path = setup_case(tmp_path)
    assert controller.save_project(str(path))
    asset = stored_asset(host, path)
    expected = asset.read_bytes()
    asset.unlink()
    assert controller.save_project(str(path))
    assert asset.read_bytes() == expected


def test_changed_file_attributes_revalidate_but_do_not_reencode_same_content(tmp_path):
    host, controller, path = setup_case(tmp_path)
    assert controller.save_project(str(path))
    asset = stored_asset(host, path)
    old = asset.stat()
    os.utime(asset, ns=(old.st_atime_ns, old.st_mtime_ns + 10_000_000))
    with (
        forbid_encoding(),
        patch.object(raster_asset_reuse, "file_sha256", wraps=raster_asset_reuse.file_sha256) as hashes,
    ):
        assert controller.save_project(str(path))
        assert hashes.call_count == 1
        assert controller.save_project(str(path))
        assert hashes.call_count == 1


def test_corrupted_cached_asset_does_not_silently_succeed_or_overwrite_project(tmp_path):
    host, controller, path = setup_case(tmp_path)
    assert controller.save_project(str(path))
    asset = stored_asset(host, path)
    previous = path.read_bytes()
    bad = bytearray(asset.read_bytes())
    bad[-1] ^= 0xFF  # Same byte count: size alone must not validate reuse.
    asset.write_bytes(bad)
    document = host.project.documents[0]
    document.mark_session_dirty()
    assert not controller.save_project(str(path))
    assert path.read_bytes() == previous and asset.read_bytes() == bytes(bad)
    assert document.dirty_flags.session_dirty
    assert "已损坏" in host.warnings[-1][1]
    assert list(asset.parent.iterdir()) == [asset]


def test_failed_json_does_not_publish_new_receipt_or_lose_previous_asset(tmp_path):
    host, controller, path = setup_case(tmp_path)
    assert controller.save_project(str(path))
    document = host.project.documents[0]
    previous_path, previous_json = document.path, path.read_bytes()
    previous_receipt = controller._raster_asset_receipts[document.id]
    previous_plane = host.plane
    host.plane = numpy_to_raster_plane(np.zeros((10, 10, 3), dtype=np.uint8))
    with patch.object(ProjectIO, "save_payload", side_effect=OSError("injected JSON failure")):
        assert not controller.save_project(str(path))
    assert path.read_bytes() == previous_json and document.path == previous_path
    assert controller._raster_asset_receipts[document.id] is previous_receipt
    assert list(stored_asset(host, path).parent.iterdir()) == [stored_asset(host, path)]
    host.plane = previous_plane
    with forbid_encoding():
        assert controller.save_project(str(path))
    host.plane = numpy_to_raster_plane(np.zeros((10, 10, 3), dtype=np.uint8))
    assert controller.save_project(str(path))


@pytest.mark.parametrize("failure", ["json", "copy", "copy_verification"])
def test_failed_save_as_removes_only_new_files_and_retries_without_encoding(tmp_path, failure):
    host, controller, path = setup_case(tmp_path)
    assert controller.save_project(str(path))
    previous_json = path.read_bytes()
    original = stored_asset(host, path)
    previous_asset = original.read_bytes()
    destination = tmp_path / "second.fdmproj"
    if failure == "json":
        inject = patch.object(ProjectIO, "save_payload", side_effect=OSError("injected"))
    elif failure == "copy":
        inject = patch.object(raster_asset_reuse, "atomic_replace_file", side_effect=OSError("injected"))
    else:
        inject = patch.object(raster_asset_reuse.shutil, "copyfile", side_effect=lambda src, dst: Path(dst).write_bytes(b"broken-copy"))
    with forbid_encoding(), inject:
        assert not controller.save_project(str(destination))
    assert not destination.exists()
    assert not [p for p in project_assets_root(destination).rglob("*") if p.is_file()]
    assert host._project_path == path and path.read_bytes() == previous_json
    assert original.read_bytes() == previous_asset
    with forbid_encoding():
        assert controller.save_project(str(destination))
    assert stored_asset(host, destination).read_bytes() == previous_asset


def test_later_copy_failure_rolls_back_earlier_copies_in_multi_image_project(tmp_path):
    host, controller, path = setup_case(tmp_path)
    second = ImageDocument(
        id="second", path="imports/second.png", image_size=(10, 10),
        source_type="project_asset", raster_pixel_type=host.plane.pixel_type,
    )
    second.initialize_runtime_state()
    host.project.documents.append(second)
    assert controller.save_project(str(path))
    previous = path.read_bytes()
    receipts = dict(controller._raster_asset_receipts)
    destination = tmp_path / "other.fdmproj"
    copy_file = raster_asset_reuse.shutil.copyfile
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("second image copy failed")
        return copy_file(source, target)

    with forbid_encoding(), patch.object(raster_asset_reuse.shutil, "copyfile", fail_second):
        assert not controller.save_project(str(destination))
    assert calls == 2
    assert not [p for p in project_assets_root(destination).rglob("*") if p.is_file()]
    assert controller._raster_asset_receipts == receipts
    assert path.read_bytes() == previous and not destination.exists()
    with forbid_encoding():
        assert controller.save_project(str(destination))


def test_existing_corrupted_save_as_target_is_preserved_and_rejected(tmp_path):
    host, controller, path = setup_case(tmp_path)
    assert controller.save_project(str(path))
    destination = tmp_path / "other.fdmproj"
    target_asset = project_assets_root(destination) / host.project.documents[0].path
    target_asset.parent.mkdir(parents=True)
    target_asset.write_bytes(b"corrupted existing resource")
    with forbid_encoding():
        assert not controller.save_project(str(destination))
    assert target_asset.read_bytes() == b"corrupted existing resource"
    assert not destination.exists() and host._project_path == path


def test_reuse_does_not_bypass_declared_pixel_contract(tmp_path):
    host, controller, path = setup_case(tmp_path)
    assert controller.save_project(str(path))
    before = path.read_bytes()
    host.project.documents[0].image_size = (20, 20)
    with forbid_encoding():
        assert not controller.save_project(str(path))
    assert "图片尺寸" in host.warnings[-1][1]
    assert path.read_bytes() == before


def test_receipts_are_released_on_document_close_and_workspace_reset(tmp_path):
    host, controller, path = setup_case(tmp_path)
    assert controller.save_project(str(path))
    receipt = controller._raster_asset_receipts["asset"]
    controller.remove_document("asset")
    assert not controller._raster_asset_receipts
    controller.remember_raster_asset("asset", receipt)
    controller.clear_unresolved_documents()
    assert not controller._raster_asset_receipts


def test_file_replaced_during_decode_cannot_be_registered_as_old_pixels(tmp_path):
    host, _, _ = setup_case(tmp_path)
    source = tmp_path / "session.png"
    assert write_native_raster_asset(host.plane, source)
    stamp = AssetFileStamp.read(source)
    assert write_native_raster_asset(numpy_to_raster_plane(np.zeros((10, 10, 3), dtype=np.uint8)), source)
    assert RasterAssetReceipt.from_verified_file(source, host.plane, expected_stamp=stamp) is None


def test_batch_reopen_receipt_reuses_loaded_native_pixels(tmp_path):
    host, controller, path = setup_case(tmp_path, np.arange(400, dtype=np.uint16).reshape(20, 20))
    assert controller.save_project(str(path))
    source = stored_asset(host, path)
    request = ImageLoadRequest(str(source), document=host.project.documents[0])
    worker = ImageBatchLoaderWorker([request])
    worker.run()
    assert request.raster_asset_receipt is not None
    host.plane, host.metadata = request.raster_plane, request.raster_metadata
    reopened = ProjectSessionController(host)
    reopened.remember_raster_asset("asset", request.raster_asset_receipt)
    with forbid_encoding():
        assert reopened.save_project(str(path))
    assert stored_asset(host, path) == source


@pytest.mark.parametrize("suffix,all_channels", [("dsx", False), ("dsx", True), ("poir", True)])
def test_device_import_first_save_reopen_and_save_as_never_reencode(
    tmp_path, desktop_application, suffix, all_channels,
):
    from test_device_image_import import dsx_file, poir_file, wait_until
    from fdm.services.device_image_io import inspect_source
    from fdm.ui.device_import_dialog import DeviceReadWorker
    from fdm.ui.main_window import MainWindow

    original = tmp_path / f"instrument.{suffix}"
    (dsx_file if suffix == "dsx" else poir_file)(original)
    window = MainWindow()
    window._session_processed_root = tmp_path / "session"
    channels = inspect_source(original)
    if not all_channels:
        channels = channels[:1]
    worker = DeviceReadWorker("import", channels, asset_root=window._session_processed_root)
    worker.itemReady.connect(window._mount_device_channel)
    worker.run()
    assert len(window.project.documents) == len(channels)
    pixels = {doc.id: window._rasters[doc.id].data for doc in window.project.documents}
    output = tmp_path / "设备项目.fdmproj"
    try:
        with forbid_encoding():
            assert window.save_project(str(output))
            assert window.save_project(str(output))
            original.unlink()
            window._reset_workspace()
            assert window.project_session_controller.load_project_from_path(output)
            wait_until(desktop_application, lambda: not window.is_image_loading())
            for document in window.project.documents:
                assert window._rasters[document.id].data == pixels[document.id]
            assert window.save_project(str(output))
            assert window.save_project(str(tmp_path / "迁移" / "项目.fdmproj"))
    finally:
        window._reset_workspace()
        window.close()
