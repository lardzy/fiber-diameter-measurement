from __future__ import annotations

from pathlib import Path
import sqlite3
from unittest.mock import patch

import pytest
from PySide6.QtGui import QColor, QImage

from fdm.models import ImageDocument
from fdm.ui.project_session_controller import ProjectSessionController
from fdm.services.digital_slide_store import (
    DOCUMENT_KIND_DIGITAL_SLIDE,
    DIGITAL_SLIDE_TILE_CODEC_JPEG,
    DIGITAL_SLIDE_TILE_CODEC_PNG,
    DigitalSlideManifest,
    DigitalSlideStore,
    DigitalSlideTile,
    compress_slide_file,
    copy_slide_file,
    qimage_to_image_bytes,
)
from fdm.services.motion_control import AXIS_Z, DIR_POS, MotionController, build_motion_frame


def _solid_image(width: int, height: int, color: str) -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    return image


def test_motion_frame_matches_recovered_protocol_example() -> None:
    frame = build_motion_frame(AXIS_Z, 11200, DIR_POS)
    assert frame.hex(" ").upper() == "AA 55 00 02 00 00 2B C0 00 00 00 01"


def test_motion_shutdown_is_idempotent_and_closes_serial() -> None:
    class FakeSerial:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    controller = MotionController()
    serial = FakeSerial()
    controller._serial = serial  # noqa: SLF001
    controller.enabled = True

    first = controller.shutdown("test")
    second = controller.shutdown("test-again")

    assert first.closed is True
    assert first.was_enabled is True
    assert first.error is None
    assert second.closed is True
    assert second.was_enabled is False
    assert serial.closed == 1


def test_motion_shutdown_failure_retains_serial_handle_for_retry() -> None:
    class FailingSerial:
        def __init__(self) -> None:
            self.fail_close = True
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.fail_close:
                raise OSError("port is busy")

    controller = MotionController()
    serial = FailingSerial()
    controller._serial = serial  # noqa: SLF001
    controller.enabled = True

    failed = controller.shutdown("test-failure")

    assert failed.closed is False
    assert failed.error == "port is busy"
    assert controller.enabled is False
    assert controller._serial is serial  # noqa: SLF001

    serial.fail_close = False
    retried = controller.shutdown("test-retry")

    assert retried.closed is True
    assert retried.error is None
    assert controller._serial is None  # noqa: SLF001
    assert serial.close_calls == 2


@pytest.mark.parametrize(
    ("commit_fails", "close_fails", "expected_open"),
    ((True, False, False), (False, True, True), (True, True, True)),
)
def test_digital_slide_store_close_failure_tracks_physical_handle_state(
    commit_fails: bool,
    close_fails: bool,
    expected_open: bool,
) -> None:
    class FailingConnection:
        def commit(self) -> None:
            if commit_fails:
                raise sqlite3.OperationalError("commit failed")

        def close(self) -> None:
            if close_fails:
                raise sqlite3.OperationalError("close failed")

    store = DigitalSlideStore("unused.fdmslide")
    store._conn = FailingConnection()  # type: ignore[assignment]  # noqa: SLF001
    store._image_cache[(0, "cached")] = _solid_image(1, 1, "#112233")  # noqa: SLF001
    store._image_cache_bytes = 4  # noqa: SLF001

    with pytest.raises(sqlite3.OperationalError):
        store.close()

    assert store.is_open() is expected_open
    assert not store._image_cache  # noqa: SLF001
    assert store._image_cache_bytes == 0  # noqa: SLF001


def test_digital_slide_store_writes_manifest_and_renders_viewport(tmp_path: Path) -> None:
    slide_path = tmp_path / "sample.fdmslide"
    store = DigitalSlideStore.create(
        slide_path,
        DigitalSlideManifest(
            version=1,
            width=180,
            height=100,
            viewport_width=100,
            viewport_height=80,
            focus_levels=[-1, 0, 1],
        ),
    )
    try:
        store.write_tile(
            DigitalSlideTile(z_index=1, x=0, y=0, width=100, height=80),
            _solid_image(100, 80, "#FF0000"),
        )
        store.write_tile(
            DigitalSlideTile(z_index=1, x=80, y=0, width=100, height=80),
            _solid_image(100, 80, "#00FF00"),
        )
        manifest = store.read_manifest()
        assert manifest.tile_count == 2
        viewport = store.render_viewport(x=70, y=0, width=100, height=80, z_index=1)
        assert viewport.width() == 100
        assert viewport.height() == 80
        assert QColor(viewport.pixel(5, 10)).red() > 200
        assert QColor(viewport.pixel(50, 10)).green() > 200
        blended = store.render_viewport(x=70, y=0, width=100, height=80, z_index=1, blend_width=48)
        blend_color = QColor(blended.pixel(15, 10))
        assert blend_color.red() > 80
        assert blend_color.green() > 80
    finally:
        store.close()


def test_digital_slide_store_writes_and_reads_jpeg_tiles(tmp_path: Path) -> None:
    slide_path = tmp_path / "jpeg.fdmslide"
    store = DigitalSlideStore.create(
        slide_path,
        DigitalSlideManifest(
            version=1,
            width=40,
            height=30,
            viewport_width=40,
            viewport_height=30,
            focus_levels=[0],
        ),
    )
    try:
        store.write_tile(
            DigitalSlideTile(z_index=0, x=0, y=0, width=40, height=30),
            _solid_image(40, 30, "#3366CC"),
            codec=DIGITAL_SLIDE_TILE_CODEC_JPEG,
            quality=75,
        )
        tiles = list(store.iter_tiles())
        assert len(tiles) == 1
        tile, image, codec, quality = tiles[0]
        assert tile.width == 40
        assert image.width() == 40
        assert codec == DIGITAL_SLIDE_TILE_CODEC_JPEG
        assert quality == 75
        viewport = store.render_viewport(x=0, y=0, width=40, height=30, z_index=0)
        color = QColor(viewport.pixel(10, 10))
        assert color.blue() > 120
    finally:
        store.close()


def test_digital_slide_manifest_rejects_non_finite_json_values(tmp_path: Path) -> None:
    slide_path = tmp_path / "manifest.fdmslide"
    store = DigitalSlideStore.create(
        slide_path,
        DigitalSlideManifest(1, 20, 20, 10, 10, [0], status="ready"),
    )
    try:
        manifest = store.read_manifest()
        manifest.metadata["invalid"] = float("nan")

        with pytest.raises(ValueError):
            store.write_manifest(manifest)

        assert store.read_manifest().status == "ready"
        assert "invalid" not in store.read_manifest().metadata
    finally:
        store.close()


def test_compress_slide_file_writes_copy_without_changing_source(tmp_path: Path) -> None:
    source = tmp_path / "source.fdmslide"
    target = tmp_path / "source_compressed.fdmslide"
    store = DigitalSlideStore.create(
        source,
        DigitalSlideManifest(
            version=1,
            width=40,
            height=30,
            viewport_width=40,
            viewport_height=30,
            focus_levels=[0],
        ),
    )
    try:
        store.write_tile(
            DigitalSlideTile(z_index=0, x=0, y=0, width=40, height=30),
            _solid_image(40, 30, "#00AA55"),
        )
        source_tiles = list(store.iter_tiles())
        assert source_tiles[0][2] == DIGITAL_SLIDE_TILE_CODEC_PNG
    finally:
        store.close()

    compress_slide_file(source, target, codec=DIGITAL_SLIDE_TILE_CODEC_JPEG, quality=80)

    source_store = DigitalSlideStore(source)
    target_store = DigitalSlideStore(target)
    try:
        source_store.open()
        target_store.open()
        assert source_store.tile_count() == 1
        assert target_store.tile_count() == 1
        target_manifest = target_store.read_manifest()
        assert target_manifest.metadata["tile_codec"] == DIGITAL_SLIDE_TILE_CODEC_JPEG
        assert target_manifest.metadata["tile_quality"] == 80
        target_tiles = list(target_store.iter_tiles())
        assert target_tiles[0][2] == DIGITAL_SLIDE_TILE_CODEC_JPEG
        assert target_tiles[0][3] == 80
    finally:
        source_store.close()
        target_store.close()


def test_copy_slide_file_uses_consistent_backup_including_uncheckpointed_wal(tmp_path: Path) -> None:
    source = tmp_path / "working.fdmslide"
    target = tmp_path / "saved.fdmslide"
    source_store = DigitalSlideStore.create(
        source,
        DigitalSlideManifest(
            version=1,
            width=40,
            height=30,
            viewport_width=40,
            viewport_height=30,
            focus_levels=[0],
        ),
    )
    try:
        source_store._connection().execute("PRAGMA wal_autocheckpoint=0")
        source_store.write_tile(
            DigitalSlideTile(z_index=0, x=0, y=0, width=40, height=30),
            _solid_image(40, 30, "#2255AA"),
        )
        source_wal = Path(f"{source}-wal")
        assert source_wal.exists()
        assert source_wal.stat().st_size > 0

        result = copy_slide_file(source, target)

        assert result == target
        with sqlite3.connect(target) as connection:
            assert connection.execute("SELECT COUNT(*) FROM tiles").fetchone()[0] == 1
            assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert not Path(f"{target}-wal").exists()
        assert not Path(f"{target}-shm").exists()
    finally:
        source_store.close()


def test_copy_slide_file_preserves_existing_target_when_replace_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.fdmslide"
    target = tmp_path / "target.fdmslide"
    source_store = DigitalSlideStore.create(
        source,
        DigitalSlideManifest(1, 20, 20, 10, 10, [0], status="source"),
    )
    source_store.close()
    target_store = DigitalSlideStore.create(
        target,
        DigitalSlideManifest(1, 30, 30, 10, 10, [0], status="previous"),
    )
    target_store.close()
    previous = target.read_bytes()

    with patch(
        "fdm.services.digital_slide_store.atomic_replace_file",
        side_effect=OSError("injected replace failure"),
    ):
        with pytest.raises(OSError, match="injected replace failure"):
            copy_slide_file(source, target)

    assert target.read_bytes() == previous
    with sqlite3.connect(target) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert not list(tmp_path.glob(f".{target.name}.*.sqlite.tmp"))


def test_copy_slide_file_refuses_live_wal_target_without_mutating_old_database(tmp_path: Path) -> None:
    source = tmp_path / "source.fdmslide"
    target = tmp_path / "target.fdmslide"
    source_store = DigitalSlideStore.create(
        source,
        DigitalSlideManifest(1, 20, 20, 10, 10, [0], status="source"),
    )
    source_store.close()
    target_store = DigitalSlideStore.create(
        target,
        DigitalSlideManifest(1, 30, 30, 10, 10, [0], status="previous"),
    )
    try:
        target_store._connection().execute("PRAGMA wal_autocheckpoint=0")
        target_store.write_tile(
            DigitalSlideTile(z_index=0, x=0, y=0, width=10, height=10),
            _solid_image(10, 10, "#884422"),
        )
        wal_path = Path(f"{target}-wal")
        shm_path = Path(f"{target}-shm")
        before = {
            target: target.read_bytes(),
            wal_path: wal_path.read_bytes(),
            shm_path: shm_path.read_bytes(),
        }

        with pytest.raises(sqlite3.OperationalError, match="WAL sidecars"):
            copy_slide_file(source, target)

        assert {path: path.read_bytes() for path in before} == before
        assert not list(tmp_path.glob(f".{target.name}.*.sqlite.tmp"))
    finally:
        target_store.close()


def test_copy_slide_file_preserves_existing_target_when_quick_check_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.fdmslide"
    target = tmp_path / "target.fdmslide"
    source_store = DigitalSlideStore.create(
        source,
        DigitalSlideManifest(1, 20, 20, 10, 10, [0], status="source"),
    )
    source_store.close()
    target.write_bytes(b"previous target")

    with patch(
        "fdm.services.digital_slide_store._quick_check_connection",
        side_effect=sqlite3.DatabaseError("injected quick_check failure"),
    ):
        with pytest.raises(sqlite3.DatabaseError, match="injected quick_check failure"):
            copy_slide_file(source, target)

    assert target.read_bytes() == b"previous target"
    assert not list(tmp_path.glob(f".{target.name}.*.sqlite.tmp"))


def test_compress_slide_file_preserves_existing_target_when_conversion_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.fdmslide"
    target = tmp_path / "target.fdmslide"
    source_store = DigitalSlideStore.create(
        source,
        DigitalSlideManifest(1, 20, 20, 20, 20, [0]),
    )
    source_store.write_tile(
        DigitalSlideTile(z_index=0, x=0, y=0, width=20, height=20),
        _solid_image(20, 20, "#00AA55"),
    )
    source_store.close()
    target.write_bytes(b"previous target")

    def fail_progress(_completed: int, _total: int) -> None:
        raise RuntimeError("injected conversion failure")

    with pytest.raises(RuntimeError, match="injected conversion failure"):
        compress_slide_file(source, target, progress_callback=fail_progress)

    assert target.read_bytes() == b"previous target"
    assert not list(tmp_path.glob(f".{target.name}.*.fdmslide.tmp"))


def test_image_document_digital_slide_round_trips_without_sidecar() -> None:
    document = ImageDocument(
        id="slide_1",
        path="slides/sample.fdmslide",
        image_size=(300, 200),
        source_type="project_asset",
        document_kind=DOCUMENT_KIND_DIGITAL_SLIDE,
    )
    document.initialize_runtime_state()
    payload = document.to_dict()
    loaded = ImageDocument.from_dict(payload)
    assert loaded.is_digital_slide()
    assert loaded.document_kind == DOCUMENT_KIND_DIGITAL_SLIDE
    assert not loaded.uses_sidecar()


def test_project_asset_digital_slide_persist_copies_slide_without_image_cache(tmp_path: Path) -> None:
    source = tmp_path / "working.fdmslide"
    store = DigitalSlideStore.create(
        source,
        DigitalSlideManifest(
            version=1,
            width=20,
            height=20,
            viewport_width=10,
            viewport_height=10,
            focus_levels=[0],
        ),
    )
    store.close()
    document = ImageDocument(
        id="slide_1",
        path="slides/sample.fdmslide",
        image_size=(20, 20),
        source_type="project_asset",
        document_kind=DOCUMENT_KIND_DIGITAL_SLIDE,
        metadata={"digital_slide": {"working_path": str(source)}},
    )
    document.initialize_runtime_state()

    class Host:
        project = type("Project", (), {"documents": [document]})()

        @staticmethod
        def _show_project_warning(title: str, message: str) -> None:
            raise AssertionError(f"{title}: {message}")

        @staticmethod
        def _document_display_name(doc: ImageDocument) -> str:
            return doc.path

        @staticmethod
        def _project_asset_image_for_save(doc: ImageDocument):
            raise AssertionError("digital slides must not require an in-memory QImage")

    controller = ProjectSessionController(Host())
    project_path = tmp_path / "project.fdmproj"

    result = controller.persist_project_assets(project_path)
    assert result
    revised_path = result.project.documents[0].path
    assert revised_path.startswith("slides/sample.rev-")
    assert revised_path.endswith(".fdmslide")
    assert (tmp_path / "project.assets" / revised_path).exists()


def test_digital_slide_decode_cache_honors_item_and_byte_limits_and_clears_on_close(tmp_path: Path) -> None:
    slide_path = tmp_path / "cache.fdmslide"
    store = DigitalSlideStore.create(
        slide_path,
        DigitalSlideManifest(
            version=1,
            width=8,
            height=8,
            viewport_width=8,
            viewport_height=8,
            focus_levels=[0],
        ),
    )
    payload = qimage_to_image_bytes(_solid_image(2, 2, "#336699"))
    for tile_id in range(65):
        store._decode_tile_image(tile_id, payload, DIGITAL_SLIDE_TILE_CODEC_PNG)  # noqa: SLF001
    assert len(store._image_cache) == 64  # noqa: SLF001

    store._image_cache_byte_limit = 8  # noqa: SLF001
    store._decode_tile_image(1000, payload, DIGITAL_SLIDE_TILE_CODEC_PNG)  # noqa: SLF001
    assert store._image_cache_bytes <= 8  # noqa: SLF001

    store.close()
    assert not store._image_cache  # noqa: SLF001
    assert store._image_cache_bytes == 0  # noqa: SLF001
