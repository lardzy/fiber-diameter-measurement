from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QImage

from fdm.models import ImageDocument
from fdm.ui.project_session_controller import ProjectSessionController
from fdm.services.digital_slide_store import (
    DOCUMENT_KIND_DIGITAL_SLIDE,
    DigitalSlideManifest,
    DigitalSlideStore,
    DigitalSlideTile,
)
from fdm.services.motion_control import AXIS_Z, DIR_POS, build_motion_frame


def _solid_image(width: int, height: int, color: str) -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    return image


def test_motion_frame_matches_recovered_protocol_example() -> None:
    frame = build_motion_frame(AXIS_Z, 11200, DIR_POS)
    assert frame.hex(" ").upper() == "AA 55 00 02 00 00 2B C0 00 00 00 01"


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

    assert controller.persist_project_assets(project_path)
    assert (tmp_path / "project.assets" / "slides" / "sample.fdmslide").exists()
