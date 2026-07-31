from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.analysis_artifacts import (
    AnalysisArtifact,
    AnalysisAssetKind,
    AnalysisAssetReference,
)
from fdm.models import project_assets_root
from fdm.services.analysis_asset_io import (
    validate_analysis_asset_reference,
    write_safe_analysis_npz,
)
from fdm.ui.project_session_controller import ProjectSessionController

from test_project_export_controllers import _ProjectHost


def _reference(
    source: Path,
    *,
    path: str = "analysis/result/mask.npz",
) -> AnalysisAssetReference:
    array = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    metadata = {
        "schema": "fdm.test-mask.v1",
        "allow_pickle": False,
        "members": {
            "mask": {
                "dtype": "uint8",
                "shape": [2, 2],
            }
        },
    }
    info = write_safe_analysis_npz(
        source,
        schema="fdm.test-mask.v1",
        arrays={"mask": array},
        metadata=metadata,
    )
    return AnalysisAssetReference(
        kind=AnalysisAssetKind.MASK,
        path=path,
        sha256=info.sha256,
        media_type="application/x-npz",
        metadata=metadata,
    )


def _artifact(document_id: str, reference: AnalysisAssetReference) -> AnalysisArtifact:
    return AnalysisArtifact(
        id="analysis-mask",
        source_document_id=document_id,
        source_pixel_revision=0,
        tool_id="fdm.particles",
        tool_version="1",
        parameters={"threshold": 127},
        scalars={"accepted_count": 1},
        assets=(reference,),
    )


class _AnalysisAssetHost(_ProjectHost):
    def __init__(self, tmp_dir: Path) -> None:
        super().__init__(tmp_dir)
        self.analysis_sources: dict[str, Path] = {}
        self.warnings: list[tuple[str, str]] = []

    def _analysis_asset_source_for_save(
        self,
        reference: AnalysisAssetReference,
    ) -> Path | None:
        return self.analysis_sources.get(reference.path)

    def _show_project_warning(self, title: str, message: str) -> None:
        self.warnings.append((title, message))


def test_analysis_npz_is_verified_revisioned_and_rewritten_in_project() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "session" / "mask.npz"
        reference = _reference(source)
        host = _AnalysisAssetHost(root)
        host.analysis_sources[reference.path] = source
        host.project.analysis_artifacts.append(
            _artifact(host.project.documents[0].id, reference)
        )
        controller = ProjectSessionController(host)
        target = root / "analysis-demo.fdmproj"

        result = controller.save_project(str(target))

        assert result.success
        assert not host.warnings
        payload = json.loads(target.read_text(encoding="utf-8"))
        saved_reference = AnalysisAssetReference.from_dict(
            payload["analysis_artifacts"][0]["assets"][0]
        )
        assert saved_reference.path.startswith("analysis/result/mask.rev-")
        saved_path = project_assets_root(target) / saved_reference.path
        validate_analysis_asset_reference(saved_path, saved_reference)
        assert host.project.analysis_artifacts[0].assets[0] == saved_reference


def test_save_as_copies_loaded_analysis_asset_without_session_source() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "session" / "mask.npz"
        reference = _reference(source)
        host = _AnalysisAssetHost(root)
        host.analysis_sources[reference.path] = source
        host.project.analysis_artifacts.append(
            _artifact(host.project.documents[0].id, reference)
        )
        controller = ProjectSessionController(host)
        first = root / "first.fdmproj"
        second = root / "second.fdmproj"
        assert controller.save_project(str(first)).success
        host.analysis_sources.clear()
        source.unlink()

        result = controller.save_project(str(second))

        assert result.success
        saved_reference = host.project.analysis_artifacts[0].assets[0]
        validate_analysis_asset_reference(
            project_assets_root(second) / saved_reference.path,
            saved_reference,
        )


def test_missing_analysis_asset_does_not_replace_existing_project() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        host = _AnalysisAssetHost(root)
        controller = ProjectSessionController(host)
        target = root / "stable.fdmproj"
        assert controller.save_project(str(target)).success
        before = target.read_bytes()
        missing_reference = AnalysisAssetReference(
            kind=AnalysisAssetKind.MASK,
            path="analysis/missing/mask.npz",
            sha256="1" * 64,
            media_type="application/x-npz",
            metadata={
                "schema": "fdm.test-mask.v1",
                "allow_pickle": False,
                "members": {
                    "mask": {
                        "dtype": "uint8",
                        "shape": [2, 2],
                    }
                },
            },
        )
        host.project.analysis_artifacts.append(
            _artifact(host.project.documents[0].id, missing_reference)
        )

        result = controller.save_project(str(target))

        assert not result.success
        assert target.read_bytes() == before
        assert host.warnings
