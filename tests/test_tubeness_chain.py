from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm.analysis_artifacts import (  # noqa: E402
    AnalysisArtifact,
    AnalysisAssetKind,
    AnalysisAssetReference,
    AnalysisDependencySignature,
    AnalysisRegionSnapshot,
    AnalysisSourceDescriptor,
)
from fdm.services.analysis_asset_io import write_safe_analysis_npz  # noqa: E402
from fdm.services.tubeness_chain import (  # noqa: E402
    TubenessChainError,
    build_tubeness_threshold_mask,
    tubeness_response_reference,
)


class TubenessChainTests(unittest.TestCase):
    def _artifact(
        self,
        *,
        source: Path,
        include_best_scale: bool = True,
    ) -> AnalysisArtifact:
        response = np.asarray(
            ((0.0, 0.10, 0.25), (0.50, 0.75, 1.0)),
            dtype=np.float32,
        )
        arrays = {"response": response, "scales": np.asarray((1.0, 2.0))}
        if include_best_scale:
            arrays["best_scale"] = np.asarray(
                ((0.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
                dtype=np.float32,
            )
        info = write_safe_analysis_npz(
            source,
            schema="fdm.tubeness.v1",
            arrays=arrays,
        )
        reference = AnalysisAssetReference(
            kind=AnalysisAssetKind.OTHER,
            path="analysis/source/tubeness.npz",
            sha256=info.sha256,
            media_type="application/x-npz",
            metadata={
                "schema": info.schema,
                "allow_pickle": False,
                "members": {
                    name: {"dtype": dtype, "shape": list(shape)}
                    for name, dtype, shape in info.members
                },
            },
        )
        return AnalysisArtifact(
            id="analysis_tubeness",
            source_document_id="image_1",
            source_pixel_revision=3,
            tool_id="fdm.tubeness",
            tool_version="1",
            parameters={"scales": [1.0, 2.0]},
            source_descriptor=AnalysisSourceDescriptor(
                kind="raster",
                pixel_sha256="a" * 64,
            ),
            dependency_signature=AnalysisDependencySignature(
                calibration={"pixels_per_unit": 2.0, "unit": "um"},
            ),
            region_snapshot=AnalysisRegionSnapshot(
                mask_sha256="b" * 64,
                pixel_center_rule="pixel-center",
                components=1,
                holes=0,
                rings=(),
                source="whole-image",
            ),
            calibration_signature="2|um",
            scalars={"maximum_response": 1.0},
            assets=(reference,),
        )

    def test_threshold_uses_persisted_response_and_records_parent_sha(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "tubeness.npz"
            artifact = self._artifact(source=source)

            result = build_tubeness_threshold_mask(
                artifact,
                source,
                threshold=0.5,
            )

            np.testing.assert_array_equal(
                result.mask,
                np.asarray(
                    ((False, False, False), (True, True, True)),
                    dtype=bool,
                ),
            )
            self.assertFalse(result.mask.flags.writeable)
            self.assertEqual(result.foreground_pixel_count, 3)
            self.assertEqual(result.included_pixel_count, 6)
            self.assertEqual(result.best_scale_minimum, 2.0)
            self.assertEqual(result.best_scale_maximum, 2.0)
            self.assertEqual(
                result.response_asset_sha256,
                tubeness_response_reference(artifact).sha256,
            )

    def test_missing_best_scale_is_rejected_with_a_recalculation_message(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "legacy-tubeness.npz"
            artifact = self._artifact(
                source=source,
                include_best_scale=False,
            )
            with self.assertRaisesRegex(
                TubenessChainError,
                "best_scale.*重新计算",
            ):
                build_tubeness_threshold_mask(
                    artifact,
                    source,
                    threshold=0.5,
                )

    def test_missing_response_asset_is_reported_as_legacy_result(self) -> None:
        artifact = AnalysisArtifact(
            id="legacy",
            source_document_id="image_1",
            source_pixel_revision=0,
            tool_id="fdm.tubeness",
            tool_version="1",
        )
        with self.assertRaisesRegex(
            TubenessChainError,
            "缺少 response / best_scale.*重新计算",
        ):
            tubeness_response_reference(artifact)

    def test_non_positive_or_above_maximum_threshold_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "tubeness.npz"
            artifact = self._artifact(source=source)
            for threshold in (0.0, -0.1, float("nan"), 1.1):
                with self.subTest(threshold=threshold):
                    with self.assertRaises(TubenessChainError):
                        build_tubeness_threshold_mask(
                            artifact,
                            source,
                            threshold=threshold,
                        )


if __name__ == "__main__":
    unittest.main()
