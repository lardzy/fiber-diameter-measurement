from __future__ import annotations

import unittest

import numpy as np

from fdm.ui.image_parameter_data import (
    count_parameter_range,
    parameter_histogram_snapshot,
    scalar_parameter_samples,
)
from fdm.ui.image_processing_workbench import array_to_raster_plane


class ImageParameterDataTests(unittest.TestCase):
    def test_uint16_histogram_keeps_native_threshold_range(self) -> None:
        raster = array_to_raster_plane(
            np.asarray([[1000, 2000], [3000, 4000]], dtype=np.uint16)
        )
        snapshot = parameter_histogram_snapshot(raster)

        self.assertEqual((snapshot.minimum, snapshot.maximum), (0.0, 65535.0))
        self.assertEqual(snapshot.finite_count, 4)
        self.assertEqual(sum(snapshot.counts), 4)

    def test_float_histogram_excludes_nonfinite_and_honours_roi(self) -> None:
        raster = array_to_raster_plane(
            np.asarray(
                [[-2.0, np.nan, 4.0], [np.inf, 8.0, 10.0]],
                dtype=np.float32,
            )
        )
        roi = np.asarray(
            [[True, True, True], [False, True, False]],
            dtype=np.bool_,
        )
        snapshot = parameter_histogram_snapshot(raster, roi_mask=roi)

        self.assertEqual(snapshot.finite_count, 3)
        self.assertEqual(snapshot.nonfinite_count, 1)
        self.assertEqual(snapshot.masked_out_count, 2)
        self.assertEqual(sum(snapshot.counts), 3)
        self.assertEqual((snapshot.minimum, snapshot.maximum), (-2.0, 8.0))

    def test_luminance_matches_processing_channel_definition(self) -> None:
        raster = array_to_raster_plane(
            np.asarray([[[100, 150, 200, 9]]], dtype=np.uint8)
        )
        samples = scalar_parameter_samples(raster, channel="luminance")

        self.assertAlmostEqual(
            float(samples[0, 0]),
            100 * 0.2126 + 150 * 0.7152 + 200 * 0.0722,
            places=4,
        )
        self.assertEqual(
            int(scalar_parameter_samples(raster, channel="red")[0, 0]),
            100,
        )

    def test_all_rgb_channels_preserve_each_channel_for_levels_statistics(
        self,
    ) -> None:
        raster = array_to_raster_plane(
            np.asarray(
                [
                    [[0, 10, 20], [30, 40, 50]],
                    [[60, 70, 80], [90, 100, 110]],
                ],
                dtype=np.uint8,
            )
        )
        roi = np.asarray(
            [[True, False], [False, True]],
            dtype=np.bool_,
        )

        samples = scalar_parameter_samples(
            raster,
            channel="all_channels",
        )
        snapshot = parameter_histogram_snapshot(
            raster,
            channel="all_channels",
            roi_mask=roi,
        )
        selected, total = count_parameter_range(
            raster,
            lower=10,
            upper=100,
            channel="all_channels",
            roi_mask=roi,
        )

        self.assertEqual(samples.shape, (2, 2, 3))
        self.assertEqual(snapshot.finite_count, 6)
        self.assertEqual(snapshot.masked_out_count, 6)
        self.assertEqual(sum(snapshot.counts), 6)
        self.assertEqual((selected, total), (4, 6))

    def test_float_range_hint_keeps_existing_thresholds_editable(self) -> None:
        raster = array_to_raster_plane(
            np.asarray([[5.0, 7.0, 10.0]], dtype=np.float32)
        )
        snapshot = parameter_histogram_snapshot(
            raster,
            range_hint=(0.0, 12.0),
        )

        self.assertEqual((snapshot.minimum, snapshot.maximum), (0.0, 12.0))
        self.assertEqual(sum(snapshot.counts), 3)

    def test_exact_counts_preserve_existing_threshold_boundaries(self) -> None:
        raster = array_to_raster_plane(
            np.asarray([[0, 1, 2, 3]], dtype=np.uint8)
        )

        self.assertEqual(
            count_parameter_range(
                raster,
                lower=1,
                upper=2,
            ),
            (2, 4),
        )
        self.assertEqual(
            count_parameter_range(
                raster,
                lower=1,
                single_threshold=True,
            ),
            (2, 4),
        )
        self.assertEqual(
            count_parameter_range(
                raster,
                lower=1,
                single_threshold=True,
                invert=True,
            ),
            (2, 4),
        )


if __name__ == "__main__":
    unittest.main()
