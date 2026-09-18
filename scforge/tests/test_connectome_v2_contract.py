from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scforge.connectome import (
    REQUIRED_MATRICES,
    build_connectome_command_set,
    derive_count_invnodevol,
)


EXPECTED_MATRICES = (
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)


def _commands():
    return build_connectome_command_set(
        tracks=Path("tracks.tck"),
        nodes=Path("nodes.nii.gz"),
        out_dir=Path("out"),
        sift2_weights=Path("sift2_weights.csv"),
        fa_mif=Path("fa.mif"),
        md_mif=Path("md.mif"),
        rd_mif=Path("rd.mif"),
        ad_mif=Path("ad.mif"),
    )


class ConnectomeCommandContractTests(unittest.TestCase):
    def test_declares_exactly_nine_required_outputs(self) -> None:
        commands = _commands()
        self.assertEqual(REQUIRED_MATRICES, EXPECTED_MATRICES)
        self.assertEqual(tuple(name for name, _ in commands.matrix_outputs), EXPECTED_MATRICES)
        self.assertEqual(len(commands.matrix_outputs), 9)
        self.assertEqual(
            commands.output_path("count_invnodevol"),
            Path("out/SC_AAL_count_invnodevol.csv"),
        )

    def test_raw_count_and_fd_sum_have_distinct_weight_semantics(self) -> None:
        commands = _commands()
        self.assertNotIn("-tck_weights_in", commands.count)
        self.assertEqual(commands.count[commands.count.index("-stat_edge") + 1], "sum")
        self.assertEqual(
            commands.fd_sum[commands.fd_sum.index("-tck_weights_in") + 1],
            "sift2_weights.csv",
        )
        self.assertEqual(commands.fd_sum[commands.fd_sum.index("-stat_edge") + 1], "sum")

    def test_fd_sum_fails_closed_without_sift2_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "sift2_weights is required"):
            build_connectome_command_set(
                tracks=Path("tracks.tck"),
                nodes=Path("nodes.nii.gz"),
                out_dir=Path("out"),
            )

    def test_length_commands_are_unweighted_separate_edge_means(self) -> None:
        commands = _commands()
        self.assertIn("-scale_length", commands.len_mean)
        self.assertNotIn("-scale_invlength", commands.len_mean)
        self.assertIn("-scale_invlength", commands.invlen_mean)
        self.assertNotIn("-scale_length", commands.invlen_mean)
        for command in (commands.len_mean, commands.invlen_mean):
            self.assertNotIn("-tck_weights_in", command)
            self.assertEqual(command[command.index("-stat_edge") + 1], "mean")

    def test_tensor_commands_use_two_level_unweighted_means(self) -> None:
        commands = _commands()
        metrics = (
            (commands.fa_sample, commands.fa_mean, "fa.mif", "out/fa_mean.tsf"),
            (commands.md_sample, commands.md_mean, "md.mif", "out/md_mean.tsf"),
            (commands.rd_sample, commands.rd_mean, "rd.mif", "out/rd_mean.tsf"),
            (commands.ad_sample, commands.ad_mean, "ad.mif", "out/ad_mean.tsf"),
        )
        for sample, edge_mean, metric_image, sample_file in metrics:
            self.assertEqual(
                sample,
                ("tcksample", "tracks.tck", metric_image, sample_file, "-stat_tck", "mean"),
            )
            self.assertNotIn("-tck_weights_in", edge_mean)
            self.assertEqual(edge_mean[edge_mean.index("-scale_file") + 1], sample_file)
            self.assertEqual(edge_mean[edge_mean.index("-stat_edge") + 1], "mean")

    def test_command_plan_contains_no_padding_or_clipping(self) -> None:
        tokens = [token.lower() for row in _commands().to_records() for token in row["command"].split()]
        self.assertFalse(any("pad" in token or "clip" in token for token in tokens))
        self.assertNotIn("-scale_invnodevol", tokens)


class CountInverseNodeVolumeTests(unittest.TestCase):
    def test_uses_raw_count_and_physical_node_volumes(self) -> None:
        count = np.array(
            [
                [0, 10, 4],
                [10, 0, 2],
                [4, 2, 0],
            ],
            dtype=np.int64,
        )
        volumes = np.array([2.0, 6.0, 10.0])
        observed = derive_count_invnodevol(count, volumes)
        expected = np.array(
            [
                [0.0, 2.5, 2.0 / 3.0],
                [2.5, 0.0, 0.25],
                [2.0 / 3.0, 0.25, 0.0],
            ]
        )
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)
        np.testing.assert_array_equal(count, np.array([[0, 10, 4], [10, 0, 2], [4, 2, 0]]))
        np.testing.assert_array_equal(volumes, np.array([2.0, 6.0, 10.0]))

    def test_rejects_shape_mismatch_instead_of_padding_or_cropping(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact square"):
            derive_count_invnodevol(np.zeros((2, 3)), np.ones(2))
        with self.assertRaisesRegex(ValueError, "exactly one value"):
            derive_count_invnodevol(np.zeros((3, 3)), np.ones(2))

    def test_rejects_invalid_raw_count_instead_of_sanitising(self) -> None:
        invalid_counts = (
            np.array([[0.0, 1.5], [1.5, 0.0]]),
            np.array([[0.0, -1.0], [-1.0, 0.0]]),
            np.array([[0.0, 1.0], [2.0, 0.0]]),
            np.array([[1.0, 2.0], [2.0, 0.0]]),
            np.array([[0.0, np.nan], [np.nan, 0.0]]),
        )
        for count in invalid_counts:
            with self.subTest(count=count), self.assertRaises(ValueError):
                derive_count_invnodevol(count, np.ones(2))

    def test_rejects_nonphysical_node_volumes(self) -> None:
        count = np.array([[0, 1], [1, 0]])
        for volumes in (np.array([1.0, 0.0]), np.array([1.0, -2.0]), np.array([1.0, np.nan])):
            with self.subTest(volumes=volumes), self.assertRaisesRegex(ValueError, "positive physical volumes"):
                derive_count_invnodevol(count, volumes)


if __name__ == "__main__":
    unittest.main()
