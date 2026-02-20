from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from scripts.analysis.scan_loss_surface import (
    build_grid,
    choose_output_paths,
    load_completed_points,
    set_trainable_from_values,
    trainable_scalar_count,
)


class TinyLossSurfaceTests(unittest.TestCase):
    def test_build_grid_20x_endpoints(self) -> None:
        values = build_grid(-0.05, 0.05, 20)
        self.assertEqual(len(values), 20)
        self.assertAlmostEqual(values[0], -0.05, places=9)
        self.assertAlmostEqual(values[-1], 0.05, places=9)

    def test_load_completed_points_reads_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "loss_surface.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["i", "j", "p1", "p2", "loss"])
                writer.writeheader()
                writer.writerow({"i": 0, "j": 1, "p1": "0.0", "p2": "0.1", "loss": "1.2"})
                writer.writerow({"i": 4, "j": 3, "p1": "0.2", "p2": "-0.1", "loss": "1.0"})
            done = load_completed_points(csv_path)
            self.assertEqual(done, {(0, 1), (4, 3)})

    def test_choose_output_paths_uses_timestamp_when_existing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "loss_surface.csv").write_text("x", encoding="utf-8")
            paths = choose_output_paths(root_dir=root, resume=False)
            self.assertNotEqual(paths.csv_path.name, "loss_surface.csv")
            self.assertTrue(paths.csv_path.name.startswith("loss_surface_"))
            self.assertTrue(paths.meta_path.name.startswith("meta_"))
            self.assertTrue(paths.best_path.name.startswith("best_point_"))

    def test_set_trainable_from_values_assigns_all_scalars(self) -> None:
        class Toy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p1 = nn.Parameter(torch.zeros(1))
                self.p2 = nn.Parameter(torch.zeros(1))

        toy = Toy()
        params = [(name, p) for name, p in toy.named_parameters() if p.requires_grad]
        self.assertEqual(trainable_scalar_count(params), 2)
        set_trainable_from_values(params, [0.03, -0.04])
        self.assertAlmostEqual(float(toy.p1.item()), 0.03, places=7)
        self.assertAlmostEqual(float(toy.p2.item()), -0.04, places=7)


if __name__ == "__main__":
    unittest.main()
