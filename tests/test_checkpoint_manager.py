from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tinylora.checkpoint_manager import CheckpointManager


class CheckpointManagerTests(unittest.TestCase):
    def test_save_and_load_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir, keep_last_n=3)
            manager.save_json_checkpoint("run1", step=1, state={"step": 1, "loss": 1.0})
            manager.save_json_checkpoint("run1", step=2, state={"step": 2, "loss": 0.8})

            latest = manager.find_latest_valid_checkpoint("run1")
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.step, 2)

            loaded = manager.load_latest_state("run1")
            self.assertEqual(loaded["step"], 2)

    def test_invalid_latest_falls_back_to_previous(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir, keep_last_n=3)
            manager.save_json_checkpoint("run2", step=1, state={"step": 1})
            second = manager.save_json_checkpoint("run2", step=2, state={"step": 2})

            state_path = second.state_path
            state_path.write_text("{broken_json", encoding="utf-8")

            latest = manager.find_latest_valid_checkpoint("run2")
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.step, 1)

    def test_archives_old_checkpoints_instead_of_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir, keep_last_n=2)
            for step in range(1, 6):
                manager.save_json_checkpoint("run3", step=step, state={"step": step})

            ckpt_root = Path(temp_dir) / "runs" / "run3" / "checkpoints"
            archive_root = Path(temp_dir) / "runs" / "run3" / "archive"
            current = [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("step_")]
            archived = [p for p in archive_root.iterdir() if p.is_dir() and p.name.startswith("step_")]

            self.assertEqual(len(current), 2)
            self.assertGreaterEqual(len(archived), 3)

    def test_manifest_hash_matches_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir, keep_last_n=2)
            record = manager.save_json_checkpoint("run4", step=10, state={"hello": "world"})
            with record.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertIn("files", manifest)
            self.assertTrue(manager.validate_checkpoint_dir(record.checkpoint_dir))

    def test_save_files_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir, keep_last_n=2)
            record = manager.save_files_checkpoint(
                run_id="run5",
                step=11,
                files={
                    "a.bin": b"\x00\x01\x02",
                    "nested/b.txt": b"hello",
                },
            )
            self.assertTrue((record.checkpoint_dir / "a.bin").exists())
            self.assertTrue((record.checkpoint_dir / "nested" / "b.txt").exists())
            self.assertTrue(manager.validate_checkpoint_dir(record.checkpoint_dir))


if __name__ == "__main__":
    unittest.main()
