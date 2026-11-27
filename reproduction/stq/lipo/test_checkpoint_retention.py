"""Boundary checks for LIPO checkpoint retention without environment dependencies."""

from pathlib import Path
import tempfile
import unittest

from lipo.train import CheckpointSchedule, _save_archive_if_absent


class CheckpointRetentionTests(unittest.TestCase):
    def test_step_thresholds_align_to_actual_population_update_boundaries(self):
        schedule = CheckpointSchedule(250_000)
        self.assertEqual(schedule.crossed(230_400), ())
        self.assertEqual(schedule.crossed(460_800), (250_000,))
        self.assertEqual(schedule.crossed(691_200), (500_000,))

    def test_existing_archive_is_skipped_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "step_460800.pt"
            archive.write_bytes(b"original checkpoint")
            save_calls = []

            saved = _save_archive_if_absent(archive, lambda path: save_calls.append(path))

            self.assertFalse(saved)
            self.assertEqual(archive.read_bytes(), b"original checkpoint")
            self.assertEqual(save_calls, [])

            new_archive = Path(directory) / "step_691200.pt"
            saved = _save_archive_if_absent(new_archive, lambda path: path.write_bytes(b"new checkpoint"))
            self.assertTrue(saved)
            self.assertEqual(new_archive.read_bytes(), b"new checkpoint")


if __name__ == "__main__":
    unittest.main()
