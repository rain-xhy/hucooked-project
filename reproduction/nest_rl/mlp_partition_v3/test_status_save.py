import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .experiment import read, save


class StatusSaveTests(unittest.TestCase):
    def test_transient_windows_lock_retries_atomic_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            save(path, {"update": 7})
            original = Path.replace
            calls = []
            def replace(source, target):
                calls.append(1)
                if len(calls) < 3:
                    self.assertEqual(read(path), {"update": 7})
                    error = PermissionError("locked")
                    error.winerror = 5
                    raise error
                return original(source, target)
            with patch.object(Path, "replace", replace), patch("time.sleep"):
                save(path, {"update": 8})
            self.assertEqual(len(calls), 3)
            self.assertEqual(read(path), {"update": 8})

    def test_persistent_failure_is_bounded_and_preserves_old_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            save(path, {"update": 7})
            error = PermissionError("locked")
            error.winerror = 32
            with patch.object(Path, "replace", side_effect=error) as replace, patch("time.sleep"):
                with self.assertRaises(PermissionError):
                    save(path, {"update": 8})
            self.assertEqual(replace.call_count, 30)
            self.assertEqual(read(path), {"update": 7})
