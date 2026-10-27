"""Regression: create-only H155 source gate precedes optional Playwright import."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h155.source_train_dev import collect


class SourceWriteOnceGate(unittest.TestCase):
    def test_consumed_corpus_is_rejected_without_browser_dependency(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "private-h155" / "source-train-dev-v1"
            folder.mkdir(parents=True)
            with patch("h155.source_train_dev.admit_upstream",
                       return_value=Path(root)):
                with self.assertRaises(FileExistsError):
                    collect(root, root)


if __name__ == "__main__":
    unittest.main()
