"""H155 source/manifest governance tests requiring no real data or GPU."""
import inspect
import tempfile
import unittest

from h155.source_train_dev import SPLITS, START_SEED, owner_label
from h155.audit_source_train_dev import audit
from h155.action_contract import H155_SUPPORTED_TASKS


class H155TrainDevSourceGovernance(unittest.TestCase):
    def test_frozen_balanced_new_train_dev_action_source(self):
        self.assertEqual(SPLITS, {"TRAIN": 16, "DEV": 4})
        self.assertEqual(sum(SPLITS.values())*len(H155_SUPPORTED_TASKS), 60)
        self.assertGreater(START_SEED, 204000)

    def test_owner_labels_are_not_represented_as_neural_predictions(self):
        source = inspect.getsource(owner_label)
        self.assertIn("visible_goal", source)
        self.assertIn("page.locator(", source)
        self.assertNotIn("model_predict(", source)
        with self.assertRaises(ValueError):
            owner_label(None, "unknown-task")

    def test_data_admission_fails_closed_without_private_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                audit(folder)


if __name__ == "__main__":
    unittest.main()
