"""H94 CPU checks: research permission, bounded schedule, immutable H92 seed."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from h94 import research_train as h94

ENV = {"KAGGLE_KERNEL_RUN_TYPE": "Batch",
       "HALA_NONCOMMERCIAL_RESEARCH": "1",
       "HALA_REMOTE_TRAINING_ALLOWED": "1"}


class H94SourceBoundTrainingTests(unittest.TestCase):
    def test_original_research_config_only_changes_max_steps(self):
        original = {"platform": "kaggle", "max_steps": 8,
                    "model_id": h94.MODEL_ID, "revision": h94.MODEL_REVISION,
                    "lora_rank": 8, "precision": "fp16",
                    "gradient_accumulation": 1, "batch_size": 1}
        with patch.object(h94, "h92_research_configuration", return_value=original):
            modified = h94.research_configuration("source")
        self.assertEqual(modified["max_steps"], 64)
        self.assertEqual({k: v for k, v in modified.items() if k != "max_steps"},
                         {k: v for k, v in original.items() if k != "max_steps"})

    def test_original_h92_seed_digests_are_pinned(self):
        self.assertEqual(h94.H92_CHECKPOINT_SHA,
                         "6e3a2c678da49001e7309096ff33dcae8df70a0fa9a41c0cfd00b46ec0273d88")
        self.assertEqual(h94.H92_ADAPTER_SHA,
                         "85fa3f6da1dd1b2ddd86d3db795ea19858b3e58bb12305abe2094c2a22e52497")
        self.assertEqual(h94.STEPS, 64)

    def test_live_kaggle_noncommercial_optin_required(self):
        h94.require_research_runtime(env=ENV, operating_system="Linux",
                                     scratch="/kaggle/tmp/h94-study",
                                     output="/kaggle/working/h94-out")
        for change in ({"KAGGLE_KERNEL_RUN_TYPE": None},
                       {"HALA_NONCOMMERCIAL_RESEARCH": "0"},
                       {"HALA_REMOTE_TRAINING_ALLOWED": "0"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                h94.require_research_runtime(
                    env={**ENV, **change}, operating_system="Linux",
                    scratch="/kaggle/tmp/h94-study",
                    output="/kaggle/working/h94-out")

    def test_not_authorized_on_windows_or_outside_kaggle(self):
        for os_name, scratch, output in (
            ("Windows", "/kaggle/tmp/h94-study", "/kaggle/working/h94-out"),
            ("Linux", "/home/user", "/kaggle/working/h94-out"),
            ("Linux", "/kaggle/tmp/h94-study", "/tmp/h94-out")
        ):
            with self.subTest(os_name=os_name), self.assertRaises(ValueError):
                h94.require_research_runtime(
                    env=ENV, operating_system=os_name,
                    scratch=scratch, output=output)

    def test_refuse_preexisting_output(self):
        from pathlib import Path
        with patch.object(h94, "Path") as mocked_path:
            mocked_path.return_value.exists.return_value = True
            with self.assertRaises(ValueError):
                h94.require_research_runtime(
                    env=ENV, operating_system="Linux",
                    scratch="/kaggle/tmp/h94-study",
                    output="/kaggle/working/h94-out")
