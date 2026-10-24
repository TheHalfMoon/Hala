"""H87 provider gate regression; preserves noncommercial Kaggle research only."""
import unittest
from types import SimpleNamespace
from h87.research_runtime import require_training_provider
BASE={"HALA_REMOTE_TRAINING_ALLOWED":"1"}
KAGGLE={**BASE,"KAGGLE_KERNEL_RUN_TYPE":"Batch","HALA_NONCOMMERCIAL_RESEARCH":"1"}

class NoncommercialKaggleTrainingTests(unittest.TestCase):
    def test_explicit_kaggle_noncommercial_research_is_admitted(self):
        self.assertEqual(require_training_provider(SimpleNamespace(platform="kaggle"),
            env=KAGGLE,system="Linux",workspace=True),"kaggle_noncommercial_research")
    def test_kaggle_runtime_with_inherited_colab_release_tag_is_still_kaggle(self):
        self.assertEqual(require_training_provider(SimpleNamespace(platform="kaggle"),
            env={**KAGGLE,"COLAB_RELEASE_TAG":"inherited"},system="Linux",workspace=True),
            "kaggle_noncommercial_research")
    def test_absent_runtime_or_noncommercial_marker_refused(self):
        for changed in ({"KAGGLE_KERNEL_RUN_TYPE":""},
                        {"HALA_NONCOMMERCIAL_RESEARCH":""},
                        {"HALA_REMOTE_TRAINING_ALLOWED":""}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                require_training_provider(SimpleNamespace(platform="kaggle"),
                    env={**KAGGLE,**changed},system="Linux",workspace=True)
    def test_windows_or_absent_kaggle_workspace_refused(self):
        for system,workspace in (("Windows",True),("Linux",False)):
            with self.assertRaises(ValueError):
                require_training_provider(SimpleNamespace(platform="kaggle"),
                    env=KAGGLE,system=system,workspace=workspace)
    def test_kaggle_marked_runtime_cannot_pass_as_colab(self):
        with self.assertRaises(ValueError):
            require_training_provider(SimpleNamespace(platform="colab"),
                env=KAGGLE,system="Linux")
        self.assertEqual(require_training_provider(SimpleNamespace(platform="colab"),
            env=BASE,system="Linux"),"colab_research")
    def test_unapproved_provider_denied(self):
        with self.assertRaises(ValueError):
            require_training_provider(SimpleNamespace(platform="modal"),
                env=BASE,system="Linux")
