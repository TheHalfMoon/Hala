"""CPU-only authorization checks; no real model mocked as trained."""
import unittest
from types import SimpleNamespace
from h87.research_runtime import require_training_provider
BASE={"HALA_REMOTE_TRAINING_ALLOWED":"1"}
KAGGLE={**BASE,"KAGGLE_KERNEL_RUN_TYPE":"Batch","HALA_NONCOMMERCIAL_RESEARCH":"1"}

class NoncommercialKaggleTrainingTests(unittest.TestCase):
    def test_private_research_kaggle_opt_in(self):
        self.assertEqual(require_training_provider(SimpleNamespace(platform="kaggle"),
            env=KAGGLE,system="Linux",workspace=True),"kaggle_noncommercial_research")
    def test_missing_opt_in_or_wrong_provider_refused(self):
        for extra in ({"HALA_NONCOMMERCIAL_RESEARCH":""},
                      {"KAGGLE_KERNEL_RUN_TYPE":""},
                      {"HALA_REMOTE_TRAINING_ALLOWED":""},
                      {"COLAB_RELEASE_TAG":"active"}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                require_training_provider(SimpleNamespace(platform="kaggle"),
                    env={**KAGGLE,**extra},system="Linux",workspace=True)
    def test_windows_and_non_kaggle_path_refused(self):
        for system,workspace in (("Windows",True),("Linux",False)):
            with self.assertRaises(ValueError):
                require_training_provider(SimpleNamespace(platform="kaggle"),
                    env=KAGGLE,system=system,workspace=workspace)
    def test_standard_colab_remains_eligible(self):
        self.assertEqual(require_training_provider(SimpleNamespace(platform="colab"),
            env=BASE,system="Linux"),"colab_research")
        with self.assertRaises(ValueError):
            require_training_provider(SimpleNamespace(platform="colab"),
                env=KAGGLE,system="Linux")
    def test_arbitrary_provider_blocked(self):
        with self.assertRaises(ValueError):
            require_training_provider(SimpleNamespace(platform="modal"),
                env=BASE,system="Linux")
