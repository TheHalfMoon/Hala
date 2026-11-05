"""Original H425-to-H421 FREE private T4 launch guard contract; no real GPU."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class H425PrivateH421Continuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (ROOT / "h425" / "owner_continue_to_h421.ps1").read_text("utf8")

    def test_always_replays_real_private_gpu_before_any_stage_or_upload(self):
        s = self.script
        self.assertLess(s.index("& powershell.exe"), s.index("kaggle datasets create"))
        self.assertLess(s.index("& python $StageScript"), s.index("kaggle datasets create"))
        self.assertIn("if ($LASTEXITCODE -ne 0)", s)
        self.assertIn("--sources $Sources", s)
        self.assertIn("--paired $Paired", s)
        self.assertIn("--h421 $H421", s)
        self.assertIn("H425 actual source/LoRA/checkpoints not approved", s)

    def test_dataset_is_always_private_and_provider_private_readback(self):
        s = self.script
        self.assertIn("kaggle datasets create -p $Stage -r skip", s)
        self.assertNotIn("datasets create -u", s)
        self.assertIn("$provider.info.isPrivate -ne $true", s)
        self.assertIn("datasetSlug", s)
        self.assertIn('$meta.isPrivate -ne $true', s)
        self.assertIn("$ExpectedId -notin $eval.dataset_sources", s)

    def test_no_competing_gpu_or_changed_original_384_case_evaluator(self):
        s = self.script
        self.assertLess(s.index("kaggle datasets metadata"), s.index("kaggle kernels push"))
        self.assertIn('$eval.machine_shape -cne "NvidiaTeslaT4"', s)
        self.assertIn("$eval.dataset_sources.Count -ne 8", s)
        self.assertIn("Get-FileHash $EvalCode", s)
        self.assertIn("H421 evaluation folder collision", s)
        self.assertIn("H425 owner model stage collision", s)
        self.assertNotIn("kaggle kernels push -u", s)

if __name__ == "__main__":
    unittest.main()
