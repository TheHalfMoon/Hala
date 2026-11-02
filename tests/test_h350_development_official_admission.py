"""H350: the executable historical Mind/Type LoRA entrypoint cannot evade H199.

All fixtures are invented. No model, GPU, private media or legal approval.
"""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.scripts.train_development import (
    require_official_development_training_admission, train,
)

DENIAL = "H346_OFFICIAL_GUI_TRAIN_MANIFEST_NOT_ADMITTED"


class HistoricalDevelopmentAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "invented-bundle"
        self.bundle.mkdir()
        self.raw = b'{"schema":"invented","rights_cleared":true,"official_admitted":true}'
        self.manifest = self.bundle / "manifest.json"
        self.manifest.write_bytes(self.raw)
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def test_self_certification_does_not_admit_historical_trainer(self):
        with patch("h199.native_action_sft.ADMITTED_TRAIN_MANIFEST_SHAS", frozenset()):
            with self.assertRaisesRegex(ValueError, DENIAL):
                require_official_development_training_admission(self.bundle, self.digest)

    def test_only_exact_bytes_from_test_scoped_registry_pass(self):
        with patch("h199.native_action_sft.ADMITTED_TRAIN_MANIFEST_SHAS",
                   frozenset({self.digest})):
            self.assertEqual(
                require_official_development_training_admission(self.bundle, self.digest),
                self.digest,
            )
            self.manifest.write_bytes(self.raw + b" ")
            with self.assertRaisesRegex(ValueError, "DIGEST_MISMATCH"):
                require_official_development_training_admission(self.bundle, self.digest)

    def test_forged_external_digest_not_accepted(self):
        with patch("h199.native_action_sft.ADMITTED_TRAIN_MANIFEST_SHAS",
                   frozenset({self.digest})):
            with self.assertRaisesRegex(ValueError, "DIGEST_MISMATCH"):
                require_official_development_training_admission(self.bundle, "0" * 64)

    def test_symlinked_manifest_rejected_even_if_test_registry_admits(self):
        alias = self.bundle / "manifest-alias.json"
        self.manifest.rename(alias)
        try:
            self.manifest.symlink_to(alias)
        except (OSError, NotImplementedError):
            self.skipTest("OS does not permit test symlinks")
        with patch("h199.native_action_sft.ADMITTED_TRAIN_MANIFEST_SHAS",
                   frozenset({self.digest})):
            with self.assertRaisesRegex(ValueError, "MANIFEST_SYMLINK"):
                require_official_development_training_admission(self.bundle, self.digest)

    def test_oversized_manifest_rejected_without_gpu(self):
        with self.manifest.open("wb") as handle:
            handle.truncate(4 * 1024 * 1024 + 1)
        with self.assertRaises(ValueError):
            require_official_development_training_admission(self.bundle, self.digest)

    def test_real_entrypoint_denies_before_remote_baseline_or_output(self):
        output = self.root / "must-not-exist"
        with patch("h199.native_action_sft.ADMITTED_TRAIN_MANIFEST_SHAS", frozenset()), \
             patch("training.scripts.train_development.remote_guard") as remote, \
             patch("training.scripts.train_development.preflight") as baseline:
            with self.assertRaisesRegex(ValueError, DENIAL):
                train(self.bundle, self.digest, self.root / "missing-baseline",
                      "0" * 64, "0" * 40, self.root / "missing-probe",
                      "0" * 64, "0" * 40, "TheHalfMoon/Hala", output)
            remote.assert_not_called()
            baseline.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
