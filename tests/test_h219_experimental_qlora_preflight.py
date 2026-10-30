"""H219 fails closed before downloads, GPU allocation or training.

All tests run with the Python stdlib, no Torch/Transformers/Pillow,
no model inference and no owner/media/ood access.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h219.experimental_free_gpu_qlora_research import (
    ACKNOWLEDGEMENT, MODEL_ID, MODEL_REVISION, RESEARCH_INDICES,
    MIN_VRAM_BYTES, MAX_VRAM_BYTES, source_policy,
    offline_owner_machine_block, check_runtime_request, require,
)


class FreeGpuResearchOnlyPreflightTests(unittest.TestCase):
    def test_research_only_english_arabic_fixed_case_pair(self):
        r = source_policy()
        self.assertEqual(r["kind"], "EXPERIMENT_ONLY_NOT_OFFICIAL_TRAINING")
        self.assertEqual(r["language_assignments"], ("en","ar"))
        self.assertEqual(r["synthetic_examples"], 2)
        self.assertFalse(r["official_train_approved"])
        self.assertFalse(r["independent_dev_sealed"])
        self.assertFalse(r["true_ood_consumed"])
        self.assertEqual(r["optimizer_steps"], 0)

    def test_model_revision_and_provenance_cannot_drift(self):
        self.assertEqual(MODEL_ID,"Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(MODEL_REVISION,
                         "89644892e4d85e24eaac8bacfd4f463576704203")
        self.assertEqual(RESEARCH_INDICES,(0,1))

    def test_no_original_owner_source_may_enter_candidate_batch(self):
        for fake in ((0,64),(0,2),(0,),(1,0),tuple(),list((0,1)),
                     (0,1,95), (0,True)):
            with self.subTest(research=fake):
                with self.assertRaises(ValueError):
                    source_policy(fake)

    def test_source_only_preflight_never_trains(self):
        r=offline_owner_machine_block()
        self.assertEqual(r["optimizer_steps"],0)
        self.assertEqual(r["model_calls"],0)
        self.assertFalse(r["official_train_approved"])

    def test_exact_explicit_acknowledgement_only(self):
        self.assertEqual(type(ACKNOWLEDGEMENT),str)
        self.assertTrue(ACKNOWLEDGEMENT.isascii())
        self.assertIn("NOT_HALA_OFFICIAL",ACKNOWLEDGEMENT)

    def test_requires_one_visible_14_to_25_gib_gpu(self):
        for n,ram in ((0,0),(2,MIN_VRAM_BYTES),(1,MIN_VRAM_BYTES-1),
                      (1,MAX_VRAM_BYTES+1),(True,MIN_VRAM_BYTES)):
            with self.subTest(gpus=n,ram=ram):
                self._deny(gpu_count=n,vram_bytes=ram)

    def test_no_private_hf_auth_tokens_allowed(self):
        self._deny(token_present=True)

    def test_no_paid_or_unconfirmed_session_allowed(self):
        self._deny(free_session=False)

    def test_wrong_acknowledgement_denied(self):
        for value in ("", "yes", "official", "SELF_AUTHORED_RESEARCH_ONLY"):
            with self.subTest(value=value):
                self._deny(acknowledged=value)

    def test_no_local_owner_disk_or_unapproved_output_root(self):
        for p in (Path("relative"),Path("/tmp/Hala-H219"),
                  Path("D:/Hala-H219"),Path("/kaggle/working/other")):
            with self.subTest(path=str(p)):
                self._deny(output_dir=p)

    def test_cannot_execute_training_on_windows_even_if_gpu_flags_forged(self):
        # The owner's Windows machine never has /kaggle/working.
        if not Path("/kaggle/working").is_dir():
            self._deny()

    def test_mock_kaggle_root_can_pass_policy_without_loading_a_model(self):
        # Mock only dir presence; never train or allocate torch resources.
        original=Path.is_dir
        def patched(p):
            if p==Path("/kaggle/working"):
                return True
            return original(p)
        with patch.object(Path,"is_dir",autospec=True,side_effect=patched):
            check_runtime_request(
                acknowledged=ACKNOWLEDGEMENT, free_session=True,
                gpu_count=1, vram_bytes=16*1024**3,
                output_dir=Path("/kaggle/working/Hala-H219-Research-Only"),
                cwd=Path.cwd(),
                token_present=False,
            )

    def test_original_training_can_never_be_implicitly_opened(self):
        from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS,frozenset())

    def _deny(self, **overrides):
        settings={
            "acknowledged":ACKNOWLEDGEMENT,
            "free_session":True,
            "gpu_count":1,
            "vram_bytes":16*1024**3,
            "output_dir":Path("/kaggle/working/Hala-H219-Research-Only"),
            "cwd":Path("/kaggle/working"),
            "token_present":False,
        }
        settings.update(overrides)
        with self.assertRaises(ValueError):
            check_runtime_request(**settings)


if __name__=="__main__":
    unittest.main()
