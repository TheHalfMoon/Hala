"""Offline admission gates for the H49 *real remote* inference producer.

Unit tests never load weights, run inference, provision GPUs or invent scores.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from h49 import remote_infer as infer
from h49 import paired_eval as scorer
from training.artifacts import strict_json

ROOT=Path("h49/assets")

class RemoteInferenceAdmissionTests(unittest.TestCase):
    def test_original_public_dev_identities(self):
        rows=infer.public_cases()
        self.assertEqual(len(rows),32)
        self.assertEqual(len({r["id"] for r in rows}),32)
        self.assertEqual(sum(r["language"]=="ar" for r in rows),16)
        self.assertEqual(sum(r["language"]=="en" for r in rows),16)
        self.assertEqual({r["family"] for r in rows},
                         {"municipal_portal","research_lab"})
        self.assertTrue(all(r["split"]=="dev" for r in rows))
        self.assertTrue(all("target" not in r and "controls" not in r for r in rows))
        self.assertTrue(all((ROOT/r["image"]).is_file() for r in rows))

    def test_remote_only_denies_windows_before_io(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"must-not-be-created"
            with self.assertRaises(RuntimeError):
                infer.run(output,"baseline","0"*40,"UNIT_TEST_PROVIDER_SESSION")
            self.assertFalse(output.exists())

    def test_missing_free_colab_confirmation_is_not_treated_as_free_gpu(self):
        # Deny missing free-tier confirmation on both Linux CI and Windows.
        # Never assume CI runs on Windows or impersonate a paid runtime.
        from unittest.mock import patch
        with patch.dict("os.environ",{"HALA_H49_FREE_COLAB_RESEARCH":"1",
                   "HALA_H49_NO_PAID_RUNTIME_CONFIRMED":"0",
                   "COLAB_RELEASE_TAG":"UNIT_TEST"},clear=False):
            with self.assertRaises(RuntimeError):
                infer.admit_environment()

    def test_original_evaluator_never_imported_to_remote_producer(self):
        code=Path("h49/remote_infer.py").read_text("utf-8")
        self.assertNotIn("evaluator_only.json",code)
        self.assertNotIn("target_index",code)
        self.assertNotIn("expected_index",code)
        self.assertNotIn("model.generate(",code.replace("model.generate(", "GENERATE_CALL",1))
        self.assertEqual(code.count("model.generate("),1)

    def test_exact_metadata_interface_matches_cpu_scorer(self):
        rows=infer.public_cases()
        manifest=strict_json((ROOT/"sft_manifest.json").read_text("utf-8"))
        fields=infer.make_metadata("baseline",rows,manifest,
                                  "T4 | GPU-TEST-ONLY", "UNIT_TEST_RUN_01",None)
        self.assertEqual(set(fields),scorer.META)
        self.assertEqual(fields["schema"],scorer.SCHEMA)
        self.assertEqual(fields["prompt_sha256"],
                         scorer.digest_prompt([r["goal"] for r in rows]))
        self.assertEqual(fields["raw_sha256"],"")
        self.assertEqual(fields["status"],"incomplete")
        self.assertIsNone(fields["adapter_sha256"])

    def test_adapter_sha_and_format_are_verified_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            (directory/"adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path":scorer.MODEL,"peft_type":"LORA"}),
                encoding="utf-8")
            weight=b"FAKE_UNIT_TEST_NOT_MODEL_WEIGHTS"
            (directory/"adapter_model.safetensors").write_bytes(weight)
            expected=hashlib.sha256(weight).hexdigest()
            self.assertEqual(infer.adapter_gate(directory,expected),directory.resolve())
            with self.assertRaises(ValueError):
                infer.adapter_gate(directory,"0"*64)
            (directory/"adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path":"unapproved/model","peft_type":"LORA"}),
                encoding="utf-8")
            with self.assertRaises(ValueError):
                infer.adapter_gate(directory,expected)

    def test_source_sha_and_wrong_run_identity_rejected(self):
        for value in ("bad","a"*39,"0"*41):
            with self.subTest(value=value),self.assertRaises(ValueError):
                infer.source_gate(value)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):
                infer.run(Path(folder)/"none","adapter","0"*40,
                          "UNIT_TEST_SESSION","missing-adapter","0"*64)
            self.assertFalse((Path(folder)/"none").exists())

    def test_no_authentic_provider_qualification_claim(self):
        source=Path("h49/remote_infer.py").read_text("utf-8")
        self.assertIn('"provider_attested":False',source)
        self.assertIn('"model_quality_qualified":False',source)
        self.assertIn('No execution on personal Windows/Mac',source)

if __name__=="__main__":
    unittest.main()
