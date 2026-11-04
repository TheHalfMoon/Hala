"""CPU qualification for the H424 real checkpoint comparison.

Synthetic tests prove policy/validator logic only, never trained-model quality.
An optional PRIVATE Windows test validates all 7 real model bytes on disk.
"""
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

def load(filename, name):
    spec=importlib.util.spec_from_file_location(name,filename)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class H424GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h=load(ROOT/"h424"/"kaggle"/"h424_checkpoint_eval.py","h424_probe_test")
        cls.s=load(ROOT/"h424"/"stage_private_checkpoints.py","h424_stage_test")

    def test_h419_original_final_must_remain_failed(self):
        self.assertFalse(self.h.h424_gate(76,71,18,30,240))

    def test_valid_source_only_candidate_can_pass_exact_original_gate(self):
        self.assertTrue(self.h.h424_gate(76,72,18,19,240))
        self.assertTrue(self.h.h424_gate(76,76,18,20,240))

    def test_one_below_nonregression_is_rejected(self):
        self.assertFalse(self.h.h424_gate(76,71,18,95,240))

    def test_no_fresh_improvement_is_rejected(self):
        self.assertFalse(self.h.h424_gate(76,140,18,18,240))

    def test_invalid_predictions_are_not_approved(self):
        self.assertFalse(self.h.h424_gate(76,76,18,30,239))

    def test_malformed_or_fabricated_counts_rejected(self):
        for args in [(True,76,18,20,240),(-1,76,18,20,240),
                     (76,200,18,20,240),(76,72,18,97,240),
                     (76,72,18,20,241),(76,72,18,20,240,241)]:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    self.h.h424_gate(*args)

    def test_provider_progress_requires_every_true_150_step_marker(self):
        line=lambda n: (f"H419_REAL_ORIGINAL_30000_NATIVE_GRADIENT "
                        f"{n} / 9452 image_conditioned_samples_seen {n*4} loss 0.1")
        records=[line(n) for n in range(150,9451,150)]
        self.s.verify_progress("\n".join(records))
        for corrupted in ("\n".join(records[1:]),
                          "\n".join(records+[records[-1]]),
                          "\n".join(records[:-1]+[records[-1].replace("37800","37801")])):
            with self.subTest(corruption=corrupted[:40]):
                with self.assertRaises(ValueError):
                    self.s.verify_progress(corrupted)

    def test_source_replay_matches_exact_case_identity(self):
        base={"id":"genuine","family":"original","domain":"slider",
              "language":"en","image_sha256":"a"*64,"correct":True,"valid":True}
        records=[dict(base,id=str(i)) for i in range(240)]
        self.h.h424_replay_match(records,records)
        wrong=[dict(v) for v in records]
        wrong[200]["image_sha256"]="b"*64
        with self.assertRaises(ValueError):
            self.h.h424_replay_match(wrong,records)

    def test_standalone_has_no_training_loop_or_remote_actions(self):
        source=(ROOT/"h424"/"kaggle"/"h424_checkpoint_eval.py").read_text("utf8")
        self.assertNotIn("for step in range(H419_STEPS):",source)
        self.assertNotIn("optimizer.step()",source)
        self.assertIn("is_trainable=False",source)
        self.assertIn("model_release_qualified",source)

    def test_actual_private_original_checkpoints_optional(self):
        stage=os.environ.get("HALA_H424_PRIVATE_STAGE")
        if not stage:
            self.skipTest("Real original Kaggle weights only on owner Windows, never CI")
        root=Path(stage)
        temporary=Path(tempfile.mkdtemp(prefix="hala_h424_local_"))
        self.h.only_input_file=lambda name:root/name
        self.h.H424_INPUT=root
        self.h.H424_TMP=temporary/"private"
        try:
            manifest,receipt=self.h.h424_validate_private_bundle()
            self.assertEqual(len(manifest["models"]),7)
            self.assertEqual(receipt["actual_optimizer_steps"],9452)
            self.assertFalse(receipt["source_only_research_gate"])
        finally:
            shutil.rmtree(temporary)

    def test_actual_kaggle_auto_extracted_zip_optional(self):
        stage=os.environ.get("HALA_H424_PRIVATE_STAGE")
        if not stage:
            self.skipTest("Authentic H419 weights only on owner PC")
        import zipfile
        root=Path(stage)
        temporary=Path(tempfile.mkdtemp(prefix="hala_h424_extract_"))
        try:
            extraction=temporary/self.h.H424_ZIP.removesuffix(".zip")
            with zipfile.ZipFile(root/self.h.H424_ZIP) as z:
                z.extractall(extraction)
            shutil.copy2(root/self.h.H424_MANIFEST,temporary/self.h.H424_MANIFEST)
            self.h.only_input_file=lambda name:temporary/name
            self.h.H424_INPUT=temporary
            self.h.H424_TMP=temporary/"verified"
            manifest,receipt=self.h.h424_validate_private_bundle()
            self.assertEqual(len(manifest["models"]),7)
            self.assertEqual(receipt["adapter_sha256"],self.h.H424_FINAL_SHA)
            self.assertFalse(receipt["source_only_research_gate"])
            shutil.rmtree(temporary/"verified")
            # Kaggle extraction must reject even one unlisted source member.
            (extraction/"unlisted.txt").write_text("tamper")
            with self.assertRaises(ValueError):
                self.h.h424_validate_private_bundle()
        finally:
            shutil.rmtree(temporary)

    def test_scoped_original_parent_ignores_seven_same_named_checkpoint_adapters(self):
        scoped=load(ROOT/"h424"/"kaggle"/"h424_checkpoint_eval.py",
                    "h424_scoped_parent_test")
        temporary=Path(tempfile.mkdtemp(prefix="hala_h424_scoped_"))
        try:
            own=temporary/"h424-dataset"
            own.mkdir()
            (own/scoped.H424_MANIFEST).write_text("{}")
            for name in ("1500","3000","final"):
                sub=own/Path(scoped.H424_ZIP).stem/"models"/name
                sub.mkdir(parents=True)
                (sub/"adapter_model.safetensors").write_bytes(b"checkpoint")
            original=temporary/"original-h417"/"adapter_model.safetensors"
            original.parent.mkdir()
            original.write_bytes(b"original H417 parent")
            scoped.H424_INPUT=temporary
            self.assertEqual(scoped.only_input_file("adapter_model.safetensors"),original)
            self.assertEqual(scoped.only_input_file(scoped.H424_MANIFEST),
                             own/scoped.H424_MANIFEST)
            # A second genuine original parent also fails closed.
            other=temporary/"other-parent"/"adapter_model.safetensors"
            other.parent.mkdir()
            other.write_bytes(b"other")
            with self.assertRaises(ValueError):
                scoped.only_input_file("adapter_model.safetensors")
        finally:
            shutil.rmtree(temporary)
