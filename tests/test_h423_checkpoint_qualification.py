"""H423 checkpoint rescue contract tests (fake adapter fixture is NOT trained)."""
from hashlib import sha256
import json
from pathlib import Path
import struct
import tempfile
import unittest

from h423 import qualify_h419_checkpoint as r


def fake_checkpoint(root,step=1500):
    root=Path(root)
    p=root/f"step-{step}"
    p.mkdir(parents=True,exist_ok=True)
    tensors={}
    position=0
    for i in range(16):
        length=8192
        tensors[f"base_model.model.layers.{i}.lora_A.weight"]={
            "dtype":"F16","shape":[4096],
            "data_offsets":[position,position+length]}
        position+=length
    metadata=json.dumps(tensors,separators=(",",":")).encode()
    adapter=p/"adapter_model.safetensors"
    adapter.write_bytes(struct.pack("<Q",len(metadata))+metadata+b"\0"*position)
    (p/"adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path":r.BASE,
        "peft_type":"LORA","r":16,"lora_alpha":32,
    }))
    log=root/"original-h419-private-provider.log"
    steps=range(150,step+1,150)
    log.write_text("".join(
       f"H419_REAL_ORIGINAL_30000_NATIVE_GRADIENT {z} / 9452 "
       f"image_conditioned_samples_seen {z*4} loss 2.123456\n"
       for z in steps))
    return p,log


class H423RealCheckpointContract(unittest.TestCase):
    def test_true_original_checkpoint_must_have_full_1500_step_provider_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,log=fake_checkpoint(tmp)
            doc=r.qualify(path,log,"1")
            self.assertEqual(doc["original_model_checkpoint_step"],1500)
            self.assertEqual(doc["verified_original_source_gradient_examples"],6000)
            self.assertEqual(doc["remaining_optimizer_steps_to_original_h419_plan"],7952)
            self.assertEqual(doc["remaining_grad_examples_to_original_plan"],31808)
            self.assertEqual(doc["source_notebook_slug"],r.H419_KERNEL)
            self.assertEqual(doc["real_lora_tensor_count"],16)
            self.assertEqual(doc["checkpoint_adapter_model_sha256"],
                             r.file_sha(path/"adapter_model.safetensors"))
            self.assertFalse(doc["uninterrupted_9452_step_run"])
            self.assertFalse(doc["model_release_qualified"])
            self.assertFalse(doc["source_only_original_heldout_evaluation_completed"])
            self.assertTrue(doc["optimizer_reset_required_for_resumed_training"])
            dest=Path(tmp)/"proof"
            digest=r.store(dest,doc)
            self.assertEqual(digest,sha256((dest/"receipt.json").read_bytes()).hexdigest())
            self.assertEqual((dest/"receipt.sha256").read_text(),digest)
            with self.assertRaises(FileExistsError):
                r.store(dest,doc)

    def test_checkpoint_step_3000_without_original_second_log_segment_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,log=fake_checkpoint(tmp,3000)
            s=log.read_text()
            log.write_text("\n".join(s.splitlines()[:10]))
            with self.assertRaisesRegex(ValueError,"not independently shown all steps"):
                r.qualify(path,log,"1")

    def test_forged_training_progress_micro_example_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,log=fake_checkpoint(tmp)
            s=log.read_text().replace(
               "image_conditioned_samples_seen 600 loss",
               "image_conditioned_samples_seen 200 loss")
            log.write_text(s)
            with self.assertRaisesRegex(ValueError,"inconsistent GPU"):
                r.qualify(path,log,"1")

    def test_duplicate_original_kaggle_progress_rows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,log=fake_checkpoint(tmp)
            row=log.read_text().splitlines()[0]
            log.write_text(log.read_text()+"\n"+row+"\n")
            with self.assertRaisesRegex(ValueError,"duplicated"):
                r.qualify(path,log,"1")

    def test_actual_tensor_header_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,log=fake_checkpoint(tmp)
            file=path/"adapter_model.safetensors"
            raw=file.read_bytes()
            file.write_bytes(raw[:7])
            with self.assertRaises(ValueError):
                r.qualify(path,log,"1")

    def test_fake_config_or_untrusted_step_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,log=fake_checkpoint(tmp)
            cfg=path/"adapter_config.json"
            bad=json.loads(cfg.read_text())
            bad["base_model_name_or_path"]="some-other-model"
            cfg.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,"not exact qualified"):
                r.qualify(path,log,"1")
            with self.assertRaisesRegex(ValueError,"original 1500-step"):
                r.bounded_step(1499)
            with self.assertRaisesRegex(ValueError,"source version"):
                r.qualify(path,log,"latest")

    def test_false_complete_model_claim_never_emitted_by_source_only_qualifier(self):
        source=Path(r.__file__).read_text(encoding="utf8")
        self.assertIn('"model_release_qualified":False',source)
        self.assertIn('"uninterrupted_9452_step_run":False',source)
        self.assertIn('"optimizer_state_saved_in_original_h419":False',source)
        self.assertNotIn("optimizer.step()",source)
        self.assertNotIn(".backward()",source)

if __name__=="__main__":
    unittest.main()
