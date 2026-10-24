"""H92 source/adapter admission; synthetic fixtures only (NO model training)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from h92 import research_train as t
from training.checkpoints import publish_checkpoint
from training.artifacts import sha256_file

class H92WarmStartSourceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)/"kaggle-input"
        self.root.mkdir()
        self.bundle=self.root/t.KAGGLE_H87_SLUG/"hala-h87-research-checkpoint"
        self.bundle.parent.mkdir()
        state={"config_sha256":"c"*64,
               "dataset_sha256":t.TRAIN_SHA,
               "manifest_sha256":"d"*64,
               "max_steps":1,"step":1,"metrics":[{"step":1,"loss":2.0,"seconds":1.0}],
               "changed_parameter_tensors":1,"peak_vram_bytes":64,"runtime":{}}
        def write(stage):
            (stage/"adapter").mkdir()
            (stage/"adapter"/"adapter_model.safetensors").write_bytes(b"synthetic H87 seed")
            (stage/"adapter"/"adapter_config.json").write_text("{}",encoding="utf8")
            (stage/"resume.pt").write_bytes(b"synthetic optimizer")
        self.version=publish_checkpoint(self.bundle,state,write)
        self.adapter_sha=sha256_file(self.version/"adapter/adapter_model.safetensors")
        self.ckpt_sha=sha256_file(self.version/"checkpoint.json")

    def test_exact_sha_bound_private_seed_admitted(self):
        with patch.object(t,"H87_ADAPTER_SHA",self.adapter_sha),patch.object(t,"H87_CHECKPOINT_SHA",self.ckpt_sha):
            adapter,sha=t.admit_h87_seed(self.root)
        self.assertEqual(adapter,self.version/"adapter")
        self.assertEqual(sha,self.adapter_sha)

    def test_wrong_seed_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            t.admit_h87_seed(self.root)

    def test_changed_optimizer_bytes_rejected_even_if_adapter_hash_ok(self):
        (self.version/"resume.pt").write_bytes(b"changed")
        with patch.object(t,"H87_ADAPTER_SHA",self.adapter_sha),patch.object(t,"H87_CHECKPOINT_SHA",self.ckpt_sha):
            with self.assertRaises(ValueError):
                t.admit_h87_seed(self.root)

    def test_missing_private_seed_refused(self):
        with self.assertRaises(ValueError):
            t.admit_h87_seed(self.root/"not-mounted")

    def test_research_configuration_is_eight_new_updates(self):
        config=t.research_configuration(Path(__file__).resolve().parents[1])
        self.assertEqual(config["platform"],"kaggle")
        self.assertEqual(config["max_steps"],8)
        self.assertEqual(config["audit_records"],32)
        self.assertEqual(config["lora_rank"],8)
        self.assertEqual(config["precision"],"fp16")
        self.assertEqual(config["gradient_accumulation"],1)

if __name__ == "__main__":
    unittest.main()
