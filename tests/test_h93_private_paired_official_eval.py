"""CPU-only H93 gate tests; no CUDA allocations or real benchmark claims."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h93 import eval as h93
from h92.research_train import H87_ADAPTER_SHA,TRAIN_SHA
from training.checkpoints import publish_checkpoint
from training.artifacts import sha256_file

class H93SourceBoundEval(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/"private-kaggle-input"
        self.bundle=self.root/h93.H92_KERNEL_SLUG/"hala-h92-private-checkpoint"
        self.bundle.parent.mkdir(parents=True)
        config=h93.research_configuration(Path(__file__).resolve().parents[1])
        signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        signed=hashlib.sha256((signature+H87_ADAPTER_SHA).encode()).hexdigest()
        state={"config_sha256":signed,"dataset_sha256":TRAIN_SHA,
               "manifest_sha256":"f"*64,"step":8,"max_steps":8,
               "metrics":[{"step":i,"loss":1.0,"seconds":1.0} for i in range(1,9)],
               "changed_parameter_tensors":4,"peak_vram_bytes":100,
               "runtime":{}}
        def writer(stage):
            (stage/"adapter").mkdir()
            (stage/"adapter"/"adapter_model.safetensors").write_bytes(b"synthetic weights - NOT Hala")
            (stage/"adapter"/"adapter_config.json").write_text("{}",encoding="utf8")
            (stage/"resume.pt").write_bytes(b"synthetic optimizer - NOT Hala")
        self.version=publish_checkpoint(self.bundle,state,writer)
        self.adapter_sha=sha256_file(self.version/"adapter/adapter_model.safetensors")
        self.checkpoint_sha=sha256_file(self.version/"checkpoint.json")

    def test_private_h92_sha_bound_adapter_admitted(self):
        with patch.object(h93,"H92_ADAPTER_SHA",self.adapter_sha),patch.object(h93,"H92_CHECKPOINT_SHA",self.checkpoint_sha):
            adapter=h93.admit_h92_adapter(self.root,Path(__file__).resolve().parents[1])
        self.assertEqual(adapter,self.version/"adapter")

    def test_real_private_sha_pins_reject_synthetic_lookalike(self):
        with self.assertRaises(ValueError):
            h93.admit_h92_adapter(self.root,Path(__file__).resolve().parents[1])

    def test_tampered_optimizer_refused(self):
        (self.version/"resume.pt").write_bytes(b"corrupted optimizer bytes")
        with patch.object(h93,"H92_ADAPTER_SHA",self.adapter_sha),patch.object(h93,"H92_CHECKPOINT_SHA",self.checkpoint_sha):
            with self.assertRaises(ValueError):
                h93.admit_h92_adapter(self.root,Path(__file__).resolve().parents[1])

    def test_missing_private_kernel_source_refused(self):
        with self.assertRaises(ValueError):
            h93.admit_h92_adapter(self.root/"missing",Path(__file__).resolve().parents[1])

    def test_official_eval_is_diagnostic_not_full_release(self):
        self.assertEqual(len(h93.DOMAINS),4)
        self.assertEqual(h93.MAX_SCENES_PER_DOMAIN,4)
        self.assertEqual(h93.MAX_GENERATED_TOKENS,128)
        self.assertEqual(h93.REPO,"Hcompany/DragOn")
        self.assertEqual(h93.REV,"3451c1fc3543c18fc3361a32bfe1f12cc69b64b5")
        self.assertNotIn("train",h93.SCHEMA)

if __name__=="__main__":
    unittest.main()
