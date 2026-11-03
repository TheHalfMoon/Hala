"""CPU-only sanity tests for invented H398 GPU pilot, never real Hala weights."""
import importlib.util
import json
import unittest
from pathlib import Path
from h398.kaggle.h398_synthetic_gpu import SCHEMA,samples,net,evaluate

@unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                     "Optional PyTorch unavailable in stdlib-only CI")
class H398SyntheticTests(unittest.TestCase):
    def test_source_is_deterministically_invented_and_bilingual(self):
        import torch
        a=samples(12,39810)
        b=samples(12,39810)
        self.assertEqual(a[0].shape,(12,3,48,48))
        self.assertTrue(torch.equal(a[0],b[0]))
        self.assertTrue(torch.equal(a[2],b[2]))
        self.assertEqual(set(a[1].tolist()),set(range(6)))
        self.assertTrue(bool(torch.isfinite(a[0]).all()))
        self.assertTrue(bool((a[0]>=0).all() and (a[0]<=1).all()))
    def test_original_tiny_model_training_gradient_exists(self):
        import torch
        x,t,y=samples(12,39820)
        model=net()
        output=model(x,t)
        self.assertEqual(tuple(output.shape),(12,9))
        loss=torch.nn.functional.cross_entropy(output,y)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(any(param.grad is not None and
                            bool(torch.isfinite(param.grad).all())
                            for param in model.parameters()))
        metrics=evaluate(model,(x,t,y),"cpu")
        self.assertEqual(metrics["n"],12)
        self.assertEqual(set(metrics),{"n","all","en","ar","unique_grid_positions"})
        self.assertTrue(SCHEMA.startswith("hala.h398."))
class H398StdlibReceiptTests(unittest.TestCase):
    """Always run source/receipt/privacy checks even without GPU packages."""
    def test_real_kaggle_receipt_sha256_and_synthetic_scope(self):
        from h398.validate_receipt import verify
        result=verify()
        self.assertEqual(result["status"],"SYNTHETIC_GPU_TRAINED_NOT_HALA_MODEL")
        self.assertEqual(result["dev"]["n"],180)

    def test_owner_kernel_has_no_external_data_or_public_access(self):
        metadata=Path(__file__).resolve().parents[1]/"h398"/"kaggle"/"kernel-metadata.json"
        doc=json.loads(metadata.read_text(encoding="utf8"))
        self.assertIs(doc.get("is_private"),True)
        self.assertIs(doc.get("enable_internet"),False)
        self.assertEqual(doc.get("dataset_sources"),[])
        self.assertEqual(doc.get("model_sources"),[])
        self.assertEqual(doc.get("kernel_sources"),[])

if __name__=="__main__":
    unittest.main()
