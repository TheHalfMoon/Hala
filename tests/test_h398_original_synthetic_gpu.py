"""CPU-only sanity tests for invented H398 GPU pilot, never real Hala weights."""
import unittest
from h398.kaggle.h398_synthetic_gpu import SCHEMA,samples,net,evaluate

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
if __name__=="__main__":
    unittest.main()
