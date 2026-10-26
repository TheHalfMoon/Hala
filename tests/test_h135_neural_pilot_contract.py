"""H135 2x2 study preregistration and private source gate tests."""
import unittest
from h135.ablate_gpu import MODELS,MODES,SCHEMA_OUT
from h135.source import SPLITS,FAMILIES
from h134.sealed_gpu import TRAINED_ADAPTER_SHA

class H135Design(unittest.TestCase):
    def test_all_pilot_cells_preregistered(self):
        self.assertEqual(MODELS,("pretrained","h134_adapter"))
        self.assertEqual(MODES,("downsample_640","native_960"))
        self.assertEqual(len(MODELS)*len(MODES)*SPLITS["pilot"][1]*2,96)
        self.assertEqual(len(FAMILIES),6)
        self.assertEqual(SPLITS["sealed"][1],18)

    def test_neural_source_is_pinned_no_optimizer(self):
        self.assertIn("owner-private",SCHEMA_OUT)
        self.assertEqual(len(TRAINED_ADAPTER_SHA),64)
        self.assertNotEqual(TRAINED_ADAPTER_SHA,"0"*64)
if __name__=="__main__":unittest.main()
