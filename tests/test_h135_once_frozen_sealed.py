"""H135 sealed never used during pilot, pinned to pilot-chosen native mode."""
import unittest
from h135.sealed_gpu import SEALED_MANIFEST_SHA, SELECTED_MODE, SELECTED_DECISION_SHA
from h135.source import SPLITS,SCHEMA

class H135SealedContract(unittest.TestCase):
    def test_exact_once_selected_mode_and_original_sealed_source(self):
        self.assertEqual(SELECTED_MODE,"native_960")
        self.assertEqual(SELECTED_DECISION_SHA,
            "2574aa16e2a4d112793859852fbe5027f9cb15c2699cdb1712946762dc5857fa")
        self.assertEqual(SEALED_MANIFEST_SHA,
            "5dacbcd0dbaaca0f976d0cc342328da9f22a0adf12f3f67401fe88489914ab91")
        self.assertEqual(SPLITS["sealed"][1],18)
        self.assertIn("owner-original",SCHEMA)
if __name__=="__main__":unittest.main()
