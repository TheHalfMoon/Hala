"""H134 one-shot sealed evaluation fixed preregistered inputs."""
import unittest
from h134.sealed_gpu import SEALED_MANIFEST_SHA, TRAINED_ADAPTER_SHA
from h134.kaggle_private_train import identify_owner_splits

class H134SealedContracts(unittest.TestCase):
    def test_frozen_sha_and_separate_adapter(self):
        self.assertEqual(SEALED_MANIFEST_SHA,
          "92cbd879ccb7e44a35c9045a589b5cb95e76af965c95d7c5dacb730c6040a02c")
        self.assertEqual(TRAINED_ADAPTER_SHA,
          "711f9dc76cde160818625c9d154c299ad1e098ae48bd554277634ef4b9defdad")
        self.assertNotEqual(SEALED_MANIFEST_SHA,TRAINED_ADAPTER_SHA)
if __name__=="__main__":unittest.main()
