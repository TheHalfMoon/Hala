"""H139 OOD inference kernel must be private, frozen and optimizer-free."""
import json
import unittest
from pathlib import Path
from h139.blind_gpu import (
    ADAPTER_SHA,ARCHIVE_SHA,REDACTED_MANIFEST_SHA,OWNER_MANIFEST_SHA,
)
from h139.owner_replay import ARMS,NEURAL_SCHEMA
from h139.blind_export import SCHEMA as INPUT_SCHEMA


class H139BlindGPUContracts(unittest.TestCase):
    def test_exact_immutable_inputs_and_neural_arms(self):
        self.assertEqual(ADAPTER_SHA,"c46ec0abf4d9fed545ec85867737ca19dc59f54f84e847b90304d4af6b052180")
        self.assertEqual(ARCHIVE_SHA,"7d3ddcb6db22c4fc1974c099a318e8921adcdfc514aae8de0bd37444ed055fc8")
        self.assertEqual(REDACTED_MANIFEST_SHA,"3432b51aa505688b513210ea3ae4dc34ff3d514790f671b532fc84cbc3c83c73")
        self.assertEqual(OWNER_MANIFEST_SHA,"c26c0b1c850bf4fc30c7e04bc16e547f89d69269174751f26ea2b6b824acfac9")
        self.assertEqual(ARMS,("base_pretrained","h139_adapter"))
        self.assertNotEqual(NEURAL_SCHEMA,INPUT_SCHEMA)

    def test_private_t4_only_mount_contract(self):
        root=Path(__file__).parent.parent/"h139"/"blind_kaggle"
        metadata=json.loads((root/"kernel-metadata.json").read_text("utf8"))
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(len(metadata["dataset_sources"]),2)
        self.assertTrue(any("blind8-redacted" in item for item in metadata["dataset_sources"]))
        self.assertTrue(any("frozen-64step-adapter" in item for item in metadata["dataset_sources"]))
        self.assertFalse(any("train64" in x or "dev16" in x for x in metadata["dataset_sources"]))


if __name__=="__main__":
    unittest.main()
