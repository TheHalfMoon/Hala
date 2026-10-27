"""H141 independent visual model probe cannot consume H140 gold or BLIND."""
import json
import tempfile
import unittest
from pathlib import Path

from h141.gpu_visible_probe import (
    EVAL_INPUT_SHA,EVAL_MANIFEST_SHA,ADAPTER_ZIP_SHA,V2_ADAPTER_SHA,
    visible_prompt
)
from h141.kaggle_visible_probe.kaggle_private_infer import identify_private_source,SOURCE_SHA

class H141PrivateVisibleNeuralProbeTests(unittest.TestCase):
    def test_visible_heading_instruction_never_requests_invisible_drop_target(self):
        task="Move owner task 48000 from To Do to Doing"
        a=visible_prompt(task,"start")
        b=visible_prompt(task,"visible_column_heading")
        self.assertIn(task,a)
        self.assertIn(task,b)
        self.assertIn("DESTINATION COLUMN HEADER",b)
        self.assertIn("not any invisible",b)
        with self.assertRaises(ValueError):
            visible_prompt(task,"hidden_drop")
        for value in (EVAL_INPUT_SHA,EVAL_MANIFEST_SHA,ADAPTER_ZIP_SHA,V2_ADAPTER_SHA):
            self.assertEqual(len(value),64)
        self.assertEqual(len(SOURCE_SHA),40)

    def test_owner_data_mount_exactly_one_probe_no_h140_blind(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,"unavailable"):
                identify_private_source(root)
            ds=root/"hala-h141-private-visible-probe-v1"
            ds.mkdir()
            (ds/"model-visible-probe16-redacted-v1.zip").write_bytes(b"fixture only")
            (ds/"h140-private-original-v2-rank8-lora.zip").write_bytes(b"fixture only")
            one,two=identify_private_source(root)
            self.assertTrue(one.name.endswith("redacted-v1.zip"))
            self.assertIn("rank8",two.name)
            (root/"__kaggle_system_data").mkdir()
            self.assertEqual(identify_private_source(root),(one,two))
            (root/"extra-blind-upload").mkdir()
            with self.assertRaisesRegex(ValueError,"forbidden"):
                identify_private_source(root)

    def test_private_kernel_is_free_t4_only(self):
        meta=json.loads((Path(__file__).parent.parent/"h141"/
                         "kaggle_visible_probe"/"kernel-metadata.json").read_text("utf8"))
        self.assertTrue(meta["is_private"])
        self.assertTrue(meta["enable_gpu"])
        self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(meta["dataset_sources"],[
            "abdulazizshehri/hala-h141-private-visible-probe-v1"])
        self.assertEqual(meta["model_sources"],[])
        self.assertEqual(meta["kernel_sources"],[])

if __name__=="__main__":
    unittest.main()
