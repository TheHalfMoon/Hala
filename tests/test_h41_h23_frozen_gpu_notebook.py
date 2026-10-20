"""Hard-gate original H23 CUDA notebook builder without executing any GPU code."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest

from tools.build_h41_h23_kaggle import (
    H23_SHA,FREEZE_SHA,NOTEBOOK_ID,source_check,make_notebook
)


class H41SourceFreezeTests(unittest.TestCase):
    def test_original_reviewed_h23_sha_and_53_frozen_files(self):
        identity=source_check(Path("."))
        self.assertEqual(identity["historic_source_sha"],H23_SHA)
        self.assertEqual(identity["source_freeze_sha256"],FREEZE_SHA)
        self.assertEqual(identity["frozen_source_files"],53)

    def test_notebook_private_gpu_only_and_never_auto_authorized(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"new"
            r=make_notebook(Path("."),source)
            self.assertFalse(r["opt_in"])
            self.assertFalse(r["submitted"])
            k=json.loads((source/"kernel-metadata.json").read_text())
            n=json.loads((source/"hala-h41-original-h23-cuda.ipynb").read_text())
            self.assertEqual(k["id"],NOTEBOOK_ID)
            self.assertTrue(k["is_private"] and k["enable_gpu"] and k["enable_internet"])
            self.assertFalse(k["enable_tpu"])
            self.assertFalse(n["metadata"]["hala_research_opt_in"])
            self.assertFalse(n["metadata"]["hala_model_qualified"])
            self.assertFalse(n["metadata"]["hala_native_execution"])
            code=["".join(c["source"]) for c in n["cells"] if c["cell_type"]=="code"]
            self.assertEqual(len(code),2)
            for c in code:compile(c,"<H41 Kaggle cell>","exec")
            self.assertIn("Explicit zero-cost research opt-in required",code[0])
            self.assertIn("verify_sources",code[0])
            self.assertIn("H23_SHA" if False else H23_SHA,code[0])
            self.assertIn(FREEZE_SHA,code[0])
            self.assertIn("Pillow==11.3.0",code[0])
            self.assertIn("KAGGLE_KERNEL_RUN_TYPE",code[0])
            self.assertIn("--faults",code[1])
            self.assertIn("actual_cuda_cancel_qualified=False",code[1])
            with self.assertRaises(FileExistsError):
                make_notebook(Path("."),source,research_opt_in=True)

    def test_research_optin_must_be_explicit_and_never_claim_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"new"
            make_notebook(Path("."),source,research_opt_in=True)
            n=json.loads((source/"hala-h41-original-h23-cuda.ipynb").read_text())
            self.assertTrue(n["metadata"]["hala_research_opt_in"])
            self.assertFalse(n["metadata"]["hala_model_qualified"])
            cells=["".join(c["source"]) for c in n["cells"] if c["cell_type"]=="code"]
            self.assertNotIn("Explicit zero-cost research opt-in required",cells[0])
            self.assertNotIn("trust_remote_code=True",cells[0]+cells[1])
            self.assertNotIn("execution_authorized=True",cells[0]+cells[1])


if __name__=="__main__":
    unittest.main()
