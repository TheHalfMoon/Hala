"""H401 native click receipt anti-fabrication tests, no ML dependency."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from h401.validate_receipt import verify

def valid():
    group=lambda n:{"n":n,"valid_json":n//3,"hits":n//3,"unique_cells":min(n,3)}
    return {"schema":"hala.h401.private-paired-native-invented-qwen3vl.v1",
            "status":"COMPLETE_AUTHOR_SYNTHETIC_NATIVE_PAIRED_DIAGNOSTIC",
            "base_model":"Qwen/Qwen3-VL-2B-Instruct",
            "revision":"89644892e4d85e24eaac8bacfd4f463576704203",
            "adapter_sha256":"8b6d9bdf5df8fd4ecba939a5f454e3deffd5704df3ff41622697e63ef767a537",
            "evaluation_is_independent":False,
            "official_hala_model_qualified":False,
            "source_rights_original_screenshots_approved":False,
            "all_images_are_first_party_generated":True,
            "external_ood":False,
            "model_weights_published":False,
            "sample_count":18,"device":"Tesla T4",
            "base":{"all":group(18),"en":group(9),"ar":group(9)},
            "adapted":{"all":group(18),"en":group(9),"ar":group(9)}}

def put(root,obj):
    encoded=json.dumps(obj,sort_keys=True).encode("utf8")
    (root/"receipt.json").write_bytes(encoded)
    (root/"receipt.sha256").write_text(hashlib.sha256(encoded).hexdigest())

class H401NativeReceiptTests(unittest.TestCase):
    def test_synthetic_only_model_result_is_admitted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            put(root,valid())
            self.assertEqual(verify(root)["base"]["all"]["n"],18)
    def test_false_external_ood_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            obj=valid()
            obj["external_ood"]=True
            put(root,obj)
            with self.assertRaisesRegex(ValueError,"unqualified"):
                verify(root)
    def test_wrong_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            obj=valid()
            obj["base"]["all"]["hits"]=17
            put(root,obj)
            with self.assertRaisesRegex(ValueError,"inconsistent|invalid"):
                verify(root)
    def test_modified_receipt_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            put(root,valid())
            (root/"receipt.json").write_bytes((root/"receipt.json").read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError,"SHA"):
                verify(root)
if __name__=="__main__":
    unittest.main()
