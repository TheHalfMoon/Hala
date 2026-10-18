"""Published digests for H25 source-only model-input projection V2."""
import hashlib
import unittest

from h25 import corpus as raw
from h25 import projection_v2 as reviewed

EXPECTED = {
    "projection.json": "cceed2575f3f01995d489c811e2ee38f32272492fc2b533c0197ec0ba27c8ccd",
    "manifest.json": "9668119f01297e7ba81e33ddf900aedf2acb1ef88ffe2684321df811cc89a280",
    "model-inputs.jsonl": "6d4d18d68c5bbf4646d121781dc0fed8dad3cd3fafc017ebd6572eeac6f9301d",
}


class H25PinnedProjectionTests(unittest.TestCase):
    def setUp(self):
        self.projection=reviewed.candidate()

    def test_exact_prospective_projection_digest(self):
        self.assertEqual(raw.digest(self.projection),EXPECTED["projection.json"])

    def test_manifest_digest(self):
        receipt=reviewed.validate(self.projection)
        self.assertEqual(hashlib.sha256(raw.canonical(receipt)).hexdigest(),
                         EXPECTED["manifest.json"])

    def test_model_payload_only_digest(self):
        content=b"".join(raw.canonical(r["model_input"]) for r in self.projection["rows"])
        self.assertEqual(hashlib.sha256(content).hexdigest(),EXPECTED["model-inputs.jsonl"])

    def test_manifest_and_content_remain_source_only(self):
        m=reviewed.validate(self.projection)
        self.assertTrue(m["no_test_qualification"] and m["no_gpu_run"])
        self.assertEqual(len(self.projection["rows"]),42)
        self.assertTrue(all("source_ref" in r and "source_ref" not in r["model_input"]
                            for r in self.projection["rows"]))


if __name__=="__main__":
    unittest.main()
