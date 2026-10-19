"""H28 pair-generator reproducibility and oracle isolation, CPU only."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h28 import contrasts
from tools import build_h28_contrast_plan


class H28ContrastPlanTests(unittest.TestCase):
    def test_reproducible_balanced_messages_no_source_references(self):
        model, evaluator, manifest = build_h28_contrast_plan.prepare()
        self.assertEqual(len(model.splitlines()), 72)
        self.assertEqual(len(evaluator.splitlines()), 72)
        self.assertEqual(manifest["source_sha256"], contrasts.PINNED_SOURCE_SHA256)
        self.assertEqual(manifest["model_messages_sha256"], hashlib.sha256(model).hexdigest())
        self.assertEqual(manifest["evaluator_only_sha256"], hashlib.sha256(evaluator).hexdigest())
        self.assertEqual((model, evaluator, manifest), build_h28_contrast_plan.prepare())
        for ml, el in zip(model.splitlines(), evaluator.splitlines()):
            m, e = json.loads(ml), json.loads(el)
            self.assertEqual(m["request_key"], e["request_key"])
            self.assertEqual(set(m), {"request_key", "messages"})
            body = json.dumps(m["messages"], ensure_ascii=False)
            self.assertNotIn(e["request_key"], body)
            self.assertNotIn("source_ref", body)
            self.assertNotIn("private_reference", body)
            self.assertNotIn("reference_code", body)
            self.assertEqual(set(e), {"request_key", "source_ref", "reference_code"})
        self.assertFalse(manifest["blind_test"])
        self.assertEqual(manifest["real_gpu_inference_count"], 0)

    def test_create_only_bundle_and_digest_recheck(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "H28_new"
            manifest = build_h28_contrast_plan.save(path)
            for name, key in (("model_messages.jsonl", "model_messages_sha256"),
                              ("evaluator_only.jsonl", "evaluator_only_sha256")):
                self.assertEqual(hashlib.sha256((path / name).read_bytes()).hexdigest(),
                                 manifest[key])
            self.assertEqual(json.loads((path / "manifest.json").read_text("utf-8")),
                             manifest)
            with self.assertRaises(FileExistsError):
                build_h28_contrast_plan.save(path)


if __name__ == "__main__":
    unittest.main()
