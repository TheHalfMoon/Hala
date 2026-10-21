"""First-party H52 genuine browser research training evidence tests."""
import json
from pathlib import Path
import unittest
from h52.audit import audit
from h52.factory import plans, render, RIGHTS, digest, canon

ROOT=Path(__file__).resolve().parents[1]/"h52/assets"
class H52OriginalTaskFactoryTests(unittest.TestCase):
    def test_replay_counts_and_unique_source_bytes(self):
        report=audit(ROOT)
        self.assertEqual(report["rows"],288)
        self.assertEqual(report["original_workflows"],96)
        self.assertEqual(report["independent_screenshots"],288)
        self.assertEqual(report["language_rows"],{"ar":144,"en":144})
        self.assertEqual(report["browser_wrong_targets_rejected"],4)
    def test_true_three_step_task_order_and_actual_clicks(self):
        receipts=json.loads((ROOT/"capture.json").read_text("utf8"))
        for i,p in enumerate(plans()):
            steps=receipts["cases"][i*3:(i+1)*3]
            self.assertEqual([r["stage"] for r in steps],[0,1,2])
            self.assertEqual([r["target_key"] for r in steps],p["targets"])
            self.assertEqual([r["observed_step_after"] for r in steps],[1,2,3])
        self.assertFalse(receipts["model_inference"])
        self.assertFalse(receipts["real_service_access"])
    def test_independent_png_and_frozen_html_source(self):
        records=json.loads((ROOT/"source.json").read_text("utf8"))
        self.assertEqual(records["count"],96)
        self.assertEqual(records["authored_plans_sha256"],
            digest((ROOT/"plans.json").read_bytes()))
        for p in plans():
            self.assertEqual(p["rights"],RIGHTS)
            self.assertEqual((ROOT/"html"/(p["id"]+".html")).read_bytes(),render(p))
        self.assertEqual(len(list((ROOT/"png").glob("*.png"))),288)
    def test_training_split_and_no_private_data_or_gpu(self):
        metadata=json.loads((ROOT/"sft_manifest.json").read_text("utf8"))
        self.assertTrue(metadata["rights_cleared"])
        self.assertEqual(metadata["split"],"train")
        self.assertFalse(metadata["model_training_executed"])
        self.assertFalse(metadata["independent_blind_test"])
        self.assertEqual(metadata["dev_or_benchmark_examples"],0)
        self.assertEqual(metadata["dataset_sha256"],
          digest((ROOT/"authoring_sft.jsonl").read_bytes()))
    def test_screenshot_labels_no_duplicate_image(self):
        rows=[json.loads(x) for x in (ROOT/"authoring_sft.jsonl").read_text("utf8").splitlines()]
        self.assertEqual(len({r["image_sha256"] for r in rows}),288)
        self.assertEqual(len({r["id"] for r in rows}),288)
        self.assertTrue(all(r["split"]=="train" for r in rows))
        for r in rows:
            self.assertTrue(r["prompt"].endswith("Screenshot 1024x640."))
            self.assertEqual(json.loads(r["completion"])["type"],"click")

if __name__=="__main__":
    unittest.main()
