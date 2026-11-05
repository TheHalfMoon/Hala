"""H426 source diagnostic tests. Synthetic mutations are NOT real model scores."""
import importlib.util
import json
import os
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location("h426_audit",ROOT/"h426/source_error_audit.py")
a=importlib.util.module_from_spec(sp)
sp.loader.exec_module(a)

def build(*,end=(500,300),start=(200,300),ordered=True,invalid=False):
    rows=[];traces=[]
    for i in range(384):
        target={"type":"drag","start_point":[200,300],"end_point":[500,300],
                "coordinate_space":"normalized_1000","ordered":True}
        rows.append({"id":f"synthetic-{i}","image_sha256":f"{i:064x}",
                     "source_domain":"slider","native_drag_target":target})
        prediction={"type":"drag","start_point":list(start),"end_point":list(end),
                    "coordinate_space":"normalized_1000","ordered":ordered}
        traces.append({"id":f"synthetic-{i}","image_sha256":f"{i:064x}",
                       "raw_prediction":"broken" if invalid else json.dumps(prediction)})
    return rows,traces

class H426OriginalGeometryDiagnostics(unittest.TestCase):
    def test_perfect_model_not_marked_failed(self):
        rows,preds=build()
        s,hits=a.breakdown(rows,preds)
        self.assertEqual(s["slider"]["correct"],384)
        self.assertTrue(all(hits))
        self.assertNotIn("end_point_only_failure",s["slider"])

    def test_actual_end_point_failure_is_not_start_failure(self):
        rows,preds=build(end=(800,300))
        s,hits=a.breakdown(rows,preds)
        self.assertEqual(s["slider"]["end_point_only_failure"],384)
        self.assertEqual(s["slider"]["start_near"],384)
        self.assertEqual(s["slider"].get("start_point_only_failure",0),0)
        self.assertFalse(any(hits))

    def test_actual_start_failure_is_not_end_failure(self):
        rows,preds=build(start=(800,300))
        s,_=a.breakdown(rows,preds)
        self.assertEqual(s["slider"]["start_point_only_failure"],384)
        self.assertEqual(s["slider"]["end_near"],384)

    def test_both_points_wrong(self):
        rows,preds=build(start=(800,100),end=(100,800))
        s,_=a.breakdown(rows,preds)
        self.assertEqual(s["slider"]["both_point_failure"],384)

    def test_wrong_order_flag_is_not_coordinate_failure(self):
        rows,preds=build(ordered=False)
        s,_=a.breakdown(rows,preds)
        self.assertEqual(s["slider"]["order_failure"],384)
        self.assertEqual(s["slider"].get("both_point_failure",0),0)

    def test_invalid_schema_detected(self):
        rows,preds=build(invalid=True)
        s,_=a.breakdown(rows,preds)
        self.assertEqual(s["slider"]["schema_failure"],384)

    def test_tampered_source_fingerprint_rejected(self):
        rows,preds=build()
        preds[0]["image_sha256"]="tampered"
        with self.assertRaisesRegex(ValueError,"image/prompt drift"):
            a.breakdown(rows,preds)

    def test_duplicate_or_missing_original_predictions_rejected(self):
        rows,preds=build()
        preds[1]["id"]=preds[0]["id"]
        with self.assertRaisesRegex(ValueError,"384 unique"):
            a.breakdown(rows,preds)

    def test_actual_owner_private_source_only_when_explicitly_present(self):
        root=os.environ.get("H426_OWNER_PAIR_ROOT")
        if not root:self.skipTest("Original private 384-case H421 source receipt only on owner Windows")
        result=a.audit(
            pair_root=Path(root),
            labels=Path(os.environ["H426_OWNER_LABELS"]),
            source_receipt=Path(os.environ["H426_OWNER_SOURCE_RECEIPT"]),
            source_sidecar=Path(os.environ["H426_OWNER_SOURCE_SIDECAR"]),
            adapter=Path(os.environ["H426_OWNER_ADAPTER"]),
            child_receipt=Path(os.environ["H426_OWNER_CHILD_RECEIPT"]))
        self.assertEqual(result["h425_trained"]["slide_resize"]["end_point_only_failure"],63)
        self.assertEqual(result["h425_trained"]["text_highlight"]["both_point_failure"],60)
        self.assertEqual(sum(d["correct"] for d in result["h425_trained"].values()),172)
        self.assertFalse(result["independent_external_ood"])
        self.assertFalse(result["individual_predictions_included"])

if __name__=="__main__":
    unittest.main()
