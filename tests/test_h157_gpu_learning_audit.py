"""H157 independent GPU raw prediction admission (offline)."""
import json
import tempfile
import unittest
from collections import Counter

from h157.audit_gpu_learning import checked_predictions,audit
from h157.collect_spatial_source import COUNTS

MODEL_SHA="a"*64


class H157GPUReceiptTests(unittest.TestCase):
    def setUp(self):
        self.dev=[]
        self.predictions=[]
        index=0
        for task,number in COUNTS["DEV"].items():
            for _ in range(number):
                index+=1
                sha=f"{index:064x}"
                if task.startswith("click-"):
                    actions=[{"kind":"click","point":[130,160]}]
                    needs_text=False
                elif task.startswith("enter-text"):
                    actions=[
                        {"kind":"click","point":[110,150]},
                        {"kind":"type","text":"Sara"},
                        {"kind":"submit","point":[200,200]}]
                    needs_text=True
                else:
                    actions=[
                        {"kind":"select","point":[150,160],"text":"Mali"},
                        {"kind":"submit","point":[180,250]}]
                    needs_text=True
                self.dev.append({
                    "task":task,"image_sha256":sha,
                    "training_supervision":{"actions":actions}})
                self.predictions.append({
                    "task":task,"image_sha":sha,
                    "raw":json.dumps({"actions":actions}),
                    "valid":True,"sequence":True,
                    "text_correct":True if needs_text else None})

    def test_72_valid_native_action_text_roles_are_disjoint(self):
        stats=checked_predictions(self.predictions,MODEL_SHA,self.dev)
        self.assertEqual(stats["dev"],72)
        self.assertEqual(stats["valid_json"],72)
        self.assertEqual(stats["valid_action_sequences"],72)
        self.assertEqual(stats["requires_text"],40)
        self.assertEqual(stats["correct_text_values"],40)

    def test_click_tasks_cannot_report_correct_text_and_unknown_dev_fails(self):
        rows=[dict(r) for r in self.predictions]
        rows[0]["text_correct"]=True
        with self.assertRaises(ValueError):
            checked_predictions(rows,MODEL_SHA,self.dev)
        rows=[dict(r) for r in self.predictions]
        rows[0]["image_sha"]="f"*64
        with self.assertRaises(ValueError):
            checked_predictions(rows,MODEL_SHA,self.dev)

    def test_no_new_neural_claim_without_checkpoint_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                audit(folder,folder)


if __name__=="__main__":
    unittest.main()
