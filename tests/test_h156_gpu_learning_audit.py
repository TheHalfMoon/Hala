"""Independent H156 GPU/model prediction receipts: no GPU required."""
import json
import tempfile
import unittest

from h156.audit_gpu_learning import audit,evaluate_predictions


class H156NeuralDEVIndependentAuditTests(unittest.TestCase):
    def setUp(self):
        self.model_sha="a"*64
        self.dev=[]
        self.predictions=[]
        for j,task in enumerate(("click-test","enter-text","choose-list")):
            for i in range(4):
                sha=f"{j*4+i+1:064x}"
                if task=="click-test":
                    actions=[{"kind":"click","point":[100,140]}]
                elif task=="enter-text":
                    actions=[{"kind":"click","point":[80,140]},
                             {"kind":"type","text":"Sara"},
                             {"kind":"submit","point":[150,190]}]
                else:
                    actions=[{"kind":"select","point":[180,180],"text":"Mali"},
                             {"kind":"submit","point":[200,250]}]
                raw=json.dumps({"actions":actions})
                self.dev.append({"image_sha256":sha,"task":task,
                    "training_supervision":{"actions":actions}})
                self.predictions.append({"task":task,"image_sha":sha,
                    "raw":raw,"valid":True,"sequence":True,
                    "text_correct":True})

    def test_clicks_not_miscounted_as_correct_visible_text(self):
        value=evaluate_predictions(self.predictions,self.model_sha,self.dev)
        self.assertEqual(value["n"],12)
        self.assertEqual(value["valid"],12)
        self.assertEqual(value["action_sequences"],12)
        self.assertEqual(value["text_required"],8)
        self.assertEqual(value["text_correct"],8)

    def test_no_leakage_or_fabricated_gpu_accuracy(self):
        cases=[dict(x) for x in self.predictions]
        cases[0]={"task":"click-test","image_sha":cases[0]["image_sha"],
                  "raw":"not json","valid":False,"sequence":False,
                  "text_correct":False}
        value=evaluate_predictions(cases,self.model_sha,self.dev)
        self.assertEqual(value["valid"],11)
        with self.assertRaises(ValueError):
            evaluate_predictions(self.predictions[:-1],self.model_sha,self.dev)
        bad=[dict(x) for x in self.predictions]
        bad[0]["text_correct"]=False
        with self.assertRaises(ValueError):
            evaluate_predictions(bad,self.model_sha,self.dev)

    def test_no_sft_claim_without_actual_receipt_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                audit(tmp,tmp)


if __name__=="__main__":
    unittest.main()
