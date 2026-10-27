"""H164 independent multi-original DEV/OOD split identity and leak guards."""
import inspect,unittest
from collections import Counter
from h164.new_dev_96 import cases as dev,run as dev_run
from h164.new_ood_64 import cases as ood,run as ood_run
from h164.audit_new_dev import audit as dev_audit

class H164IndependentUnseenSources(unittest.TestCase):
    def test_dev_app_language_quota_and_no_training(self):
        rows=dev()
        self.assertEqual(len(rows),96)
        self.assertEqual(Counter(x["app"] for x in rows),
                         {"simple-table":48,"calculator":48})
        self.assertEqual(Counter(x["language"] for x in rows),
                         {"ar":48,"en":48})
        self.assertEqual({x["split"] for x in rows},{"DEV"})
        self.assertEqual(len({x["index"] for x in rows}),96)
    def test_two_app_true_ood_original_app_and_language(self):
        rows=ood()
        self.assertEqual(len(rows),64)
        self.assertEqual(Counter(x["app"] for x in rows),
                         {"pell":32,"pikaday":32})
        self.assertEqual(Counter(x["language"] for x in rows),
                         {"ar":32,"en":32})
        self.assertEqual({x["split"] for x in rows},{"OOD"})
        self.assertEqual(len({x["index"] for x in rows}),64)
    def test_ood_and_dev_no_gold_on_model_side(self):
        ds=inspect.getsource(dev_run)
        os=inspect.getsource(ood_run)
        for text in (ds,os):
            self.assertIn('gold_model_inputs":False',text)
            self.assertIn('mode=="verify"',text)
            self.assertIn("native_reference_success",text)
            self.assertNotIn("optimizer.step(",text)
            self.assertNotIn("model.generate(",text)
    def test_separate_owner_gold_and_independent_audit(self):
        import h164.new_dev_96 as m
        import h164.new_ood_64 as o
        self.assertIn("owner-gold-not-for-GPU",inspect.getsource(m))
        self.assertIn("owner-gold-not-for-model",inspect.getsource(o))
        self.assertIn("validate_source_splits",inspect.getsource(dev_audit))
