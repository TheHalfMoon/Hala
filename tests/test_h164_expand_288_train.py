"""H164 scaled original TRAIN expansion fail-closed/source schema tests."""
import inspect,unittest
from collections import Counter
from h164.expand_288_train import specs,APPS,SCHEMA,prior
from h164.audit_expanded_train import audit

class H164ExpandedSourceTests(unittest.TestCase):
    def test_new_mixed_original_app_288_quota(self):
        cases=specs()
        self.assertEqual(len(cases),288)
        self.assertEqual(Counter(x["app"] for x in cases),
                         {x:96 for x in APPS})
        self.assertEqual(Counter(x["language"] for x in cases),
                         {"en":144,"ar":144})
        self.assertEqual(len({x["index"] for x in cases}),288)
        self.assertEqual({x["split"] for x in cases},{"TRAIN"})
        self.assertEqual(len(set(x["task"] for x in cases)),7)
    def test_cannot_copy_prior_eval_or_initial_pilot(self):
        code=inspect.getsource(prior)
        for part in ("locked_prior_shas","PILOT_PIXEL_QA",
                     "four-app-train-pilot-v1-quarantined"):
            self.assertIn(part,code)
    def test_actual_original_apps_and_model_free(self):
        from h164 import expand_288_train as module
        code=inspect.getsource(module)
        self.assertIn("native_calculator",code)
        self.assertIn("mouse_drag",code)
        self.assertIn("salary",code)
        self.assertIn("stable_png(page)",code)
        self.assertNotIn("optimizer.step",code)
        self.assertNotIn("model.generate",code)
        self.assertIn("train_only",code)
        self.assertIn("native_reference_success",code)
    def test_three_app_independent_auditor(self):
        source=inspect.getsource(audit)
        self.assertIn("validate_source_splits",source)
        self.assertIn("supervision(row)",source)
        self.assertIn("combined_train",source)
        self.assertIn("384",source)
