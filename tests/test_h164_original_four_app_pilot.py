"""H164 four actual original source-app TRAIN pilot split/identity tests."""
import inspect
import unittest
from collections import Counter
from h164.four_app_train_pilot import specs,APPS,N_PER_APP,run
from h164.audit_four_app_train import audit


class H164RealAppPilotTests(unittest.TestCase):
    def test_four_distinct_original_app_96_train_cases(self):
        cases=specs()
        self.assertEqual(len(cases),96)
        self.assertEqual({x["split"] for x in cases},{"TRAIN"})
        self.assertEqual(Counter(x["app"] for x in cases),
                         Counter({app:N_PER_APP for app in APPS}))
        self.assertEqual(len({(x["app"],x["index"]) for x in cases}),96)
        self.assertEqual(Counter(x["language"] for x in cases),
                         {"en":48,"ar":48})
        self.assertEqual(len(set(x["task"] for x in cases)),10)

    def test_pilot_mixes_click_drag_and_native_text_input(self):
        code=inspect.getsource(run)
        source=inspect.getsource(__import__(
            "h164.four_app_train_pilot",fromlist=["native_table"]
        ).native_table)
        self.assertIn("page.keyboard.type",source)
        self.assertIn("mouse.click",source)
        self.assertIn("native_board",code)
        self.assertIn("native_todo",code)
        self.assertIn("original",inspect.getsource(audit).lower())

    def test_only_owner_private_gold_no_model_or_consumed_data(self):
        from h164 import four_app_train_pilot as m
        body=inspect.getsource(m)
        self.assertIn("locked_prior_shas",body)
        self.assertIn('if mode=="verify"',body)
        self.assertIn("stable_png(page)",body)
        self.assertIn("native_reference_success",body)
        self.assertNotIn("optimizer.step(",body)
        self.assertNotIn("model.generate(",body)


if __name__=="__main__":
    unittest.main()
