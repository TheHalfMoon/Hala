"""H163 new upstream and real model-native click guard tests."""
import inspect
import unittest
from h163.source import specs
from h163.replay_model_native import trial, preflight, run

class TestH163Native(unittest.TestCase):
    def test_cases_and_languages(self):
        cases=specs()
        self.assertEqual(len(cases),16)
        self.assertEqual(sum(x["language"]=="ar" for x in cases),8)
        self.assertEqual(sum(x["task"]=="add_widget" for x in cases),8)

    def test_uses_only_model_click(self):
        code=inspect.getsource(trial)
        self.assertIn('page.mouse.click(*act["points"][0])',code)
        self.assertNotIn("owner_gold_click",code)
        self.assertIn('case["image_sha256"]',code)

    def test_one_shot_full_source_gate(self):
        code=inspect.getsource(preflight)+inspect.getsource(run)
        self.assertIn('used.open("x"',code)
        self.assertIn("H163_16_OF_16_ORIGINAL_NEW_APP_EXACT_PIXEL_REPLAY_PASS",code)
        self.assertIn("one_app_family_only",code)
