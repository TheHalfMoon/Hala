"""Fresh H133 eight-case browser contract and owner-only oracle actuator QA."""
import os
import unittest
from pathlib import Path
from h133.browser_env import CASES,UnseenBrowserEnv

class H133Contracts(unittest.TestCase):
    def test_exact_eight_frozen_cases_are_distinct(self):
        self.assertEqual(len(CASES),8)
        self.assertEqual(len(set(CASES)),8)
        self.assertFalse(any(x in ("slider","resize","column","drop") for x in CASES))
        content=(Path(__file__).resolve().parents[1]/"h133"/"fixture.html").read_text("utf8")
        for case in CASES: self.assertIn('"'+case+'"',content)
        self.assertIn("setPointerCapture",content)
        self.assertIn('stage.dataset.success',content)

    def test_invalid_scenes_rejected(self):
        with self.assertRaises(ValueError):UnseenBrowserEnv("slider")

@unittest.skipUnless(os.environ.get("H133_BROWSER_E2E")=="1","owner Chromium QA only")
class H133ActualBrowserQA(unittest.TestCase):
    def test_eight_new_drags_actual_pointer_events(self):
        for task in CASES:
            with self.subTest(task=task):
                with UnseenBrowserEnv(task) as env:
                    before=env.observe()
                    box=env._page.locator('[data-role="handle"]').bounding_box()
                    goal=env._page.locator("#target").bounding_box()
                    self.assertIsNotNone(box)
                    self.assertIsNotNone(goal)
                    def middle(r): return (r["x"]+r["width"]/2,r["y"]+r["height"]/2)
                    miss=env.execute_drag((40,400),middle(goal))
                    self.assertFalse(miss.success)
                    success=env.execute_drag(middle(box),middle(goal))
                    self.assertTrue(success.success,task)
                    self.assertNotEqual(before.image_sha256,success.observation.image_sha256)
                    self.assertFalse(env._errors)
                    self.assertFalse(env._external_requests)
if __name__=="__main__":unittest.main()
