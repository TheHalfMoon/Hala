"""H130 source-only controls and optional real Chromium E2E smoke.

Opt in locally with H130_BROWSER_E2E=1; CI must not install a browser.
The deterministic geometry below exists ONLY in this oracle driver, never
inside model-facing environment or observation payload.
"""
import os
from pathlib import Path
import unittest
from h130.browser_env import BrowserDragEnv, Observation, TASKS, _point

class H130ContractTests(unittest.TestCase):
    def test_fail_closed_action_points(self):
        for p in ((-1,0),(960,0),(0,720),(float("nan"),3),
                  (None,5),(True,2),(2,3,4),"3,4"):
            with self.assertRaises(ValueError):
                _point(p)
        self.assertEqual(_point([14.5, 22]),(14.5,22.0))

    def test_fixture_has_four_real_pointer_drags(self):
        html=Path(__file__).resolve().parents[1] / "h130" / "fixture.html"
        source=html.read_text(encoding="utf8")
        for task in TASKS:
            self.assertIn(f't==="{task}"',source)
        self.assertIn("setPointerCapture",source)
        self.assertIn('stage.dataset.success',source)
        self.assertIn("Content-Security-Policy",source)

    def test_agent_observation_exposes_only_pixels_and_instruction(self):
        data=Observation("Drag a slider",b"synthetic",960,720,0)
        self.assertEqual(len(data.image_sha256),64)
        self.assertEqual(set(data.__dict__),{
            "instruction","screenshot_png","width","height","step"})

@unittest.skipUnless(os.environ.get("H130_BROWSER_E2E")=="1",
                      "real Chromium owner-only E2E explicitly enabled")
class H130RealBrowserE2ETests(unittest.TestCase):
    def test_every_task_executes_actual_pointer_events(self):
        for task in TASKS:
            with self.subTest(task=task):
                with BrowserDragEnv(task) as env:
                    before=env.observe()
                    self.assertIn("Drag",before.instruction)
                    self.assertGreater(len(before.screenshot_png),5000)
                    stage=env._page.locator("#stage").bounding_box()
                    handle=env._page.locator('[data-role="handle"]').bounding_box()
                    target=env._page.locator(".target").bounding_box()
                    def center(rect):
                        return (rect["x"]+rect["width"]/2,
                                rect["y"]+rect["height"]/2)
                    # Negative probe deliberately misses handle, not target.
                    invalid=env.execute_drag((45,420),center(target))
                    self.assertFalse(invalid.success)
                    self.assertFalse(invalid.terminal)
                    # Oracle geometry ONLY in QA; no model used here.
                    if task=="resize":
                        destination=(target["x"]+target["width"]-4,
                                     target["y"]+target["height"]-4)
                    else:
                        destination=center(target)
                    result=env.execute_drag(center(handle),destination)
                    self.assertTrue(result.success,task)
                    self.assertTrue(result.terminal)
                    self.assertEqual(result.attempts,2)
                    self.assertNotEqual(before.image_sha256,
                                        result.observation.image_sha256)
                    self.assertEqual(result.observation.width,960)
                    self.assertFalse(env._external_requests)
                    self.assertFalse(env._errors)
                    with self.assertRaises(RuntimeError):
                        env.execute_drag(center(handle),destination)

if __name__=="__main__":
    unittest.main()
