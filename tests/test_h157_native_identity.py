"""H157 frozen screenshot identity preflight guard tests."""
import inspect
import unittest
from h157.preflight_native_identity import run, OUTPUT
from h157.replay_model_dev import preflight

class H157IdentityTests(unittest.TestCase):
    def test_identity_has_no_neural_execution(self):
        src=inspect.getsource(run)
        self.assertIn('page.screenshot(type="png")',src)
        self.assertIn('hashlib.sha256(raw).hexdigest()',src)
        self.assertNotIn('native_mouse_keyboard(',src)
        self.assertNotIn('model.generate(',src)

    def test_pairing_requires_frozen_identity(self):
        src=inspect.getsource(preflight)
        self.assertIn('H157_ORIGINAL_72_DEV_SCREENSHOTS_PIXEL_IDENTICAL',src)
        self.assertIn('identity.get("original_dev_screenshots")!=72',src)
        self.assertIn('identity.get("actual_model_native_reward_trials")!=0',src)

    def test_owner_private_once_receipt(self):
        self.assertEqual(OUTPUT,'H157_FROZEN_DEV_NATIVE_SCREENSHOT_IDENTITY_V1.json')

if __name__=="__main__":
    unittest.main()
