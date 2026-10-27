"""Native real-app cross-application model-reward guardrails (no private media)."""
import inspect
import tempfile
import unittest

from h160.replay_neural_native import episode,preflight,run
from h158.audit_neural_transfer import paired


class H160PrivateOneShotTests(unittest.TestCase):
    def test_native_receipt_requires_private_gpu_output(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                preflight(temp,temp,temp)

    def test_no_reference_action_or_dom_oracle_for_model_clicks(self):
        code=inspect.getsource(episode)
        self.assertIn("page.mouse.click(*click)",code)
        self.assertNotIn('reference_point',code)
        self.assertNotIn('native_gold_point',code)
        self.assertNotIn('page.evaluate("() => document.querySelector',code)
        self.assertIn("before != case",code)

    def test_consumption_guard_prevents_scoring_twice(self):
        code=inspect.getsource(run)
        self.assertIn('consumed.open("x"',code)
        self.assertIn("consumed.exists() or result.exists()",code)
        self.assertIn("paired(rows,cases)",code)
        self.assertIn("source_manifest_sha256",code)
        self.assertTrue(callable(paired))


if __name__ == "__main__":
    unittest.main()
