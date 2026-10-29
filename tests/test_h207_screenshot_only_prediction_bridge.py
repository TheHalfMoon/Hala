"""H207 no-cost screenshot-only prediction adapter safety tests.

Generated synthetic PNG bytes only; NO model, media/gold, OOD or GPU.
"""
import json
import unittest

from h201.original_native_action_executor import exact_original_loopback_url
from h207.screenshot_prediction_bridge import run_screenshot_only_policy
from test_h203_screenshot_bound_dispatch import Page, generated_960_png


def one(kind="click", **extra):
    if not extra and kind == "click":
        extra = {"point": [200, 170]}
    return json.dumps({"actions": [{"kind": kind, **extra}]})


class H207ScreenshotOnlyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.image = generated_960_png(18)
        self.page = Page(self.image)
        self.url = exact_original_loopback_url(8201)

    def run_policy(self, callback, task="Click the visible menu"):
        return run_screenshot_only_policy(self.page, task, self.url, callback)

    def test_valid_click_only_gets_image_and_instruction(self):
        seen = []
        def predictor(image, prompt):
            seen.append((image, prompt))
            return one()
        receipt = self.run_policy(predictor)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], self.image)
        self.assertIn("Click the visible menu", seen[0][1])
        self.assertIn("screenshot-only computer-use", seen[0][1])
        self.assertNotIn("owner_reference", seen[0][1])
        self.assertEqual(self.page.events, [("click", 200, 170)])
        self.assertEqual(self.page.screenshots, 2)
        self.assertEqual(receipt["native_events"], 1)
        self.assertTrue(receipt["preaction_png_verified"])
        self.assertTrue(receipt["strict_action_json_verified"])
        self.assertFalse(receipt["model_identity_attested"])
        self.assertFalse(receipt["training_data_admitted"])
        self.assertFalse(receipt["ood_evaluation_performed"])
        self.assertFalse(receipt["private_media_persisted"])

    def test_abstain_never_dispatches_input(self):
        proof = self.run_policy(lambda _img, _task: one("abstain"))
        self.assertEqual(proof["native_events"], 0)
        self.assertEqual(proof["status"], "ABSTAIN_NO_BROWSER_INPUT")
        self.assertEqual(self.page.events, [])

    def test_key_press_focus_then_registered_key(self):
        proof = self.run_policy(
            lambda _img, _task: one("key_press", point=[100, 110],
                                     key="ArrowRight"))
        self.assertEqual(proof["native_events"], 2)
        self.assertEqual(self.page.events, [
            ("click", 100, 110), ("key", "ArrowRight")])

    def test_drag_is_native_events_only(self):
        proof = self.run_policy(
            lambda _img, _task: one("drag", start=[50, 50], end=[70, 90]))
        self.assertEqual(proof["native_events"], 4)
        self.assertEqual([event[0] for event in self.page.events],
                         ["move", "down", "move", "up"])

    def test_duplicate_key_and_nan_rejected_before_any_input(self):
        variants = (
            '{"actions":[{"kind":"click","point":[1,2],"point":[3,4]}]}',
            '{"actions":[{"kind":"click","point":[NaN,4]}]}',
            '{"actions":[{"kind":"key_press","point":[1,2],"key":"Ctrl+S"}]}',
        )
        for value in variants:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.run_policy(lambda _img, _prompt: value)
                self.assertEqual(self.page.events, [])

    def test_prediction_return_type_must_be_string(self):
        for value in (None, {"actions": []}, b"{}"):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ValueError):
                    self.run_policy(lambda _img, _prompt: value)
                self.assertEqual(self.page.events, [])

    def test_predictor_exception_does_not_send_native_input(self):
        def fail(_image, _prompt):
            raise RuntimeError("inference unavailable")
        with self.assertRaisesRegex(RuntimeError, "inference unavailable"):
            self.run_policy(fail)
        self.assertEqual(self.page.events, [])

    def test_changed_gui_during_prediction_cannot_receive_stale_click(self):
        def changed(_image, _prompt):
            self.page.image = generated_960_png(19)
            return one()
        with self.assertRaisesRegex(ValueError, "stale or changed"):
            self.run_policy(changed)
        self.assertEqual(self.page.events, [])

    def test_model_side_navigation_rejects_action(self):
        def redirected(_image, _prompt):
            self.page.url = "https://untrusted.example/"
            return one()
        with self.assertRaisesRegex(ValueError, "browser left"):
            self.run_policy(redirected)
        self.assertEqual(self.page.events, [])

    def test_invalid_url_blocks_capture_and_inference(self):
        calls = []
        with self.assertRaisesRegex(ValueError, "only registered"):
            run_screenshot_only_policy(
                self.page, "Click Save",
                "http://localhost:8201/snap/index.html",
                lambda *_: calls.append(True),
            )
        self.assertEqual(calls, [])
        self.assertEqual(self.page.screenshots, 0)

    def test_invalid_task_blocks_capture_and_inference(self):
        for bad in ("", "x" * 501, "\x00task"):
            calls = []
            with self.subTest(task_length=len(bad)):
                with self.assertRaises(ValueError):
                    self.run_policy(
                        lambda *_: calls.append(True),
                        task=bad,
                    )
                self.assertEqual(calls, [])
                self.assertEqual(self.page.screenshots, 0)

    def test_missing_predictor_blocks_capture(self):
        with self.assertRaisesRegex(ValueError, "predictor"):
            self.run_policy(None)
        self.assertEqual(self.page.screenshots, 0)

    def test_bad_preaction_png_blocks_prediction(self):
        self.page.image = b"not a browser PNG"
        calls = []
        with self.assertRaises(ValueError):
            self.run_policy(lambda *_: calls.append(True))
        self.assertEqual(calls, [])
        self.assertEqual(self.page.events, [])

    def test_viewport_change_before_prediction_blocks_it(self):
        self.page.viewport_size = {"width": 720, "height": 960}
        calls = []
        with self.assertRaisesRegex(ValueError, "viewport"):
            self.run_policy(lambda *_: calls.append(True))
        self.assertEqual(calls, [])
        self.assertEqual(self.page.screenshots, 0)


if __name__ == "__main__":
    unittest.main()
