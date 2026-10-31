"""H264 invented screenshot-only policy tests. No model, gold, or private media."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import unittest
import json

from PIL import Image

from h264.screenshot_locked_research import (
    STATUS, checked_stockroom, screenshot_locked_turn,
    run_local_research_episode,
)

SOURCE = Path(__file__).resolve().parents[1] / "h262" / "stockroom.html"


def invented_png(color: str = "#f1f5f0") -> bytes:
    image = Image.new("RGB", (960, 720), color)
    bio = BytesIO()
    image.save(bio, format="PNG")
    image.close()
    return bio.getvalue()


def native(kind="click", **attrs):
    defaults = {"point": [100, 200]} if kind == "click" else {}
    return json.dumps({"actions": [{"kind": kind, **(attrs or defaults)}]},
                      ensure_ascii=False)


def page_fake(*, png=None):
    page = Mock()
    page.url = SOURCE.as_uri()
    page.viewport_size = {"width": 960, "height": 720}
    page.screenshot.return_value = invented_png() if png is None else png
    page.evaluate.return_value = True
    return page


class H264SnapshotLockedResearchTests(unittest.TestCase):
    def test_original_source_pinned_and_no_arbitrary_filenames(self):
        path, url = checked_stockroom(SOURCE)
        self.assertEqual(path, SOURCE)
        self.assertEqual(url, SOURCE.as_uri())
        with TemporaryDirectory() as directory:
            root = Path(directory) / "h262"
            root.mkdir()
            wrong = root / "stockroom.html"
            wrong.write_text("<html>not original</html>", encoding="utf-8")
            with self.assertRaises(ValueError):
                checked_stockroom(wrong)
        with self.assertRaises(ValueError):
            checked_stockroom(SOURCE.parent / "wrong.html")

    def test_fixed_observation_then_one_click(self):
        page = page_fake()
        recorded = []
        def predictor(png, prompt):
            recorded.append((png, prompt))
            return native()
        row = screenshot_locked_turn(page, SOURCE, SOURCE.as_uri(),
                                     "Click the visible stock button.", predictor)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0][0], page.screenshot.return_value)
        self.assertIn("screenshot-only", recorded[0][1])
        self.assertTrue(row["screenshot_locked"])
        self.assertIs(row["task_success_measured"], False)
        self.assertIs(row["model_identity_attested"], False)
        page.mouse.click.assert_called_once_with(100, 200)
        self.assertEqual(page.screenshot.call_count, 2)

    def test_stale_frame_prevents_even_a_valid_click(self):
        page = page_fake()
        page.screenshot.side_effect = [invented_png("white"), invented_png("black")]
        with self.assertRaisesRegex(ValueError, "stale screenshot"):
            screenshot_locked_turn(page, SOURCE, SOURCE.as_uri(),
                                   "Click the visible restock button.",
                                   lambda png, task: native())
        page.mouse.click.assert_not_called()

    def test_model_callback_navigating_out_is_refused(self):
        page = page_fake()
        def predictor(png, task):
            page.url = "https://example.org/"
            return native()
        with self.assertRaisesRegex(ValueError, "left pinned"):
            screenshot_locked_turn(page, SOURCE, SOURCE.as_uri(),
                                   "Click one.", predictor)
        page.mouse.click.assert_not_called()

    def test_invalid_model_output_never_dispatches(self):
        page = page_fake()
        with self.assertRaises(ValueError):
            screenshot_locked_turn(page, SOURCE, SOURCE.as_uri(),
                                   "Click one.", lambda png, task: "no json")
        self.assertEqual(page.screenshot.call_count, 1)
        page.mouse.click.assert_not_called()

    def test_invalid_png_headers_or_full_pixels_rejected(self):
        page = page_fake(png=b"not png pixels")
        with self.assertRaises(ValueError):
            screenshot_locked_turn(page, SOURCE, SOURCE.as_uri(),
                                   "Click one.", lambda png, task: native())
        page.mouse.click.assert_not_called()

    def test_stale_viewport_fails_before_predictor(self):
        page = page_fake()
        page.viewport_size = {"width": 1920, "height": 1080}
        predictor = Mock(return_value=native())
        with self.assertRaisesRegex(ValueError, "left pinned"):
            screenshot_locked_turn(page, SOURCE, SOURCE.as_uri(), "Click.", predictor)
        predictor.assert_not_called()

    def test_one_abstain_sends_no_input_and_closes_context(self):
        browser = Mock()
        context = browser.new_context.return_value
        page = context.new_page.return_value
        page.url = SOURCE.as_uri()
        page.viewport_size = {"width": 960, "height": 720}
        page.screenshot.return_value = invented_png()
        result = run_local_research_episode(
            browser, SOURCE, "Help with stock.", lambda image, task: native("abstain"),
            max_turns=8)
        browser.new_context.assert_called_once()
        self.assertEqual(browser.new_context.call_args.kwargs["offline"], True)
        self.assertFalse(browser.new_context.call_args.kwargs["accept_downloads"])
        context.route.assert_called_once()
        context.close.assert_called_once()
        page.mouse.click.assert_not_called()
        self.assertEqual(result["policy_calls"], 1)
        self.assertEqual(result["action_kind_counts"], {"abstain": 1})
        self.assertEqual(result["status"], STATUS)
        self.assertFalse(result["real_model_inference_verified"])
        self.assertFalse(result["ood_evaluated"])
        self.assertFalse(result["real_browser_task_success_measured"])

    def test_episode_bounds_and_network_route(self):
        browser = Mock()
        for count in (0, 9, True, -1, 1.5):
            with self.subTest(turns=count), self.assertRaises(ValueError):
                run_local_research_episode(
                    browser, SOURCE, "Click.", lambda png, prompt: native(), max_turns=count)
        browser.new_context.assert_not_called()
        context = browser.new_context.return_value
        page = context.new_page.return_value
        page.url = SOURCE.as_uri()
        page.viewport_size = {"width": 960, "height": 720}
        page.screenshot.return_value = invented_png()
        result = run_local_research_episode(
            browser, SOURCE, "Click.", lambda png, prompt: native(), max_turns=2)
        self.assertEqual(result["policy_calls"], 2)
        self.assertEqual(result["action_kind_counts"], {"click": 2})
        self.assertFalse(result["private_media_written"])
        ctx_route = context.route.call_args.args[1]
        route = Mock()
        route.request.url = "https://untrusted.example/page"
        ctx_route(route)
        route.abort.assert_called_once_with()
        route.continue_.assert_not_called()

    def test_context_closes_after_malformed_policy_output(self):
        browser = Mock()
        context = browser.new_context.return_value
        page = context.new_page.return_value
        page.url = SOURCE.as_uri()
        page.viewport_size = {"width": 960, "height": 720}
        page.screenshot.return_value = invented_png()
        with self.assertRaises(ValueError):
            run_local_research_episode(
                browser, SOURCE, "Click.", lambda png, task: "MALFORMED")
        context.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
