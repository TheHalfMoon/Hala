"""H195 synthetic negative controls for fourth original calendar source, no media."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h185.manifest_sha_only import scan_legacy_manifest_sha_fields
from h195.original_vanilla_datepicker_native import (
    PIN, SCENES, DatepickerNativeReceipt,
    validate_original_specs, validate_replay, source_novelty_research_only,
    create_isolated_calendar_context,
)


def receipts():
    return tuple(
        DatepickerNativeReceipt(
            id=s["id"], language=s["language"],
            month_step=s["next_month_clicks"],
            native_calendar_opened=True,
            native_selected_value=s["expected_value"],
            native_day_center=(450.0 + i, 350.0 + i),
            preaction_png_sha256=("1" if i == 0 else "2")*64,
            preaction_rgba_sha256=("a" if i == 0 else "b")*64,
            postaction_rgba_sha256=("c" if i == 0 else "d")*64,
        )
        for i, s in enumerate(SCENES) for _ in range(2)
    )


class OriginalCalendarResearchTests(unittest.TestCase):
    def test_public_runner_missing_owner_history_is_unknown_not_approved(self):
        with patch("h195.original_vanilla_datepicker_native.screen_pins_against_history",
                   side_effect=AssertionError("must not invent an empty owner inventory")):
            receipt = source_novelty_research_only(None)
        self.assertEqual(receipt["candidate_source_families"], 1)
        self.assertEqual(receipt["historical_git_checkouts_screened"], 0)
        self.assertIs(receipt["training_approved"], False)
        self.assertIs(receipt["source_history_completeness_proven"], False)

    def test_browser_isolation_registered_before_original_page(self):
        events = []
        class Context:
            def route(self, pattern, handler):
                events.append(("http", pattern))
                self.http_handler = handler
            def route_web_socket(self, pattern, handler):
                events.append(("websocket", pattern))
                self.ws_handler = handler
            def close(self):
                events.append(("closed",))
        class Browser:
            def new_context(self, **options):
                self.options = options
                self.context = Context()
                return self.context
        browser = Browser()
        ctx = create_isolated_calendar_context(browser, 8181)
        self.assertIs(ctx, browser.context)
        self.assertEqual(events, [("http", "**/*"), ("websocket", "**/*")])
        self.assertEqual(browser.options["service_workers"], "block")
        self.assertIs(browser.options["accept_downloads"], False)
        class Request:
            def __init__(self, url):
                self.url = url
        class Route:
            def __init__(self, url):
                self.request = Request(url)
                self.result = None
            def continue_(self):
                self.result = "continue"
            def abort(self):
                self.result = "abort"
        for url, expected in (
            ("http://127.0.0.1:8181/demo/index.html", "continue"),
            ("https://external.invalid/leak", "abort"),
            ("http://127.0.0.1:8182/not-this-server", "abort"),
        ):
            route = Route(url)
            ctx.http_handler(route)
            self.assertEqual(route.result, expected)
        class WebSocket:
            def __init__(self):
                self.closed = False
            def close(self):
                self.closed = True
        ws = WebSocket()
        ctx.ws_handler(ws)
        self.assertTrue(ws.closed)

    def test_websocket_route_failure_closes_context_fail_closed(self):
        class Context:
            def __init__(self):
                self.closed = False
            def route(self, *_):
                pass
            def route_web_socket(self, *_):
                raise RuntimeError("cannot isolate WebSockets")
            def close(self):
                self.closed = True
        class Browser:
            def __init__(self):
                self.ctx = Context()
            def new_context(self, **_):
                return self.ctx
        browser = Browser()
        with self.assertRaisesRegex(RuntimeError, "cannot isolate"):
            create_isolated_calendar_context(browser, 8181)
        self.assertTrue(browser.ctx.closed)

    def test_immutable_fourth_original_upstream_and_real_months(self):
        validate_original_specs()
        self.assertEqual(PIN["repo"], "mymth/vanillajs-datepicker")
        self.assertEqual(PIN["license"], "MIT")
        self.assertEqual([s["expected_value"] for s in SCENES],
                         ["10/15/2026", "11/22/2026"])
        self.assertEqual({s["language"] for s in SCENES}, {"ar","en"})

    def test_source_research_only_with_no_data_gold_or_ood(self):
        r = validate_replay(receipts())
        self.assertEqual(r["independent_original_date_selection_tasks"], 2)
        self.assertEqual(r["native_original_open_month_and_day_actions"], 10)
        self.assertEqual(r["unique_preaction_decoded_rgba"], 2)
        self.assertEqual(r["source_role"], "RESEARCH_UNASSIGNED_NOT_ADMITTED")
        self.assertEqual(r["train_dev_ood_admitted"], 0)
        self.assertEqual(r["persisted_images"], 0)
        self.assertEqual(r["persisted_private_gold"], 0)
        self.assertFalse(r["old_ood_accessed"])
        self.assertFalse(r["historical_source_complete"])
        self.assertEqual(r["model_calls"], 0)

    def test_missing_independent_browser_replay_rejected(self):
        with self.assertRaisesRegex(ValueError, "context missing"):
            validate_replay(receipts()[:-1])

    def test_source_language_and_month_identity_mismatch_rejected(self):
        for key, value in (("id", "invented"), ("language", "fr"),
                           ("month_step", 4)):
            with self.subTest(key=key):
                cases = list(receipts())
                cases[0] = replace(cases[0], **{key: value})
                with self.assertRaisesRegex(ValueError, "source task drift"):
                    validate_replay(tuple(cases))

    def test_native_calendar_and_chosen_date_are_required(self):
        for key, value in (("native_calendar_opened", False),
                           ("native_selected_value", "10/16/2026")):
            with self.subTest(key=key):
                cases = list(receipts())
                cases[0] = replace(cases[0], **{key: value})
                with self.assertRaisesRegex(ValueError, "reward not observed"):
                    validate_replay(tuple(cases))

    def test_identical_source_pixels_cannot_count_twice(self):
        for key, value in (("preaction_png_sha256", "1"*64),
                           ("preaction_rgba_sha256", "a"*64)):
            with self.subTest(key=key):
                cases = list(receipts())
                for j in (2,3):
                    cases[j] = replace(cases[j], **{key:value})
                with self.assertRaisesRegex(ValueError, "pixel duplicates"):
                    validate_replay(tuple(cases))

    def test_no_guessed_native_reference_point_or_action_only_focus(self):
        for key, value, error in (
            ("native_day_center", (-3.0, 23.0), "not visible"),
            ("postaction_rgba_sha256", "a"*64, "changed GUI raster"),
            ("preaction_png_sha256", "NOT_SHA", "changed GUI raster"),
        ):
            with self.subTest(key=key):
                cases=list(receipts())
                cases[0]=replace(cases[0], **{key:value})
                with self.assertRaisesRegex(ValueError, error):
                    validate_replay(tuple(cases))

    def test_fresh_context_image_and_geometry_drift_detected(self):
        for key, value in (
            ("preaction_png_sha256", "e"*64),
            ("preaction_rgba_sha256", "f"*64),
            ("native_day_center", (500.0, 500.0)),
        ):
            with self.subTest(key=key):
                cases=list(receipts())
                cases[1]=replace(cases[1], **{key:value})
                with self.assertRaisesRegex(ValueError, "differs across fresh"):
                    validate_replay(tuple(cases))

    def test_known_historic_png_or_rgba_collision_fails(self):
        for key, value in (
            ("image_sha256", "1"*64),
            ("visual_rgba_sha256", "b"*64),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                old=Path(folder)/"private-h140"
                old.mkdir()
                (old/"manifest.json").write_text(
                    json.dumps({key:value}),encoding="utf-8",
                )
                history=scan_legacy_manifest_sha_fields([folder])
                with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                    validate_replay(receipts(),history)


if __name__=="__main__":
    unittest.main()
