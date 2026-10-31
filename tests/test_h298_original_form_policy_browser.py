"""H298 REAL browser native state with STUB VLM model.generate only.

Original FieldPort 960x720 browser and H199/H263 focus+type are REAL.
Fake Qwen outputs use fixture-only source geometry and target codes;
no actual pretrained neural inference, optimizer or adapted model score.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest import TestCase, mock, skipUnless

from h296.fieldport_sequence_research import (
    ROLE, SCENES, IDS, checked_source, original_scene,
)
from h298.preloaded_form_two_stage_policy import run_fixed_original_source_research
ABSTAIN = '{"actions":[{"kind":"abstain"}]}'


class StubProcessor:
    tokenizer = object()
    def __init__(self, responses):
        self.responses = list(responses)
        self.apply_chat_template = lambda *args, **kwargs: None

    def batch_decode(self, *args, **kwargs):
        if not self.responses:
            raise ValueError("No further fake model actions")
        return [self.responses.pop(0)]


class StubModel:
    device = "cpu"
    def __init__(self):
        self.calls = 0
    def eval(self):
        return self
    def generate(self, **kwargs):
        assert callable(kwargs.get("prefix_allowed_tokens_fn"))
        self.calls += 1
        return [[10, 11]]


class FakeOnlyVisualPolicy:
    """Test-only preloaded VLM stub, never a Qwen performance result."""
    def __init__(self, processor, model, *, max_new_tokens=160):
        assert max_new_tokens == 160
        self.processor, self.model = processor, model

    def __call__(self, png, prompt):
        assert png[:4] == bytes((137, 80, 78, 71)) and "screenshot" in prompt.lower()
        self.model.eval()
        self.model.generate(input_ids=[[1]])
        return self.processor.batch_decode([[10, 11]])[0]


SOURCE = Path(__file__).resolve().parents[1] / "h296" / "fieldport.html"


@skipUnless(os.environ.get("HALA_H298_BROWSER_QUALIFY") == "1",
            "explicit source-only real browser / fake model integration")
class H298TwoStageFakeModelRealBrowserTests(TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.engine = sync_playwright().start()
        try:
            channel = os.environ.get("HALA_H298_BROWSER_CHANNEL", "chromium")
            if channel not in ("chromium", "msedge"):
                raise ValueError("unregistered H298 browser channel")
            kwargs = ({"headless": True, "channel": "msedge"}
                      if channel == "msedge" else
                      {"headless": True, "args": ["--no-sandbox"]})
            cls.browser = cls.engine.chromium.launch(**kwargs)
        except BaseException:
            cls.engine.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.engine.stop()

    def setUp(self):
        self.fake_policy = mock.patch(
            "h208.preloaded_vlm_policy.PreloadedVisionPolicy",
            FakeOnlyVisualPolicy)
        self.fake_filter = mock.patch(
            "h298.preloaded_form_two_stage_policy.model_time_filter",
            side_effect=lambda tokenizer,stage: lambda *args: [1])
        self.fake_policy.start()
        self.fake_filter.start()
        self.addCleanup(self.fake_filter.stop)
        self.addCleanup(self.fake_policy.stop)

    def original_center_as_normalized(self, scene, *, wrong=False):
        source, url = checked_source(SOURCE)
        with original_scene(self.browser, source, url, scene) as page:
            index = (scene[4]+1) % 4 if wrong else scene[4]
            rect = page.locator(
                'input[data-field-id="' + IDS[index] + '"]'
            ).bounding_box()
            self.assertIsNotNone(rect)
            x = int(rect["x"] + rect["width"]/2)
            y = int(rect["y"] + rect["height"]/2)
            self.assertTrue(0 < x < 960 and 0 < y < 720)
            return (x*1000//960, y*1000//720)

    def preprogrammed_inputs(self, scene, *, wrong=False, code=None):
        x, y = self.original_center_as_normalized(scene,wrong=wrong)
        focus = '{"actions":[{"kind":"click","point":[' + str(x) + ',' + str(y) + ']}]}'
        value = scene[5] if code is None else code
        typed = '{"actions":[{"kind":"type_text","text":"' + value + '"}]}'
        return [focus,typed]

    def test_four_english_arabic_real_browser_two_step_cases(self):
        for scene in (SCENES[0], SCENES[3], SCENES[6], SCENES[9]):
            with self.subTest(public_scene=scene[0]):
                fake_model=StubModel()
                summary=run_fixed_original_source_research(
                    self.browser,SOURCE,
                    StubProcessor(self.preprogrammed_inputs(scene)),
                    fake_model,scene)
                self.assertTrue(summary["source_original_gui_state_success"])
                self.assertEqual(summary["native_actions"],2)
                self.assertEqual(fake_model.calls,2)
                safe=summary["preloaded_model_contract"]
                self.assertEqual(safe["model_generations_via_wrapped_generate"],2)
                self.assertEqual(safe["model_time_token_filters_installed"],2)
                self.assertEqual(safe["distinct_original_pngs_observed"],2)
                self.assertFalse(safe["pretrained_model_identity_attested"])
                self.assertFalse(summary["independently_attested_model_skill"])
                self.assertTrue(summary["source_public_exposed_not_blind"])

    def test_real_native_wrong_field_never_becomes_task_success(self):
        scene=SCENES[1]
        summary=run_fixed_original_source_research(
            self.browser,SOURCE,
            StubProcessor(self.preprogrammed_inputs(scene,wrong=True)),
            StubModel(),scene)
        self.assertEqual(summary["native_actions"],2)
        self.assertFalse(summary["source_original_gui_state_success"])
        self.assertEqual(
            summary["preloaded_model_contract"]["type_calls"],1)

    def test_real_native_wrong_code_never_becomes_task_success(self):
        scene=SCENES[7]
        summary=run_fixed_original_source_research(
            self.browser,SOURCE,
            StubProcessor(self.preprogrammed_inputs(scene,code="FIC-99")),
            StubModel(),scene)
        self.assertEqual(summary["native_actions"],2)
        self.assertFalse(summary["source_original_gui_state_success"])

    def test_real_native_abstention_does_not_type_or_fabricate_success(self):
        scene=SCENES[2]
        summary=run_fixed_original_source_research(
            self.browser,SOURCE,StubProcessor([ABSTAIN]),
            StubModel(),scene)
        self.assertFalse(summary["source_original_gui_state_success"])
        self.assertEqual(summary["native_actions"],1)
        self.assertTrue(summary["preloaded_model_contract"]["abstained"])
        self.assertEqual(
            summary["preloaded_model_contract"]["type_calls"],0)
