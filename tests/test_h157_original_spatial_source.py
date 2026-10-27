"""H157 original multi-layout action contract regressions (CPU-only)."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from h157.action_contract_v3 import SEQUENCES,parse_v3
from h157.collect_spatial_source import (
    COUNTS,EXPECTED_TRAIN,EXPECTED_DEV,owner_reference_actions,prior_media_shas,
)
from h157.audit_spatial_source import verify_png

MODEL_SHA="a"*64


class H157IndependentMultiTaskTests(unittest.TestCase):
    def test_train_dev_cover_new_visual_and_dynamic_tasks(self):
        self.assertEqual(sum(COUNTS["TRAIN"].values()),EXPECTED_TRAIN)
        self.assertEqual(sum(COUNTS["DEV"].values()),EXPECTED_DEV)
        self.assertEqual(len(SEQUENCES),5)
        self.assertEqual(SEQUENCES["click-button"],("click",))
        self.assertEqual(SEQUENCES["enter-text-dynamic"],("click","type","submit"))

    def test_bounded_native_typed_action_output_on_new_original_task(self):
        good={"actions":[{"kind":"click","point":[128.5,203.5]}]}
        self.assertEqual(parse_v3(json.dumps(good),task="click-button",
                                  model_checkpoint_sha=MODEL_SHA)[0].kind,
                         "click")
        for field in (
            {"actions":[{"kind":"click","point":[128.5,203.5],
                         "dom":"#button"}]},
            {"actions":[{"kind":"type","text":"hello"}]},
            {"actions":[{"kind":"click","point":[-1,24]}]},
            {"actions":[{"kind":"click","point":[200,300]},
                        {"kind":"abstain"}]},
        ):
            with self.subTest(field=field),self.assertRaises(ValueError):
                parse_v3(json.dumps(field),task="click-button",
                         model_checkpoint_sha=MODEL_SHA)

    def test_source_oracle_cannot_evaluate_unknown_app(self):
        with self.assertRaises(ValueError):
            owner_reference_actions(None,"unknown-original-app")

    def test_untrusted_png_and_missing_prior_history_fail_closed(self):
        with self.assertRaises(ValueError):
            verify_png(b"not a native screenshot")
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(ValueError):
                prior_media_shas(path)

    def test_native_controller_implementation_reuses_dom_free_actuator(self):
        source=inspect.getsource(__import__(
            "h157.collect_spatial_source",fromlist=["collect_episode"]).collect_episode)
        self.assertIn("native_mouse_keyboard(",source)
        self.assertNotIn("select_option(",source)
        self.assertNotIn("model_predict(",source)


if __name__=="__main__":
    unittest.main()
