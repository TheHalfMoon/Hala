"""H332 invented-only source protocol tests, optional real screenshot equality.

Mock/crafted summaries are NEVER evidence of real Qwen predictions.
"""
from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from h319.beaconboard_dynamic_research import SCENES
from h332.paired_exposed_prompt_research import (
    CONTROL, VISUAL_RTL, ROLE, SCENE_INDICES, ORDERS,
    check_pair_frame_reproducibility, run_paired_exposed_study,
    summarize_pairs, valid_observed,
)

SOURCE = Path(__file__).resolve().parents[1] / "h319" / "beaconboard.html"


def forged(success: bool) -> dict:
    return {
        "status": "H323_ORIGINAL_EXPOSED_DYNAMIC_PRELOADED_BRIDGE_ONLY",
        "source_original_task_success": success,
        "real_model_generations_claimed_by_wrapper": 2 if success else 1,
        "native_actions_dispatched": 2 if success else 1,
        "source_categorical_failure": "none" if success
            else "wrong_focus_or_state",
        "pretrained_model_identity_independently_attested": False,
        "source_public_exposed": True,
        "official_train_dev_ood_admitted": False,
        "blind_ood_proven": False,
        "trained_model_gain_proven": False,
        "can_release": False,
    }


def example_pairs() -> list[dict]:
    return [
        {CONTROL: forged(True), VISUAL_RTL: forged(True)},
        {CONTROL: forged(False), VISUAL_RTL: forged(True)},
        {CONTROL: forged(False), VISUAL_RTL: forged(True)},
        {CONTROL: forged(False), VISUAL_RTL: forged(False)},
    ]


class H332InventedContract(unittest.TestCase):
    def test_frozen_bilingual_scenes_and_counterbalance(self):
        self.assertEqual(len(SCENE_INDICES), 4)
        self.assertEqual(len(set(SCENE_INDICES)), 4)
        self.assertEqual(tuple(SCENES[i][1] for i in SCENE_INDICES),
                         ("en", "en", "ar", "ar"))
        self.assertEqual(
            sum(order == (CONTROL, VISUAL_RTL) for order in ORDERS), 2)
        self.assertEqual(
            sum(order == (VISUAL_RTL, CONTROL) for order in ORDERS), 2)

    def test_matched_aggregate_not_model_skill_evidence(self):
        result = summarize_pairs(example_pairs())
        self.assertEqual(result["frozen_public_scene_count"], 4)
        self.assertEqual(result["matched_model_source_runs"], 8)
        self.assertEqual(result["by_language"]["en"]["control"]["task_success"], 1)
        self.assertEqual(result["by_language"]["en"]["candidate"]["task_success"], 2)
        self.assertEqual(result["by_language"]["ar"]["control"]["task_success"], 0)
        self.assertEqual(result["by_language"]["ar"]["candidate"]["task_success"], 1)
        self.assertEqual(result["by_language"]["ar"]["pair_outcomes"]["candidate"], 1)
        self.assertFalse(result["adapted_model_gain_proven"])
        self.assertFalse(result["fresh_blind_ood_proven"])
        self.assertFalse(result["can_release"])
        for group in result["by_language"].values():
            for arm in (CONTROL, "candidate"):
                self.assertEqual(group[arm]["tasks"], 2)
                self.assertEqual(sum(group[arm]["failure_counts"].values()), 2)

    def test_schema_rejects_extra_coordinates_and_false_model_proof(self):
        ok = forged(True)
        self.assertEqual(valid_observed(ok), (True, 2, 2, "none"))
        for key, bad in (
            ("source_public_exposed", False),
            ("pretrained_model_identity_independently_attested", True),
            ("official_train_dev_ood_admitted", True),
            ("blind_ood_proven", True),
            ("trained_model_gain_proven", True),
            ("can_release", True),
            ("source_original_task_success", 1),
            ("real_model_generations_claimed_by_wrapper", True),
            ("native_actions_dispatched", 1),
            ("source_categorical_failure", "wrong_focus_or_state"),
            ("prediction_coordinates", [5, 9]),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                valid_observed(dict(ok, **{key: bad}))

    def test_missing_pair_duplicate_wrong_cohort_refused(self):
        for pairs in (
            example_pairs()[:3],
            example_pairs() + [example_pairs()[0]],
            [dict(x) for x in example_pairs()[:3]] + [{CONTROL: forged(True)}],
            [{CONTROL: forged(True), VISUAL_RTL: forged(True),
              "secret_model_text": "do not print"}] + example_pairs()[1:],
        ):
            with self.assertRaises(ValueError):
                summarize_pairs(pairs)

    def test_role_refusal_precedes_any_browser(self):
        with self.assertRaises(ValueError):
            run_paired_exposed_study(
                None, SOURCE, None, None, role="BLIND_OOD")
        with self.assertRaises(ValueError):
            run_paired_exposed_study(
                None, SOURCE, None, None, role="TRAIN")

    def test_fake_runner_never_attests_model_identity(self):
        result_sequence = iter(example_pairs())
        expected = {id(SCENES[i]): pair for i, pair in
                    zip(SCENE_INDICES, example_pairs(), strict=True)}
        observed = []
        def fake_runner(browser, source, processor, model, scene, **kwargs):
            arm = kwargs.get("prompt_variant", CONTROL)
            observed.append((scene, arm))
            return expected[id(scene)][arm]
        with patch(
            "h332.paired_exposed_prompt_research.check_pair_frame_reproducibility",
            return_value=None,
        ):
            result = run_paired_exposed_study(
                object(), SOURCE, None, None, role=ROLE, runner=fake_runner)
        self.assertEqual(len(observed), 8)
        self.assertTrue(all(len(item) == 2 for item in observed))
        self.assertFalse(result["adapted_model_gain_proven"])
        self.assertTrue(result["genuine_model_identity_requires_external_attestation"])


@unittest.skipUnless(os.getenv("HALA_H332_BROWSER_QUALIFY") == "1",
                     "source-only real browser opt-in required")
class H332RealOriginalSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        try:
            channel = os.getenv("HALA_H332_BROWSER_CHANNEL", "chromium")
            if channel not in ("chromium", "msedge"):
                raise ValueError("unsupported source browser")
            cls.browser = cls.pw.chromium.launch(
                headless=True,
                **({"channel": "msedge"} if channel == "msedge"
                   else {"args": ["--no-sandbox"]}))
        except Exception:
            cls.pw.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def test_all_four_scene_start_frames_identical_across_fresh_sessions(self):
        for index in SCENE_INDICES:
            with self.subTest(index=index):
                check_pair_frame_reproducibility(
                    self.browser, SOURCE, SCENES[index])


if __name__ == "__main__":
    unittest.main()
