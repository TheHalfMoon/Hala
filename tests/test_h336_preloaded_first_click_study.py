"""H336 source/model boundary contracts. Invented policy results only."""
from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from h319.beaconboard_dynamic_research import SCENES
from h335.original_first_click_error_attribution import ROLE as SOURCE_ROLE
from h336.preloaded_first_click_study import (
    ROLE, ORDER, run_preloaded_first_click_study, verify_first_turn,
)

SOURCE = Path(__file__).resolve().parents[1] / "h319" / "beaconboard.html"


def fake_row(language: str, category: str) -> dict:
    return {
        "status": SOURCE_ROLE, "language": language,
        "category": category, "source_public_exposed": True,
        "native_actions_dispatched": 0 if category == "abstain" else 1,
        "genuine_model_identity_attested": False,
        "official_train_admitted": False, "fresh_blind_ood": False,
        "adapted_model_gain_proven": False, "can_release": False,
    }


def fake_summary(category: str) -> dict:
    abstain = category == "abstain"
    return {
        "model_generations_via_wrapped_generate": 1,
        "model_time_token_filters_installed": 1,
        "distinct_original_pngs_observed": 1,
        "focus_clicks": 0 if abstain else 1,
        "type_calls": 0,
        "abstained": abstain,
        "protocol_terminal": abstain,
        "protocol_failed_terminal": False,
        "pretrained_model_identity_attested": False,
        "real_owner_private_gpu_run_attested": False,
        "native_source_task_success_attested": False,
        "official_train_dev_admitted": False,
        "blind_ood_proven": False,
        "generalizable_hala_gain_proven": False,
    }


class H336InventedBoundary(unittest.TestCase):
    def test_frozen_alternating_12_public_exposed_scenes(self):
        self.assertEqual(len(ORDER), 12)
        self.assertEqual(set(ORDER), set(range(12)))
        self.assertEqual(tuple(SCENES[i][1] for i in ORDER), ("en", "ar") * 6)

    def test_one_only_model_generation_without_type_second_turn(self):
        for cat in ("correct_open_and_focus", "wrong_station_open",
                    "correct_open_wrong_focus", "no_station_open", "abstain"):
            with self.subTest(category=cat):
                verify_first_turn(fake_summary(cat), fake_row("ar", cat))

    def test_impossible_fake_model_sequences_are_refused(self):
        for key, bad in (
            ("model_generations_via_wrapped_generate", 2),
            ("model_generations_via_wrapped_generate", True),
            ("model_time_token_filters_installed", 0),
            ("distinct_original_pngs_observed", 0),
            ("type_calls", 1), ("focus_clicks", 0),
            ("protocol_failed_terminal", True),
            ("pretrained_model_identity_attested", True),
            ("generalizable_hala_gain_proven", True),
            ("model_output", "secret"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_first_turn(dict(fake_summary("wrong_station_open"),
                                       **{key: bad}),
                                  fake_row("ar", "wrong_station_open"))

    def test_fake_policy_aggregate_cannot_be_model_evidence(self):
        class DummyPolicy:
            def __init__(self, processor, model, task):
                self.task = task
                self.category = "wrong_station_open" if "افتح" in task else "correct_open_and_focus"
            def safe_contract_summary(self):
                return fake_summary(self.category)
        seen = []
        def fake_source(browser, source, scene, policy, *, role):
            seen.append(scene)
            self.assertEqual(role, SOURCE_ROLE)
            return fake_row(scene[1], policy.category)
        with patch("h336.preloaded_first_click_study.source_first_click",
                   side_effect=fake_source):
            summary = run_preloaded_first_click_study(
                None, SOURCE, None, None, role=ROLE,
                policy_factory=DummyPolicy)
        self.assertEqual(len(seen), 12)
        self.assertEqual(summary["claimed_preloaded_model_generations"], 12)
        self.assertEqual(summary["claimed_native_clicks"], 12)
        self.assertEqual(summary["by_language"]["en"]["categories"]
                         ["correct_open_and_focus"], 6)
        self.assertEqual(summary["by_language"]["ar"]["categories"]
                         ["wrong_station_open"], 6)
        self.assertFalse(summary["learned_adapter_gain"])
        self.assertFalse(summary["fresh_blind_ood"])
        self.assertTrue(summary["real_model_identity_requires_external_attestation"])

    def test_missing_model_step_fails_before_partial_reporting(self):
        class DummyPolicy:
            def __init__(self, processor, model, task):
                pass
            def safe_contract_summary(self):
                return fake_summary("correct_open_and_focus")
        calls = []
        def source(browser, source, scene, policy, *, role):
            calls.append(1)
            if len(calls) == 7:
                raise ValueError("source error")
            return fake_row(scene[1], "correct_open_and_focus")
        with patch("h336.preloaded_first_click_study.source_first_click",
                   side_effect=source), self.assertRaises(ValueError):
            run_preloaded_first_click_study(
                None, SOURCE, None, None, role=ROLE, policy_factory=DummyPolicy)
        self.assertEqual(len(calls), 7)

    def test_unauthorized_roles_refuse_without_source_or_model(self):
        for role in ("TRAIN", "DEV", "BLIND_OOD", "H333", None):
            with self.subTest(role=role), patch(
                "h336.preloaded_first_click_study.checked_source",
                side_effect=AssertionError("should never read source"),
            ):
                with self.assertRaises(ValueError):
                    run_preloaded_first_click_study(
                        None, SOURCE, None, None, role=role)

    def test_source_row_false_promotion_rejected(self):
        good = fake_row("ar", "abstain")
        for key, bad in (
            ("native_actions_dispatched", 1),
            ("category", "raw coordinates"),
            ("fresh_blind_ood", True),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_first_turn(fake_summary("abstain"), dict(good, **{key: bad}))


if __name__ == "__main__":
    unittest.main()
