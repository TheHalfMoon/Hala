"""H334 invented-only paired orchestration tests: NO actual model inference."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from h319.beaconboard_dynamic_research import SCENES
from h331.click_prompt_candidate import CONTROL, VISUAL_RTL
from h334.frozen_full_paired_preloaded_probe import (
    ARM_ORDER, SCENE_ORDER, SAFE_KEYS, PairedProbeRefused,
    _safe_arm, run_paired_exposed_preloaded_probe,
)


def row(success=False, failure="abstain"):
    if success:
        failure = "none"
    n = 1 if failure in {"abstain", "wrong_focus_or_state"} else 2
    dispatched = {
        "none": 2, "abstain": 0, "wrong_focus_or_state": 1,
        "wrong_code": 1, "source_state_mismatch": 2,
    }[failure]
    return {
        "status": "H323_ORIGINAL_EXPOSED_DYNAMIC_PRELOADED_BRIDGE_ONLY",
        "source_original_task_success": success,
        "real_model_generations_claimed_by_wrapper": n,
        "native_actions_dispatched": dispatched,
        "source_categorical_failure": failure,
        "pretrained_model_identity_independently_attested": False,
        "source_public_exposed": True,
        "official_train_dev_ood_admitted": False,
        "blind_ood_proven": False,
        "trained_model_gain_proven": False,
        "can_release": False,
    }


class H334PairedTests(unittest.TestCase):
    def test_frozen_balanced_language_and_arm_order(self):
        self.assertEqual(len(SCENE_ORDER), 12)
        self.assertEqual(set(SCENE_ORDER), set(range(12)))
        self.assertEqual(len(ARM_ORDER), 12)
        for lang in ("en", "ar"):
            order = [pair[0] for i, pair in zip(SCENE_ORDER, ARM_ORDER)
                     if SCENES[i][1] == lang]
            self.assertEqual(order.count(CONTROL), 3)
            self.assertEqual(order.count(VISUAL_RTL), 3)

    def test_genuine_bridge_is_called_twice_per_scene_and_aggregate_only(self):
        calls = []
        def fake(_browser, _source, _processor, _model, scene, *,
                 prompt_variant):
            index = SCENES.index(scene)
            calls.append((index, prompt_variant))
            succeeds = (index % 3 == 0 if prompt_variant == CONTROL
                        else index % 3 in (0, 1))
            return row(success=succeeds)
        with patch("h334.frozen_full_paired_preloaded_probe._run_arm", fake), patch("h334.frozen_full_paired_preloaded_probe._check_pair", lambda *_args: None):
            report = run_paired_exposed_preloaded_probe(
                object(), object(), object(), object())
        self.assertEqual(len(calls), 24)
        self.assertEqual(calls, [
            (i, variant) for i, order in zip(SCENE_ORDER, ARM_ORDER)
            for variant in order])
        self.assertEqual(report["paired_exposed_cases"], 12)
        self.assertEqual(report["arm_runs"], 24)
        self.assertEqual(report["aggregate"]["control_success"], 4)
        self.assertEqual(report["aggregate"]["candidate_success"], 8)
        self.assertEqual(report["aggregate"]["candidate_only"], 4)
        self.assertEqual(report["aggregate"]["control_only"], 0)
        self.assertEqual(report["aggregate"]["both_failed"], 4)
        for lang in ("en", "ar"):
            self.assertEqual(report["by_language"][lang]["pairs"], 6)
            self.assertEqual(report["by_language"][lang]["candidate_success"], 4)
        self.assertFalse(report["pretrained_model_identity_attested"])
        self.assertFalse(report["source_disjoint_blind_ood"])
        self.assertFalse(report["adapter_trained_or_gain_proven"])
        self.assertFalse(report["can_release"])
        self.assertNotIn("scene", str(report).lower())
        self.assertNotIn("predicted", str(report).lower())
        self.assertNotIn("SIM-", str(report))

    def test_safe_arm_rejects_extra_raw_model_output(self):
        bad = row(success=True)
        bad["raw_prediction"] = '{"actions":[]}'
        with self.assertRaises(PairedProbeRefused):
            _safe_arm(bad)
        self.assertEqual(set(row()), SAFE_KEYS)

    def test_safe_arm_rejects_forged_types_and_claims(self):
        for key, value in (
            ("source_original_task_success", 1),
            ("real_model_generations_claimed_by_wrapper", True),
            ("native_actions_dispatched", False),
            ("source_public_exposed", False),
            ("blind_ood_proven", True),
            ("trained_model_gain_proven", True),
            ("can_release", True),
        ):
            bad = row(success=True)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(PairedProbeRefused):
                _safe_arm(bad)

    def test_safe_arm_rejects_impossible_terminal_state(self):
        for changes in (
            {"source_categorical_failure": "none",
             "source_original_task_success": False},
            {"source_categorical_failure": "wrong_focus_or_state",
             "real_model_generations_claimed_by_wrapper": 2},
            {"source_categorical_failure": "abstain",
             "native_dispatched_actions": 2},
            {"source_categorical_failure": "unknown"},
        ):
            bad = row()
            bad.update(changes)
            with self.subTest(changes=changes), self.assertRaises(PairedProbeRefused):
                _safe_arm(bad)

    def test_one_failed_arm_aborts_without_partial_aggregate(self):
        calls = []
        def fail(_browser, _source, _processor, _model, scene, *,
                 prompt_variant):
            calls.append((scene, prompt_variant))
            if len(calls) == 5:
                raise RuntimeError("PRIVATE RAW MODEL RESPONSE MUST NOT LEAK")
            return row(success=True)
        with patch("h334.frozen_full_paired_preloaded_probe._run_arm", fail), patch("h334.frozen_full_paired_preloaded_probe._check_pair", lambda *_args: None):
            with self.assertRaisesRegex(
                PairedProbeRefused, "^H334 PAIRED_EXPOSED_PROBE_REFUSED$"
            ) as err:
                run_paired_exposed_preloaded_probe(
                    object(), object(), object(), object())
        self.assertEqual(len(calls), 5)
        self.assertNotIn("PRIVATE", str(err.exception))

    def test_wrong_private_field_refused_before_report(self):
        def fake(*_args, **_kwargs):
            bad = row(success=True)
            bad["task_gold"] = "secret"
            return bad
        with patch("h334.frozen_full_paired_preloaded_probe._run_arm", fake), patch("h334.frozen_full_paired_preloaded_probe._check_pair", lambda *_args: None):
            with self.assertRaises(PairedProbeRefused):
                run_paired_exposed_preloaded_probe(
                    object(), object(), object(), object())

    def test_no_model_call_at_import_and_all_categories_are_bounded(self):
        for failure in ("abstain", "wrong_focus_or_state",
                        "wrong_code", "source_state_mismatch"):
            if failure == "wrong_focus_or_state":
                value = row(failure=failure)
            elif failure == "abstain":
                value = row()
            else:
                value = row(failure=failure)
            self.assertEqual(_safe_arm(value)[1], failure)
        self.assertEqual(_safe_arm(row(success=True))[0], True)


if __name__ == "__main__":
    unittest.main()
