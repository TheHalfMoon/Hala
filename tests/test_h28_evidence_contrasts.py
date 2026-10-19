"""H28 evidence-contrast CPU-only scientific contract tests, no model."""
import copy
import json
import unittest

from h28 import contrasts


class H28EvidenceContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = contrasts.dataset()
        cls.receipt = contrasts.validate(cls.data)

    def test_complete_balanced_human_goal_counterfactuals(self):
        self.assertEqual(self.receipt["cases"], 72)
        self.assertEqual(self.receipt["pairs"], 36)
        self.assertEqual(self.receipt["en"], 36)
        self.assertEqual(self.receipt["ar"], 36)
        self.assertFalse(self.receipt["independent_test"])
        self.assertFalse(self.receipt["causal_model_claim"])
        self.assertEqual(self.receipt["actual_model_runs"], 0)
        self.assertEqual(len(set(contrasts.FAMILIES)), 3)
        self.assertEqual(len(contrasts.PAIRS), 6)
        self.assertTrue(set(contrasts.FAMILIES).isdisjoint({
            "civic_permit_desk", "cold_chain_handoff",
            "community_event_registry", "marine_specimen_log",
            "library_circulation", "public_transit_access", "seedbank_curation"
        }))

    def test_each_pair_has_identical_goal_one_changed_observation_dimension(self):
        for a, b in zip(self.data["records"][::2], self.data["records"][1::2]):
            self.assertEqual(a["model_input"]["goal"], b["model_input"]["goal"])
            self.assertNotEqual(a["private_reference"]["code"], b["private_reference"]["code"])
            va, vb = a["model_input"]["observation"], b["model_input"]["observation"]
            self.assertEqual(set(va), set(vb))
            self.assertEqual(len([k for k in va if va[k] != vb[k]]), 1)
            self.assertEqual(a["source_ref"]["variant"], "control")
            self.assertEqual(b["source_ref"]["variant"], "challenger")

    def test_model_messages_contain_no_label_state_or_source_id(self):
        for case in self.data["records"]:
            for instruction_locale in ("en", "ar"):
                model = contrasts.messages(case, instruction_locale)
                self.assertEqual([m["role"] for m in model], ["system", "user"])
                payload = json.loads(model[1]["content"])
                self.assertEqual(payload, case["model_input"])
                whole = json.dumps(model, ensure_ascii=False)
                self.assertNotIn(case["source_ref"]["request_key"], whole)
                self.assertNotIn("source_ref", whole)
                self.assertNotIn("private_reference", whole)

    def test_reject_tampered_oracle_evidence_or_tracking_envelopes(self):
        altered = copy.deepcopy(self.data)
        altered["records"][0]["private_reference"]["code"] = "C7"
        with self.assertRaises(ValueError):
            contrasts.validate(altered)
        altered = copy.deepcopy(self.data)
        altered["records"][1]["model_input"]["goal"] = "Changed goal"
        with self.assertRaises(ValueError):
            contrasts.validate(altered)
        altered = copy.deepcopy(self.data)
        altered["records"][1]["model_input"]["observation"]["frame_current"] = "changed"
        with self.assertRaises(ValueError):
            contrasts.validate(altered)
        malformed = copy.deepcopy(self.data["records"][0])
        malformed["model_input"]["oracle"] = "C1"
        with self.assertRaises(ValueError):
            contrasts.messages(malformed)
        with self.assertRaises(ValueError):
            contrasts.messages({"model_input": {}})

    def test_all_c1_synthetic_fixture_reports_no_pair_responsiveness(self):
        score = contrasts.score_fixture_codes(["C1"] * 72)
        self.assertEqual(score["semantic_correct"], 36)
        self.assertEqual(score["both_correct_pairs"], 0)
        self.assertEqual(score["unresponsive_identical_code_pairs"], 36)
        self.assertEqual(score["wire_distribution"]["C1"], 72)
        self.assertTrue(score["source_only_not_actual_inference"])
        self.assertFalse(score["native_authority"])

    def test_pair_categories_require_two_valid_codes_and_partition_pairs(self):
        for raw in ([None] * 72, ["junk"] * 72):
            score = contrasts.score_fixture_codes(raw)
            self.assertEqual(score["unresponsive_identical_code_pairs"], 0)
        none = contrasts.score_fixture_codes([None] * 72)
        junk = contrasts.score_fixture_codes(["junk"] * 72)
        self.assertEqual(none["pair_categories"]["contains_inference_error"], 36)
        self.assertEqual(junk["pair_categories"]["contains_invalid_output"], 36)
        mixed = (["C1", "C2"] * 12 + ["C1", "junk"] * 12
                 + [None, "C1"] * 12)
        score = contrasts.score_fixture_codes(mixed)
        self.assertEqual(score["pair_categories"], {
            "both_valid_codes": 12,
            "contains_inference_error": 12,
            "contains_invalid_output": 12,
        })
        self.assertEqual(score["unresponsive_identical_code_pairs"], 0)
        error_precedence = contrasts.score_fixture_codes([None, "junk"] * 36)
        self.assertEqual(error_precedence["pair_categories"]["contains_inference_error"], 36)

    def test_identical_valid_c0_and_c1_pairs_are_counted(self):
        for raw in ("C0", "C1"):
            score = contrasts.score_fixture_codes([raw] * 72)
            self.assertEqual(score["pair_categories"]["both_valid_codes"], 36)
            self.assertEqual(score["unresponsive_identical_code_pairs"], 36)

    def test_perfect_oracle_fixture_is_machinery_not_model_evidence(self):
        gold = [x["private_reference"]["code"] for x in self.data["records"]]
        score = contrasts.score_fixture_codes(gold)
        self.assertEqual(score["semantic_correct"], 72)
        self.assertEqual(score["both_correct_pairs"], 36)
        self.assertEqual(score["unresponsive_identical_code_pairs"], 0)
        self.assertEqual(score["by_language"]["ar"]["correct"], 36)
        self.assertEqual(score["by_language"]["en"]["correct"], 36)
        self.assertTrue(score["not_a_heldout_test"])

    def test_withholding_does_not_become_any_proven_action_or_refusal(self):
        score = contrasts.score_fixture_codes(["C0"] * 72)
        self.assertEqual(score["semantic_correct"], 6)  # 3 families × 2 languages × permission challenger
        self.assertEqual(score["both_correct_pairs"], 0)
        self.assertFalse(score["native_authority"])

    def test_reject_broken_prediction_list_and_types(self):
        for raw in (None, {}, "C1", [], ["C1"] * 71, ["C1"] * 73):
            with self.subTest(raw=type(raw).__name__), self.assertRaises(ValueError):
                contrasts.score_fixture_codes(raw)
        broken = ["C1"] * 72
        broken[1] = 8
        with self.assertRaises(ValueError):
            contrasts.score_fixture_codes(broken)
        self.assertFalse(hasattr(contrasts, "execute"))
        self.assertFalse(hasattr(contrasts, "approve"))


if __name__ == "__main__":
    unittest.main()
