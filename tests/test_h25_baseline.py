"""H25 CPU-only strict prospective base-model benchmark source gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from h25 import baseline as b, corpus, projection_v2


def perfect_outcomes():
    rows = []
    for projection, expected, state in b.expected_rows():
        ref = projection["source_ref"]
        raw = json.dumps(expected)
        row = {
            "id": ref["id"], "family": ref["family"], "language": ref["language"],
            "state": state, "raw": raw, "error": None,
            "wall_seconds": 0.8,
        }
        row.update(b.evaluate_raw(expected, raw))
        rows.append(row)
    return rows


class H25BaselineTests(unittest.TestCase):
    def setUp(self):
        self.paired = b.expected_rows()

    def test_42_dev_cases_and_bilingual_partition(self):
        self.assertEqual(len(self.paired), 42)
        self.assertEqual({x["source_ref"]["family"] for x, _, _ in self.paired},
                         set(corpus.SPLITS["dev"]))
        self.assertEqual(sum(x["source_ref"]["language"] == "en"
                             for x, _, _ in self.paired), 21)
        self.assertEqual(sum(x["source_ref"]["language"] == "ar"
                             for x, _, _ in self.paired), 21)

    def test_all_seven_expected_pairs_supported(self):
        expected = {(e["tool"], e["subgoal"]) for _, e, _ in self.paired}
        allowed = {(tool, sub) for tool, values in b.MODEL_SUBGOALS.items()
                   for sub in values}
        self.assertEqual(expected, allowed)
        self.assertEqual(len(expected), 7)
        for _, oracle, _ in self.paired:
            self.assertEqual(b.typed_proposal(json.dumps(oracle)), oracle)

    def test_actual_prompt_excludes_oracle_tracking_and_family(self):
        for row, expected, state in self.paired:
            raw = b.user_payload(row["model_input"])
            visible = json.loads(raw)
            self.assertEqual(set(visible), {"goal", "locale", "observation"})
            self.assertNotIn('"source_ref"', raw)
            self.assertNotIn('"oracle_private"', raw)
            self.assertNotIn('"expected"', raw)
            self.assertNotIn('"subgoal"', raw)
            self.assertNotIn('"partition"', raw)
            self.assertNotIn('"state"', raw)
            self.assertNotIn(row["source_ref"]["id"], raw)
            self.assertNotIn(row["source_ref"]["family"], raw)

    def test_wrong_input_type_cannot_enter_model_prompt(self):
        with self.assertRaises(ValueError):
            b.user_payload({"goal": "x", "observation": {}, "source_ref": {}})
        with self.assertRaises(ValueError):
            b.user_payload({"goal": "x", "locale": "en-US"})
        with self.assertRaises(ValueError):
            b.user_payload([1, 2, 3])

    def test_hostile_tool_kind_extra_field_and_crossed_pairs_rejected(self):
        candidates = (
            {"kind": "abstain", "tool": "abstain", "subgoal": "reject_external_command"},
            {"kind": "next", "tool": "shell", "subgoal": "delete_files"},
            {"kind": "next", "tool": "visual", "subgoal": "reject_external_command"},
            {"kind": "next", "tool": "observe", "subgoal": "request_missing_parameter"},
            {"kind": "next", "tool": "visual", "subgoal": "locate_verified_target", "execute": True},
            {"kind": "next", "tool": [], "subgoal": "locate_verified_target"},
            {"kind": "next", "tool": "clarify", "subgoal": "human readable command"},
        )
        for item in candidates:
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    b.typed_proposal(json.dumps(item))

    def test_valid_json_not_automatically_a_valid_proposal(self):
        oracle = self.paired[0][1]
        raw = json.dumps({"kind": "abstain", "tool": "abstain",
                          "subgoal": "reject_external_command"})
        scored = b.evaluate_raw(oracle, raw)
        self.assertFalse(scored["valid"])
        self.assertFalse(scored["exact_correct"])
        self.assertFalse(scored["tool_correct"])

    def test_reference_scoring_and_seven_state_summary(self):
        outcomes = perfect_outcomes()
        report = b.summarize(outcomes)
        self.assertEqual(report["all"]["cases"], 42)
        self.assertEqual(report["en"]["cases"], 21)
        self.assertEqual(report["ar"]["cases"], 21)
        self.assertEqual(report["all"]["exact_correct"], 42)
        self.assertEqual({k: v["cases"] for k, v in report["all"]["per_state"].items()},
                         {s: 6 for s in corpus.STATES})

    def test_tampered_result_summaries_rejected(self):
        original = perfect_outcomes()
        for mutate in (
            lambda a: a[0].update({"tool_correct": False}),
            lambda a: a[0].update({"family": "museum_accessions"}),
            lambda a: a[0].update({"language": "fr"}),
            lambda a: a[0].update({"state": "unsafe_user"}),
            lambda a: a[1].update({"id": a[0]["id"]}),
            lambda a: a[0].update({"raw": '{"kind":"next"}'}),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                b.summarize(changed)

    def test_result_error_classification_is_consistent_with_raw_reply(self):
        correct = perfect_outcomes()
        wrong = copy.deepcopy(correct)
        wrong[0]["error"] = "model_inference_error"
        with self.assertRaises(ValueError):
            b.summarize(wrong)
        wrong = copy.deepcopy(correct)
        wrong[0]["error_type"] = "RuntimeError"
        with self.assertRaises(ValueError):
            b.summarize(wrong)
        wrong = copy.deepcopy(correct)
        wrong[0]["raw"] = None
        wrong[0].update(valid=False, tool_correct=False, exact_correct=False,
                        predicted_tool=None, error="model_inference_error")
        with self.assertRaises(ValueError):
            b.summarize(wrong)
        wrong[0]["error_type"] = "RuntimeError"
        self.assertEqual(b.summarize(wrong)["all"]["inference_errors"],1)
        wrong[0]["error_type"] = "x"*101
        with self.assertRaises(ValueError):
            b.summarize(wrong)

    def test_free_text_json_failure_is_invalid_not_inference_error(self):
        outcomes = perfect_outcomes()
        first = outcomes[0]
        first["raw"] = '{"kind":"abstain","tool":"abstain","subgoal":"reject_external_command"}'
        first.update(b.evaluate_raw(self.paired[0][1],first["raw"]))
        reported = b.summarize(outcomes)
        self.assertEqual(reported["all"]["inference_errors"],0)
        self.assertEqual(reported["all"]["typed_valid"],41)

    def test_missing_case_rejected(self):
        with self.assertRaises(ValueError):
            b.summarize(perfect_outcomes()[:-1])

    def test_inference_error_counts_as_failure_not_success(self):
        rows = perfect_outcomes()
        row = rows[0]
        row.update(raw=None, valid=False, tool_correct=False,
                   exact_correct=False, predicted_tool=None,
                   error="model_inference_error", error_type="RuntimeError")
        summary = b.summarize(rows)
        self.assertEqual(summary["all"]["inference_errors"], 1)
        self.assertEqual(summary["all"]["exact_correct"], 41)

    def test_preflight_rejects_windows_before_downloading_or_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "new"
            with self.assertRaises(RuntimeError):
                b.remote_admit("a" * 40, root)
            self.assertFalse(root.exists())

    def test_runner_not_importing_model_weights_on_cpu(self):
        self.assertFalse(hasattr(b, "torch"))
        self.assertFalse(hasattr(b, "model"))
        self.assertFalse(hasattr(b, "processor"))

    def test_protocol_does_not_mislabel_h25_as_test(self):
        self.assertIn("DEV", b.__doc__)
        self.assertIn("No training", b.__doc__)
        self.assertEqual(b.PROTOCOL, "h25-seven-pair-dev-first-v1")
        self.assertEqual(b.MAX_TOKENS, 160)


if __name__ == "__main__":
    unittest.main()
