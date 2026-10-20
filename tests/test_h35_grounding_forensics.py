"""CPU-only, retained-digest H35 failure forensics; no model inference."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from h34.corpus import build_cases
from h35.forensics import analyze, classify

RETAINED = Path("reviews/h34-kaggle-run-20261020-v2")
RAW_SHA = "f9dfbba4a4d8adee149caa0f5b2c18def8b9383f472c0fc1207d171996809bcf"


class H35GroundingForensicsTests(unittest.TestCase):
    def setUp(self):
        self.case = build_cases()[0]

    def test_actual_immutable_kaggle_run_classification(self):
        report = analyze(RETAINED, RAW_SHA)
        self.assertTrue(report["replayed_h34_integrity"])
        self.assertFalse(report["model_qualified"])
        self.assertTrue(report["retrospective_dev_only"])
        self.assertEqual(report["raw_sha256"], RAW_SHA)
        self.assertEqual(report["attempted_records"], 32)
        self.assertEqual(report["class_counts"], {
            "hit": 8, "wrong_control": 2, "off_controls": 22,
        })
        self.assertEqual(report["groups"]["language"]["ar"]["hit"], 5)
        self.assertEqual(report["groups"]["language"]["en"]["hit"], 3)
        self.assertEqual(report["groups"]["layout"]["column"]["hit"], 0)
        self.assertEqual(report["groups"]["layout"]["rtl_row"]["hit"], 4)
        self.assertEqual(report["groups"]["domain"]["calendar"]["hit"], 1)
        self.assertEqual(len(report["cases"]), 32)
        self.assertEqual(sum(report["class_counts"].values()), 32)

    def test_known_unique_hit_and_wrong_control(self):
        target = next(x for x in self.case["controls"] if x["is_expected"])
        other = next(x for x in self.case["controls"] if not x["is_expected"])
        for control, category in ((target, "hit"), (other, "wrong_control")):
            x, y, w, h = control["bounds"]
            raw = json.dumps({"point": [(x + w / 2) / self.case["width"],
                                        (y + h / 2) / self.case["height"]]})
            detail = classify(self.case, raw)
            self.assertEqual(detail["class"], category)
            self.assertEqual(detail["point_px"], [x + w / 2, y + h / 2])

    def test_off_control_overlap_abstain_and_malformed_are_distinct(self):
        self.assertEqual(classify(self.case, '{"point":[0.001,0.001]}')["class"],
                         "off_controls")
        self.assertEqual(classify(self.case, '{"abstain":true}')["class"], "abstain")
        self.assertEqual(classify(self.case, '{"point":[NaN,0.1]}')["class"],
                         "parse_invalid")
        self.assertEqual(classify(self.case, None, "TimeoutError")["class"],
                         "inference_error")
        collision = copy.deepcopy(self.case)
        collision["controls"][1]["bounds"] = collision["controls"][0]["bounds"]
        x, y, w, h = collision["controls"][0]["bounds"]
        raw = json.dumps({"point": [(x+w/2)/960, (y+h/2)/540]})
        self.assertEqual(classify(collision, raw)["class"], "overlapping_controls")

    def test_digest_required_and_tampering_fails_closed(self):
        with self.assertRaises(ValueError):
            analyze(RETAINED, "0" * 64)

    def test_independent_h34_boolean_scoring_agrees(self):
        report = analyze(RETAINED, RAW_SHA)
        for row in report["cases"]:
            self.assertEqual(row["target_hit"], row["class"] == "hit")
            self.assertEqual(row["gate_valid_unique"],
                             row["class"] in ("hit", "wrong_control"))


    def test_partial_run_and_backend_errors_keep_full_denominator(self):
        from tests.test_h34_bilingual_visual import evidence
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            digest, _, _ = evidence(path, count=2)
            result = analyze(path, digest)
            self.assertEqual(result["attempted_records"], 2)
            self.assertEqual(result["denominator"], 32)
            self.assertEqual(result["class_counts"], {"abstain": 2, "not_attempted": 30})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            digest, _, _ = evidence(path, count=3, error="TimeoutError")
            result = analyze(path, digest)
            self.assertEqual(result["class_counts"], {"inference_error": 3, "not_attempted": 29})

    def test_retained_derived_report_matches_replay(self):
        report_path = Path("docs/reviews/h35-h34-v2-grounding-failures.json")
        if not report_path.exists():
            self.skipTest("H35 derived report not generated yet")
        report = analyze(RETAINED, RAW_SHA)
        canonical = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        self.assertEqual(report_path.read_bytes(), canonical.encode("utf-8"))

    def test_no_run_side_effects_or_hidden_label_leak(self):
        report = analyze(RETAINED, RAW_SHA)
        self.assertFalse(report["native_actions_executed"])
        self.assertFalse(report["new_model_inference"])
        self.assertFalse(report["independent_test_qualified"])
        self.assertTrue(all("expected_label" not in item for item in report["cases"]))


if __name__ == "__main__":
    unittest.main()
