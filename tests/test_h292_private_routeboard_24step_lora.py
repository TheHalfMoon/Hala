"""H292 source-only adversarial contract tests; no model, media or gold."""
from __future__ import annotations

import copy
import json
import unittest

from h280.verify_private_native_aggregate import EvidenceRejected
from h292 import private_routeboard_24step_lora as pilot
from h292 import verify_private_routeboard_lora as verifier


def invented_report() -> dict:
    steps = [
        {"changed": 112, "coordinates_supervised": True,
         "loss_finite": True, "gradient_nonzero": True}
        for _ in range(24)
    ]
    def rows(wins: tuple[bool, ...]) -> list[dict]:
        return [
            {"scene": scene[0], "model_generation": 1, "grammar_call": 1,
             "strict": True, "kind": "click", "native_click": True,
             "task_success": wins[i]}
            for i, scene in enumerate(pilot.ROUTE_SCENES)
        ]
    before = tuple("invented-frame-%d" % i for i in range(4))
    train = frozenset("invented-train-%02d" % i for i in range(24))
    return pilot.verified_report(
        steps, rows((True, False, False, False)),
        rows((True, False, True, False)),
        before, before, {"free_t4": True, "devices": 2}, train)


class H292ClosedPrivateResearch(unittest.TestCase):
    def test_pinned_original_sources_and_prospective_roles(self):
        self.assertEqual(pilot.PIN, verifier.SOURCE_COMMIT)
        self.assertEqual(pilot.EVAL_SOURCE_SHA,
                         "5a7608f533897ac9c10d41b3d97c4a0b245a000b")
        self.assertEqual(pilot.ROUTE_BLOB, verifier.EVAL_BLOB)
        self.assertEqual(pilot.SHIFT_BLOB, verifier.TRAIN_BLOB)
        self.assertEqual(pilot.MODEL_REV, verifier.MODEL_REVISION)
        self.assertEqual(pilot.REPORT, verifier.REPORT_NAME)
        self.assertEqual(pilot.STATUS, verifier.STATUS)
        self.assertEqual(pilot.N_STEPS, 24)
        self.assertEqual(pilot.STEP_INDICES,
                         tuple(i for pair in zip(range(12), range(12, 24))
                               for i in pair))
        self.assertEqual(tuple(s[0] for s in pilot.ROUTE_SCENES),
                         verifier.ORIGINAL_SCENES)
        self.assertEqual(tuple(s[1] for s in pilot.ROUTE_SCENES),
                         ("en", "en", "ar", "ar"))
        self.assertEqual(len({s[-1] for s in pilot.ROUTE_SCENES}), 3)

    def test_positive_and_negative_delta_are_descriptive_not_gain(self):
        doc = invented_report()
        self.assertEqual(doc["paired_delta"], 1)
        summary = verifier.verify_bytes(json.dumps(doc).encode())
        self.assertEqual(summary["real_adapter_optimizer_steps"], 24)
        self.assertEqual(summary["paired_source_cases"], 4)
        self.assertEqual(summary["paired_delta"], 1)
        self.assertFalse(summary["generalizable_model_gain_established"])
        negative = copy.deepcopy(doc)
        negative["paired_adapted"][2]["task_success"] = False
        negative["paired_pretrained"][1]["task_success"] = True
        negative["paired_adapted_successes"] = 1
        negative["paired_pretrained_successes"] = 2
        negative["paired_delta"] = -1
        self.assertEqual(verifier.closed_report(negative)["paired_delta"], -1)

    def test_no_fabricated_score_from_changed_adapter_tensors(self):
        doc = invented_report()
        doc["paired_delta"] = 3
        with self.assertRaises(EvidenceRejected):
            verifier.closed_report(doc)
        doc = invented_report()
        doc["research_train_optimizer_steps"] = 8
        with self.assertRaises(EvidenceRejected):
            verifier.closed_report(doc)

    def test_refuse_private_fields_and_false_model_claims(self):
        for field, value in (
            ("raw_model_output", "secret"), ("target_coordinates", [1, 2]),
            ("generalizable_hala_improvement_proven", True),
            ("official_train_dev_ood_admitted", True),
            ("adapter_weights_published", True),
        ):
            with self.subTest(field=field):
                doc = invented_report()
                doc[field] = value
                with self.assertRaises(EvidenceRejected):
                    verifier.closed_report(doc)

    def test_refuse_row_drift_and_inconsistent_native_success(self):
        for change in ("scene", "grammar_call", "native_click", "extra"):
            with self.subTest(change=change):
                doc = invented_report()
                row = doc["paired_adapted"][2]
                if change == "scene":
                    row["scene"] = "dev_00"
                elif change == "grammar_call":
                    row["grammar_call"] = 0
                elif change == "native_click":
                    row["native_click"] = False
                else:
                    row["private_point"] = [100, 200]
                with self.assertRaises(EvidenceRejected):
                    verifier.closed_report(doc)

    def test_duplicate_keys_and_nonfinite_json_rejected(self):
        raw = json.dumps(invented_report())
        with self.assertRaises(EvidenceRejected):
            verifier.verify_bytes(raw.replace(
                '"paired_delta": 1', '"paired_delta": 1, "paired_delta": 1').encode())
        with self.assertRaises(EvidenceRejected):
            verifier.verify_bytes(raw.replace(
                '"paired_delta": 1', '"paired_delta": NaN').encode())

    def test_training_and_eval_image_overlap_refused(self):
        doc = invented_report()
        self.assertTrue(doc["train_eval_exact_png_disjoint"])
        with self.assertRaises(ValueError):
            pilot.verified_report(
                [{"changed": 1, "coordinates_supervised": True,
                  "loss_finite": True, "gradient_nonzero": True}
                 for _ in range(24)],
                doc["paired_pretrained"], doc["paired_adapted"],
                ("a", "b", "c", "d"), ("a", "b", "c", "d"),
                {"free_t4": True, "devices": 2},
                frozenset(["a"] + ["t%d" % i for i in range(23)]))

    def test_stage_failure_is_not_accuracy(self):
        report = pilot.failure_receipt("SOURCE", "ValueError")
        self.assertEqual(report["optimizer_steps_verified"], 0)
        self.assertFalse(report["paired_gui_task_score_verified"])
        self.assertFalse(report["model_improvement_proven"])
        self.assertNotIn("raw_model_output", report)


if __name__ == "__main__":
    unittest.main()
