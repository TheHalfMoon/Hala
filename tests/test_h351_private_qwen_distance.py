"""H351 invented-only evidence-bound prospective private Qwen proximity tests.

NO GPU, no Kaggle submission and no actual model-output claim.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from h337.source_miss_distance_bands import CATEGORIES, ROLE
from h351.private_qwen_miss_distance import (
    FLAG, PIN, safe_report, validate_observed_counts, qualify_free,
)
from h351.verify_private_receipt import (
    parse_unique, verify_success, verify_stage_only,
)
from h351.build_private_payload import build

def source_counts():
    groups = {}
    for language in ("en", "ar"):
        categories = {key: 0 for key in CATEGORIES}
        categories["correct_open_and_focus"] = 6 if language == "en" else 4
        categories["miss_farther_than_48px"] = 0 if language == "en" else 2
        groups[language] = {
            "scenes": 6, "native_actions": 6, "categories": categories}
    return {
        "status": ROLE, "source_public_exposed_scenes": 12,
        "by_language": groups, "source_only_not_model_attested": True,
        "new_model_generations": 0, "new_optimizer_steps": 0,
        "official_train_admitted": False, "fresh_blind_ood": False,
        "learned_gain_proven": False, "can_release": False,
    }

def summaries():
    return [{
        "model_generations_via_wrapped_generate": 1,
        "model_time_token_filters_installed": 1,
        "distinct_original_pngs_observed": 1,
        "type_calls": 0,
        "pretrained_model_identity_attested": False,
        "real_owner_private_gpu_run_attested": False,
        "generalizable_hala_gain_proven": False,
    } for _ in range(12)]

HW = {"free_owner_private_t4": True, "t4_count": 2}

class H351ProtocolTests(unittest.TestCase):
    def test_aggregate_claims_require_twelve_model_time_filters(self):
        g = validate_observed_counts(source_counts(), summaries())
        self.assertEqual(g["ar"]["categories"]["miss_farther_than_48px"], 2)
        for index in (0, 11):
            forged = summaries()
            forged[index]["model_time_token_filters_installed"] = 0
            with self.assertRaises(ValueError):
                validate_observed_counts(source_counts(), forged)
        with self.assertRaises(ValueError):
            validate_observed_counts(source_counts(), summaries()[:11])

    def test_bad_source_semantics_are_refused(self):
        for mutation in ("train", "ood", "model", "unexpected",
                         "bad-category", "impossible-count", "bad-total"):
            cohort = source_counts()
            if mutation == "train":
                cohort["official_train_admitted"] = True
            elif mutation == "ood":
                cohort["fresh_blind_ood"] = True
            elif mutation == "model":
                cohort["new_model_generations"] = 12
            elif mutation == "unexpected":
                cohort["private_point"] = [1, 2]
            elif mutation == "bad-category":
                cohort["by_language"]["ar"]["categories"]["raw_prediction"] = 1
            elif mutation == "impossible-count":
                cohort["by_language"]["en"]["categories"]["abstain"] = -1
            else:
                cohort["by_language"]["ar"]["native_actions"] = 7
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_observed_counts(cohort, summaries())

    def test_bounded_private_receipt_and_no_leaks(self):
        receipt = safe_report(source_counts(), summaries(), HW)
        raw = json.dumps(receipt).encode()
        self.assertEqual(
            verify_success(parse_unique(raw))["claimed_model_generations"], 12)
        self.assertFalse(verify_success(receipt)["adapted_model_gain_proven"])
        for mutation in ("private_point", "learned", "blind", "bytes", "model",
                         "generations", "sum", "bool"):
            bad = copy.deepcopy(receipt)
            if mutation == "private_point":
                bad["private_point"] = [5, 9]
            elif mutation == "learned":
                bad["adapted_model_gain_proven"] = True
            elif mutation == "blind":
                bad["fresh_blind_ood"] = True
            elif mutation == "bytes":
                bad["source_commit"] = "0" * 40
            elif mutation == "model":
                bad["model_id"] = "unapproved"
            elif mutation == "generations":
                bad["claimed_wrapped_model_generations"] = 11
            elif mutation == "sum":
                bad["by_language"]["ar"]["categories"]["abstain"] = 1
            else:
                bad["new_optimizer_steps"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_success(bad)
        for raw in (b'{"a":1,"a":2}', b'{"x":NaN}',
                    b'{"extra":"\\uZZZZ"}', b"x" * 4097):
            with self.assertRaises(ValueError):
                parse_unique(raw)

    def test_stage_receipt_cannot_be_promoted_to_score(self):
        from h351.private_qwen_miss_distance import stage_only_receipt
        stage = stage_only_receipt()
        self.assertFalse(verify_stage_only(stage)["can_release"])
        with self.assertRaises(ValueError):
            verify_success(stage)

    def test_no_gpu_or_kaggle_artifacts_from_payload_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "private-payload"
            source = Path("h351/private_qwen_miss_distance.py")
            audit = build(source, dest)
            self.assertFalse(audit["gpu_job_submitted"])
            self.assertEqual(audit["images_or_weights_uploaded"], 0)
            self.assertEqual(
                {p.name for p in dest.iterdir()},
                {"h351_private_qwen_miss_distance.py", "kernel-metadata.json"})
            meta = json.loads((dest / "kernel-metadata.json").read_text())
            self.assertEqual(meta["is_private"], "true")
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            self.assertFalse(meta["enable_tpu"] == "true")
            self.assertEqual(meta["dataset_sources"], [])
            script = (dest / "h351_private_qwen_miss_distance.py").read_text()
            self.assertIn(PIN, script)
            self.assertIn(FLAG, script)
            with self.assertRaises(ValueError):
                build(source, dest)

    def test_local_runtime_refuses_before_any_remote_source(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return True
        class FakeTorch:
            cuda = FakeCuda()
        with self.assertRaises(ValueError):
            qualify_free(FakeTorch())

if __name__ == "__main__":
    unittest.main()
