"""H426 synthetic real-processor sweep fail-closed source-only tests."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "h426_processor_sweep", ROOT / "h426/synthetic_full_processor_sweep.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fake(split, index, domain, language):
    sha = hashlib.sha256(f"{split}:{domain}:{index}".encode()).hexdigest()
    return {
        "split": split,
        "domain": domain,
        "language": language,
        "synthetic_source_image_sha256": sha,
        "synthetic_source_family": "invented-ui-family-0" if split == "train"
                                   else "invented-ui-family-3",
        "source_is_admitted_to_train": False,
        "new_model_trained": False,
        "model_release_qualified": False,
        "image_patch_tokens": 228,
        "weighted_coordinate_tokens": 12,
    }


class ProcessorSweepContractTests(unittest.TestCase):
    def test_paired_multisplit_aggregates_without_case_data(self):
        result = m.sweep_paired_processor(fake, 2)
        self.assertEqual(result["generated_source_images_verified"], 12)
        self.assertEqual(result["image_conditioned_actions_verified"], 24)
        self.assertEqual(len(result["by_split_domain_language"]), 12)
        self.assertTrue(all(v == 2 for v in result["by_split_domain_language"].values()))
        self.assertFalse(result["release_qualified"])
        self.assertFalse(result["new_model_trained"])
        self.assertNotIn("synthetic_source_image_sha256", result)
        self.assertNotIn("raw_predictions", result)

    def test_zero_invalid_bool_and_large_count_refused(self):
        for n in (0, -1, 65, True, False, 1.5, "8"):
            with self.subTest(count=n), self.assertRaises(ValueError):
                m.sweep_paired_processor(fake, n)
        with self.assertRaises(ValueError):
            m.sweep_paired_processor(None, 1)

    def test_untrusted_processor_report_refused(self):
        for mutate in (
            {"source_is_admitted_to_train": True},
            {"new_model_trained": True},
            {"model_release_qualified": True},
            {"weighted_coordinate_tokens": 0},
            {"image_patch_tokens": 0},
            {"synthetic_source_image_sha256": "wrong"},
            {"synthetic_source_family": "unverified"},
            {"domain": "other"},
            {"language": "other"},
        ):
            def bad(split, index, domain, language):
                return {**fake(split, index, domain, language), **mutate}
            with self.subTest(change=mutate), self.assertRaises(ValueError):
                m.sweep_paired_processor(bad, 1)

    def test_mismatched_bilingual_source_sha_refused(self):
        def mismatch(split, index, domain, language):
            record = fake(split, index, domain, language)
            if language == "ar":
                record["synthetic_source_image_sha256"] = "0" * 64
            return record
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            m.sweep_paired_processor(mismatch, 1)

    def test_duplicate_image_sha_across_cases_refused(self):
        def duplicate(split, index, domain, language):
            return {**fake(split, index, domain, language),
                    "synthetic_source_image_sha256": "b" * 64}
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            m.sweep_paired_processor(duplicate, 2)

    def test_train_dev_family_overlap_refused(self):
        def overlap(split, index, domain, language):
            return {**fake(split, index, domain, language),
                    "synthetic_source_family": "invented-ui-family-0"}
        with self.assertRaisesRegex(ValueError, "family overlap"):
            m.sweep_paired_processor(overlap, 1)


if __name__ == "__main__":
    unittest.main()
