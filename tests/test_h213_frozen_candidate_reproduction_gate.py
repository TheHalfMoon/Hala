"""H213 adversarial frozen image/SHA provenance controls, no model/media."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from h212.procedural_native_gui_candidates import COUNT, SCHEMA
from h213.frozen_candidate_reproduction_gate import (
    MAX_FROZEN_BYTES,
    canonical_frozen_body, verify_checked_in_source,
    verify_exact_metadata_text,
)


class FrozenCandidateReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Path(__file__).resolve().parent.parent / "h212" / (
            "RESEARCH_CANDIDATE_METADATA_NOT_ADMITTED.json"
        )
        cls.body = cls.p.read_text(encoding="utf-8")
        cls.doc = json.loads(cls.body)
        cls.canonical = canonical_frozen_body(cls.doc)

    def test_actual_reproduction_rerenders_all_96_images(self):
        result = verify_checked_in_source()
        self.assertEqual(result["candidate_cases_checked"], 96)
        # Windows pins original PNG *bytes*. Linux pins the original
        # Windows metadata SHA while comparing the identical RGBA pixels:
        # Pillow's PNG stream encoding is not portable across OS.
        self.assertEqual(result["png_sha_matches"]
                         + result.get("png_encoding_differences", 0), 96)
        self.assertEqual(result["rgba_sha_matches"], 96)
        self.assertFalse(result["official_train_admitted"])
        self.assertFalse(result["official_dev_sealed"])
        self.assertFalse(result["official_ood_evaluated"])
        self.assertEqual(result["model_calls"], 0)

    def test_committed_snapshot_matches_exact_canonical_form(self):
        self.assertEqual(self.body, self.canonical)
        self.assertEqual(len(self.doc["cases"]), COUNT)
        self.assertEqual(self.doc["schema"], SCHEMA)
        self.assertEqual(verify_exact_metadata_text(
            self.body, self.canonical)["png_sha_matches"], 96)

    def test_crlf_checkouts_are_normalized_but_not_semantically_rewritten(self):
        crlf = self.canonical.replace("\n", "\r\n")
        self.assertEqual(verify_exact_metadata_text(
            crlf, self.canonical)["rgba_sha_matches"], 96)

    def expect_refused(self, altered):
        with self.assertRaises(ValueError):
            verify_exact_metadata_text(altered, self.canonical)

    def test_any_png_digest_mutation_is_rejected(self):
        sha = self.doc["cases"][0]["image_png_sha256"]
        self.expect_refused(self.body.replace(sha, "f"*64 if sha!="f"*64 else "e"*64,1))

    def test_any_rgba_digest_mutation_is_rejected(self):
        sha = self.doc["cases"][95]["image_decoded_rgba_sha256"]
        self.expect_refused(self.body.replace(sha, "e"*64 if sha!="e"*64 else "f"*64,1))

    def test_generator_source_sha_mutation_is_rejected(self):
        sha = self.doc["generator_source_sha256"]
        self.expect_refused(self.body.replace(sha, "0"*64 if sha!="0"*64 else "1"*64,1))

    def test_research_role_relabel_or_split_swapping_denied(self):
        self.expect_refused(self.body.replace(
            '"role": "RESEARCH_DEV_CANDIDATE_NOT_SEALED"',
            '"role": "RESEARCH_TRAIN_CANDIDATE_NOT_ADMITTED"',1))
        doc=json.loads(self.body)
        doc["cases"][0],doc["cases"][1]=doc["cases"][1],doc["cases"][0]
        self.expect_refused(json.dumps(doc,sort_keys=True,indent=2)+"\n")

    def test_arabic_or_english_language_drift_denied(self):
        self.expect_refused(self.body.replace('"language": "ar"', '"language": "en"',1))

    def test_training_admission_gold_seal_and_ood_flip_denied(self):
        for field in ("official_admitted", "dev_gold_sealed", "official_ood_evaluated",
                      "historical_source_history_complete", "old_rgba_history_complete"):
            with self.subTest(field=field):
                self.expect_refused(self.body.replace(
                    '"'+field+'": false', '"'+field+'": true',1))

    def test_forged_extra_metadata_key_and_duplicate_key_denied(self):
        self.expect_refused(self.body.replace(
            '{\n  "cases":', '{\n  "extra": true,\n  "cases":',1))
        self.expect_refused(self.body.replace(
            '{\n  "cases":',
            '{\n  "schema": "hala.h212.generated-original-gui-candidates.v1",\n  "cases":',1))

    def test_nonstandard_nan_json_and_wrong_field_types_denied(self):
        self.expect_refused(self.body.replace(
            '"gpu_optimizer_steps": 0','"gpu_optimizer_steps": NaN',1))
        self.expect_refused(self.body.replace(
            '"registered_candidate_count": 96','"registered_candidate_count": true',1))

    def test_truncation_oversize_and_wrong_input_type_denied(self):
        self.expect_refused(self.body[:-120])
        self.expect_refused(self.body+"X"*(MAX_FROZEN_BYTES+1))
        self.expect_refused(None)

    def test_cannot_create_an_admitted_manifest(self):
        fake=json.loads(self.body)
        fake["official_admitted"]=True
        with self.assertRaises(ValueError):
            canonical_frozen_body(fake)


if __name__=="__main__":
    unittest.main()
