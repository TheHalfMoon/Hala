"""H343 tests use only original public source HTML; never private media."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import json
import shutil
import unittest

from h343.independent_rights_review_request import (
    FAMILIES, QUESTIONS, build_public_review_request,
    verify_request_integrity_only,
)

ROOT = Path(__file__).resolve().parents[1]


class TestIndependentRightsRequest(unittest.TestCase):
    def setUp(self):
        self.packet = build_public_review_request(ROOT)

    def test_exact_public_source_bound_request(self):
        p = self.packet
        self.assertEqual(len(p["families"]), 2)
        self.assertEqual(sum(f["exposed_candidate_scenes"]
                             for f in p["families"]), 144)
        self.assertEqual(sum(f["exposed_candidate_action_stages"]
                             for f in p["families"]), 288)
        self.assertEqual(len(p["request_sha256"]), 64)
        self.assertEqual(tuple(p["questions_for_independent_reviewer"]), QUESTIONS)

    def test_explicit_no_training_or_release(self):
        p = self.packet
        self.assertFalse(p["can_train"])
        self.assertFalse(p["can_release"])
        self.assertFalse(p["official_h199_train_admitted"])
        self.assertFalse(p["blind_ood_qualified"])
        self.assertFalse(p["adapted_gain_proven"])
        self.assertEqual(p["optimizer_steps"], 0)
        self.assertTrue(all(not f["screenshot_ml_training_rights_approved"]
                            for f in p["families"]))

    def test_no_raw_media_gold_or_prompts(self):
        serialized = json.dumps(self.packet, sort_keys=True)
        self.assertLess(len(serialized), 3500)
        for forbidden in ("data:image", "PNG", "base64", "coordinate",
                          "model_prediction", "private_receipt", "SIM-"):
            self.assertNotIn(forbidden, serialized)

    def test_deterministic_digest(self):
        self.assertEqual(build_public_review_request(ROOT), self.packet)
        self.assertEqual(verify_request_integrity_only(self.packet, ROOT),
                         {"request_integrity": True,
                          "independent_rights_approval": False,
                          "official_train_admission": False,
                          "can_train": False, "can_release": False})

    def test_self_approval_refused(self):
        for field in ("can_train", "official_h199_train_admitted",
                      "blind_ood_qualified", "adapted_gain_proven"):
            bad = copy.deepcopy(self.packet)
            bad[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_request_integrity_only(bad, ROOT)

    def test_digest_or_source_identity_tamper_refused(self):
        for mutate in ("digest", "blob", "review", "extra"):
            bad = copy.deepcopy(self.packet)
            if mutate == "digest":
                bad["request_sha256"] = "0" * 64
            elif mutate == "blob":
                bad["families"][0]["pinned_git_blob"] = "0" * 40
            elif mutate == "review":
                bad["review_status"] = "APPROVED"
            else:
                bad["owner_says_approved"] = True
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                verify_request_integrity_only(bad, ROOT)

    def test_modified_original_source_refused(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for _, relative, *_ in FAMILIES:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            self.assertEqual(build_public_review_request(root), self.packet)
            target = root / FAMILIES[0][1]
            target.write_bytes(target.read_bytes() + b"\n<!-- tampered -->")
            with self.assertRaises(ValueError):
                build_public_review_request(root)

    def test_missing_source_refused(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_public_review_request(Path(tmp))

    def test_no_external_approval_from_valid_packet(self):
        report = verify_request_integrity_only(self.packet, ROOT)
        self.assertFalse(report["independent_rights_approval"])
        self.assertFalse(report["official_train_admission"])


if __name__ == "__main__":
    unittest.main()
