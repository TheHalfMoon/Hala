"""Invented metadata only: prove H315 cannot admit assets or emit asset details."""
import copy
import unittest

from h315.asset_custody_audit import audit

S = "a" * 40
D = "sha256:" + "b" * 64
ITEM = {
    "source_repo": "fictional/fixture", "source_commit": S,
    "asset_path": "invented/icon.svg", "asset_sha256": D,
    "license_spdx": "OFL-1.1",
    "license_evidence_commit": S, "author_attestation": True,
    "rendered_digest": D, "historical_provenance_review": True,
    "independent_rights_review": True,
}


class CustodyTests(unittest.TestCase):
    def test_even_fully_attested_stays_quarantined(self):
        output = audit({"assets": [ITEM]})
        self.assertEqual(output["decision"], "NO_ADMISSION")
        self.assertEqual(output["admitted_count"], 0)
        self.assertEqual(output["quarantined_count"], 1)
        self.assertNotIn("fictional", str(output))
        self.assertNotIn("icon.svg", str(output))

    def test_empty_manifest_never_implies_admission(self):
        self.assertEqual(audit({"assets": []})["admitted_count"], 0)

    def test_duplicate_identity_rejected(self):
        with self.assertRaises(ValueError):
            audit({"assets": [ITEM, copy.deepcopy(ITEM)]})

    def test_unexpected_fields_rejected(self):
        bad = dict(ITEM, private_screenshot="SECRET")
        with self.assertRaises(ValueError):
            audit({"assets": [bad]})

    def test_missing_evidence_rejected(self):
        bad = dict(ITEM)
        bad.pop("author_attestation")
        with self.assertRaises(ValueError):
            audit({"assets": [bad]})

    def test_malformed_hash_rejected(self):
        bad = dict(ITEM, rendered_digest="not-a-digest")
        with self.assertRaises(ValueError):
            audit({"assets": [bad]})

    def test_non_boolean_provenance_rejected(self):
        bad = dict(ITEM, independent_rights_review="yes")
        with self.assertRaises(ValueError):
            audit({"assets": [bad]})

    def test_malformed_root_rejected(self):
        for bad in (None, [], {}, {"assets": "nope"}, {"assets": [], "permit": True}):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError):
                    audit(bad)


if __name__ == "__main__":
    unittest.main()
