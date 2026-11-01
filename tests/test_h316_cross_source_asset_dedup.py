"""H316 cross-path and cross-source metadata duplicate refusal; invented data."""
import copy
import unittest
from h315.asset_custody_audit import audit

A = {
    "source_repo": "fictional/app-a", "source_commit": "a"*40,
    "asset_path": "a/icon.svg", "asset_sha256": "sha256:"+"b"*64,
    "license_spdx": "OFL-1.1", "license_evidence_commit": "a"*40,
    "author_attestation": False, "rendered_digest": "sha256:"+"c"*64,
    "historical_provenance_review": False, "independent_rights_review": False,
}


class CrossSourceDedupTests(unittest.TestCase):
    def test_identical_bytes_under_different_source_refused(self):
        b = dict(A, source_repo="fictional/app-b", asset_path="b/new.svg",
                 rendered_digest="sha256:"+"d"*64)
        with self.assertRaisesRegex(ValueError, "duplicate asset content"):
            audit({"assets": [A, b]})

    def test_identical_rendered_frame_refused(self):
        b = dict(A, source_repo="fictional/app-b", asset_path="b/new.svg",
                 asset_sha256="sha256:"+"e"*64)
        with self.assertRaisesRegex(ValueError, "duplicate rendered content"):
            audit({"assets": [A, b]})

    def test_distinct_assets_still_never_admitted(self):
        b = dict(A, source_repo="fictional/app-b", asset_path="b/new.svg",
                 asset_sha256="sha256:"+"e"*64,
                 rendered_digest="sha256:"+"d"*64)
        result = audit({"assets": [A, b]})
        self.assertEqual(result["admitted_count"], 0)
        self.assertEqual(result["quarantined_count"], 2)
        self.assertNotIn("fictional", str(result))


if __name__ == "__main__":
    unittest.main()
