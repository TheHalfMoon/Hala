"""H318 invented-only cross-platform name collision tests; no real media."""
import unittest

from h315.asset_custody_audit import audit

BASE = {
    "source_repo": "Example/Project", "source_commit": "a" * 40,
    "asset_path": "assets/Icon.png", "asset_sha256": "sha256:" + "b" * 64,
    "license_spdx": "OFL-1.1", "license_evidence_commit": "c" * 40,
    "author_attestation": False,
    "rendered_digest": "sha256:" + "d" * 64,
    "historical_provenance_review": False,
    "independent_rights_review": False,
}


def second(**kwargs):
    return dict(BASE, asset_sha256="sha256:" + "e" * 64,
                rendered_digest="sha256:" + "f" * 64, **kwargs)


class CrossPlatformCustodyTests(unittest.TestCase):
    def test_windows_case_alias_with_different_digests_refused(self):
        b = second(asset_path="ASSETS/icon.PNG")
        with self.assertRaisesRegex(ValueError, "duplicate asset identity"):
            audit({"assets": [BASE, b]})

    def test_repository_case_alias_with_different_digests_refused(self):
        b = second(source_repo="example/project")
        with self.assertRaisesRegex(ValueError, "duplicate asset identity"):
            audit({"assets": [BASE, b]})

    def test_unicode_nfkc_alias_with_different_digests_refused(self):
        # Mathematical bold capital I is compatibility-equivalent to I.
        b = second(asset_path="assets/𝐈con.png")
        with self.assertRaisesRegex(ValueError, "duplicate asset identity"):
            audit({"assets": [BASE, b]})

    def test_non_nfc_alias_refused_independently(self):
        b = dict(BASE, asset_path="assets/cafe\u0301.png")
        with self.assertRaisesRegex(ValueError, "invalid relative asset"):
            audit({"assets": [b]})

    def test_windows_device_names_refused_in_any_segment(self):
        for name in ("CON", "NUL.txt", "prn.svg", "COM1", "LPT9.png",
                     "COM¹.txt", "assets/AUX.jpeg", "assets/nul.txt"):
            with self.subTest(name=name):
                b = dict(BASE, asset_path=name)
                with self.assertRaisesRegex(ValueError, "invalid relative asset"):
                    audit({"assets": [b]})

    def test_benign_similar_names_still_quarantined(self):
        for name in ("assets/console.png", "assets/COM10.png",
                     "assets/دفتر.png", "assets/مرحبا.svg"):
            with self.subTest(name=name):
                result = audit({"assets": [dict(BASE, asset_path=name)]})
                self.assertEqual(result["decision"], "NO_ADMISSION")
                self.assertEqual(result["admitted_count"], 0)

    def test_distinct_filenames_remain_distinct_but_quarantined(self):
        b = second(asset_path="assets/new-icon.png")
        result = audit({"assets": [BASE, b]})
        self.assertEqual(result["quarantined_count"], 2)
        self.assertEqual(result["admitted_count"], 0)


if __name__ == "__main__":
    unittest.main()
