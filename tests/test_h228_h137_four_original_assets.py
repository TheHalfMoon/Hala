"""H228 synthetic-only tests for exact original H137 JS+CSS asset provenance.

All files are freshly fabricated small test fixtures. Never open owner
images/gold, consumed H173 OOD, real binaries or model weights in CI.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from h176.source_family_attribution import (
    KNOWN_NPM_SOURCES, SCHEMA as H176_SCHEMA,
)
from h228.original_npm_asset_integrity import (
    H137_VENDOR_SOURCES, JS_TO_H176, MAX_VENDOR_BYTES,
    _read_bounded_json, _source_contract, verify_h137_original_vendor_assets,
)


class OriginalFourAssetByteIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.runtime = self.root / "h137-original-synthetic-runtime"
        self.vendor = self.runtime / "vendor"
        self.vendor.mkdir(parents=True)
        self.provenance = {}
        self.synthetic_bytes = {}
        for i, (name, (pkg, ver, _rel)) in enumerate(
            sorted(H137_VENDOR_SOURCES.items())
        ):
            data = ("H228 fake source asset " + str(i) + " " + name).encode() * 3
            self.synthetic_bytes[name] = data
            (self.vendor / name).write_bytes(data)
            self.provenance[name] = {
                "package": pkg, "version": ver,
                "sha256": sha256(data).hexdigest(), "license": "MIT",
            }
        self.rows = {
            app: {
                "app": app, "package": package, "version": version,
                "artifact_sha256": self.provenance[name]["sha256"],
            }
            for name, app in JS_TO_H176.items()
            for package, version in (KNOWN_NPM_SOURCES[app],)
        }
        self.metadata = {
            "schema": H176_SCHEMA,
            "families": list(self.rows.values()),
        }
        self.flush_provenance()

    def flush_provenance(self):
        (self.runtime / "provenance.json").write_text(
            json.dumps(self.provenance, sort_keys=True),
            encoding="utf-8",
        )

    def pass_check(self):
        return verify_h137_original_vendor_assets(self.runtime, self.metadata)

    def deny(self, message=None):
        with self.assertRaises(ValueError) as seen:
            self.pass_check()
        if message:
            self.assertIn(message, str(seen.exception))

    def test_four_exact_public_asset_files_and_css_receipt(self):
        report = self.pass_check()
        self.assertEqual(report["public_asset_files_verified"], 4)
        self.assertEqual(report["js_against_h176_historical_apps"], 3)
        self.assertEqual(report["original_css_assets_verified"], 1)
        self.assertFalse(report["publisher_artifact_authenticity_proven"])
        self.assertFalse(report["historical_source_completeness_proven"])
        self.assertFalse(report["official_train_admitted"])
        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["consumed_ood_opened"], 0)

    def test_h137_public_source_code_declares_four_js_css_files(self):
        _source_contract()
        self.assertEqual(set(H137_VENDOR_SOURCES), {
            "nouislider.min.js", "nouislider.min.css",
            "sortable.min.js", "split.min.js",
        })
        self.assertEqual(set(JS_TO_H176.values()), set(KNOWN_NPM_SOURCES))

    def test_missing_css_fails_even_when_all_three_h176_javascript_shas_match(self):
        (self.vendor / "nouislider.min.css").unlink()
        self.deny("incomplete")

    def test_css_bytes_modified_same_length_fails(self):
        p = self.vendor / "nouislider.min.css"
        data = p.read_bytes()
        p.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        self.deny("do not match owner provenance")

    def test_original_javascript_bytes_modified_same_length_fails(self):
        p = self.vendor / "split.min.js"
        data = p.read_bytes()
        p.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        self.deny("do not match owner provenance")

    def test_css_missing_from_provenance_does_not_pass(self):
        self.provenance.pop("nouislider.min.css")
        self.flush_provenance()
        self.deny("all four")

    def test_h137_provenance_extra_unknown_file_rejected(self):
        self.provenance["extra.js"] = self.provenance["split.min.js"]
        self.flush_provenance()
        self.deny("all four")

    def test_extra_vendor_binary_rejected_without_reading(self):
        (self.vendor / "unknown-other-original.js").write_bytes(b"FAKE")
        self.deny("extra entries")

    def test_h176_original_app_sha_mismatch_refused(self):
        self.rows["sortablejs"]["artifact_sha256"] = "0"*64
        self.deny("H176 original npm app artifact hash disagrees")

    def test_bad_h137_package_version_or_license_rejected(self):
        for key, bad in (
            ("package", "another-package"), ("version", "17.0.0"),
            ("license", "commercial"), ("sha256", "a"*40),
        ):
            with self.subTest(key=key):
                previous = self.provenance["nouislider.min.css"][key]
                self.provenance["nouislider.min.css"][key] = bad
                self.flush_provenance()
                self.deny()
                self.provenance["nouislider.min.css"][key] = previous
                self.flush_provenance()

    def test_invalid_h176_attribution_never_approves(self):
        originals = self.metadata
        for bad in (
            {"schema": "wrong", "families": list(self.rows.values())},
            {"schema": H176_SCHEMA, "families": []},
            {"schema": H176_SCHEMA, "families": [self.rows["splitjs"]]},
            {"schema": H176_SCHEMA, "families": list(self.rows.values()),
             "private_gold": "never"},
        ):
            self.metadata = bad
            self.deny()
        self.metadata = originals

    def test_h176_npm_record_cannot_claim_fabricated_git(self):
        self.rows["splitjs"]["git_sha"] = "a"*40
        self.deny("fake Git/source")
        self.rows["splitjs"].pop("git_sha")

    def test_duplicate_provenance_json_key_blocked(self):
        p = self.runtime / "provenance.json"
        original = p.read_text("utf-8")
        p.write_text('{"nouislider.min.js":{},"nouislider.min.js":{}}',
                     encoding="utf-8")
        with self.assertRaises(ValueError):
            _read_bounded_json(p)
        p.write_text(original, encoding="utf-8")

    def test_nonfinite_json_constant_refused(self):
        p = self.runtime / "provenance.json"
        p.write_text('{"a": NaN}', encoding="utf-8")
        with self.assertRaises(ValueError):
            _read_bounded_json(p)

    def test_missing_provenance_file_denied(self):
        (self.runtime / "provenance.json").unlink()
        self.deny("public metadata")

    def test_zero_and_oversized_vendor_file_denied(self):
        p = self.vendor / "split.min.js"
        p.write_bytes(b"")
        self.deny("vendor size")
        p.write_bytes(b"x"*(MAX_VENDOR_BYTES+1))
        self.deny("vendor size")

    def test_file_symlink_rejected_even_if_bytes_identical(self):
        p = self.vendor / "split.min.js"
        data = p.read_bytes()
        p.unlink()
        target = self.root / "outside-synthetic-public-data.js"
        target.write_bytes(data)
        try:
            p.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("Windows user does not permit symbolic links")
        self.deny("linked")

    def test_vendor_directory_symlink_is_refused(self):
        other = self.root / "alternate-synthetic-vendor"
        try:
            other.symlink_to(self.vendor, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Windows user does not permit directory symbolic links")
        original = self.vendor
        self.vendor = other
        self.runtime = self.root / "symlink-synthetic-runtime"
        self.runtime.mkdir()
        (self.runtime / "vendor").symlink_to(original, target_is_directory=True)
        self.deny("linked")

    def test_relative_runtime_and_unexpected_owner_folder_denied(self):
        with self.assertRaises(ValueError):
            verify_h137_original_vendor_assets(Path("relative"), self.metadata)
        with self.assertRaises(ValueError):
            verify_h137_original_vendor_assets(self.root / "not-owned", self.metadata)


if __name__ == "__main__":
    unittest.main()
