"""H229 owner-local public H137 versus separately pinned npm bytes.

Fabricated ONLY. No owner original media, gold, old browser traces, models.
The temporary patch of published SHA pins is a UNIT FIXTURE ONLY and cannot
change H229 command line's fixed actual npm original SHA-256 reference.
"""
from __future__ import annotations

from hashlib import sha256
from unittest.mock import patch
import unittest

from test_h228_h137_four_original_assets import (
    OriginalFourAssetByteIntegrityTests as H228Fixture,
)
from h229.verified_published_npm_assets import (
    PINNED_PUBLISHED_ORIGINAL_SHA256,
)
from h229.verify_original_h137_against_published import (
    verify_original_h137_matches_real_npm_published_bytes,
)


class IndependentPublishedVersusOriginalOwnerBytesTests(unittest.TestCase):
    def setUp(self):
        self.synthetic=H228Fixture("test_four_exact_public_asset_files_and_css_receipt")
        self.synthetic.setUp()
        self.addCleanup(self.synthetic.doCleanups)
        self.published_fixture={
            name:sha256(data).hexdigest()
            for name, data in self.synthetic.synthetic_bytes.items()
        }

    def audit(self):
        return verify_original_h137_matches_real_npm_published_bytes(
            self.synthetic.runtime,
            self.synthetic.metadata,
        )

    def test_real_published_reference_sha_are_all_four_exact_assets(self):
        self.assertEqual(set(PINNED_PUBLISHED_ORIGINAL_SHA256),
                         set(self.published_fixture))
        self.assertEqual(len(PINNED_PUBLISHED_ORIGINAL_SHA256),4)
        for digest in PINNED_PUBLISHED_ORIGINAL_SHA256.values():
            self.assertEqual(len(digest),64)
            self.assertTrue(all(c in "0123456789abcdef" for c in digest))

    def test_fabricated_owner_assets_cannot_match_real_published_refs(self):
        with self.assertRaisesRegex(ValueError, "independently published npm bytes"):
            self.audit()

    def test_fixture_consistency_path_when_all_four_safe_fake_pins_agree(self):
        # This patch is test-only; proves the H228/H229 full comparison works.
        with patch.dict(
            PINNED_PUBLISHED_ORIGINAL_SHA256, self.published_fixture,
            clear=True,
        ):
            result=self.audit()
        self.assertEqual(result["original_files_checked"],4)
        self.assertEqual(result["published_npm_packages_pinned"],3)
        self.assertFalse(result["official_training_approved"])
        self.assertFalse(result["publisher_attestation_verified"])
        self.assertFalse(result["historical_sources_complete"])
        self.assertEqual(result["model_calls"],0)
        self.assertEqual(result["optimizer_steps"],0)
        self.assertEqual(result["consumed_ood_opened"],0)

    def test_fake_valid_h137_css_rejected_when_publication_reference_differs(self):
        altered=dict(self.published_fixture)
        altered["nouislider.min.css"]="0"*64
        with patch.dict(PINNED_PUBLISHED_ORIGINAL_SHA256,altered,clear=True):
            with self.assertRaisesRegex(ValueError,"independently published npm bytes"):
                self.audit()

    def test_fake_valid_h137_js_rejected_when_publication_reference_differs(self):
        for name in ("nouislider.min.js","sortable.min.js","split.min.js"):
            altered=dict(self.published_fixture)
            altered[name]="1"*64
            with self.subTest(name=name), patch.dict(
                PINNED_PUBLISHED_ORIGINAL_SHA256,altered,clear=True
            ):
                with self.assertRaisesRegex(
                    ValueError,"independently published npm bytes"
                ):
                    self.audit()

    def test_h228_owner_source_mutation_fails_even_if_published_pins_forged(self):
        path=self.synthetic.vendor/"nouislider.min.css"
        data=path.read_bytes()
        path.write_bytes(bytes([data[0]^1])+data[1:])
        with patch.dict(
            PINNED_PUBLISHED_ORIGINAL_SHA256,self.published_fixture,clear=True
        ):
            with self.assertRaisesRegex(ValueError,"owner provenance"):
                self.audit()

    def test_missing_h176_js_sha_cannot_be_overridden_by_publisher_match(self):
        self.synthetic.rows["sortablejs"]["artifact_sha256"]="a"*64
        with patch.dict(
            PINNED_PUBLISHED_ORIGINAL_SHA256,self.published_fixture,clear=True
        ):
            with self.assertRaisesRegex(ValueError,"H176 original npm"):
                self.audit()

    def test_incomplete_publisher_reference_is_fail_closed(self):
        incomplete=dict(self.published_fixture)
        incomplete.pop("nouislider.min.css")
        with patch.dict(PINNED_PUBLISHED_ORIGINAL_SHA256,incomplete,clear=True):
            with self.assertRaisesRegex(ValueError,"four assets"):
                self.audit()


if __name__=="__main__":
    unittest.main()
