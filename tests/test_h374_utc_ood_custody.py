"""H374: UTC-day correctness of independent OOD custodian signature metadata.

All claims and keys are invented ephemeral test objects. NO actual OOD,
source names, private cases, legal approval, TRAIN admission or model.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import unittest
from unittest.mock import patch

from h356 import sealed_ood_custody as custody
from h355.signed_rights_intake import TRUSTED_REVIEWERS
from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from tests.test_h356_sealed_ood_custody import (
    Ed25519PrivateKey, invented, sign,
)


class H374UtcBoundaryPureTests(unittest.TestCase):
    def test_utc_calendar_uses_timezone_aware_utc_not_local_date(self):
        for now, expected in (
            (datetime(2026, 11, 2, 23, 59, 59, tzinfo=timezone.utc),
             date(2026, 11, 2)),
            (datetime(2026, 11, 3, 0, 0, 1, tzinfo=timezone.utc),
             date(2026, 11, 3)),
        ):
            with self.subTest(now=now.isoformat()):
                with patch.object(custody, "datetime") as fake:
                    fake.now.return_value = now
                    self.assertEqual(custody.utc_calendar_today(), expected)
                    fake.now.assert_called_once_with(timezone.utc)

    def test_dates_must_be_canonical_and_expiration_window_bounded(self):
        for invalid in ("2026-11-03Z", "2026-11-3", "2026-02-30",
                        "2026/11/03", "2026-11-03+03:00"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    custody.dated(invalid)

    def test_production_external_trust_is_still_empty(self):
        self.assertEqual(len(custody.TRUSTED_CUSTODIANS), 0)
        self.assertEqual(len(TRUSTED_REVIEWERS), 0)
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())


@unittest.skipIf(Ed25519PrivateKey is None,
                 "Only dedicated cryptography-pinned CI can run signed fixtures")
class H374UtcBoundarySignedTests(unittest.TestCase):
    def test_valid_through_utc_expiration_day_never_grants_ood(self):
        claim, key, kid, registry = invented()
        claim["signature_date_utc"] = "2026-11-02"
        claim["expiry_date_utc"] = "2026-11-03"
        with patch.object(custody, "TRUSTED_CUSTODIANS", registry), (
             patch.object(custody, "utc_calendar_today",
                          return_value=date(2026, 11, 3))):
            receipt = custody.verify_sealed_ood_commitment(
                sign(claim, key, kid))
        self.assertTrue(receipt["external_custodian_signature_verified"])
        for permission in (
            "claim_substantively_independently_audited",
            "actual_family_nonoverlap_proven", "rights_proven",
            "blinded_ood_scoring_admitted", "new_training_authorized",
            "model_gain_proven", "can_release",
        ):
            self.assertFalse(receipt[permission])

    def test_signed_receipt_expired_at_next_utc_day_is_refused(self):
        claim, key, kid, registry = invented()
        claim["signature_date_utc"] = "2026-11-02"
        claim["expiry_date_utc"] = "2026-11-03"
        with patch.object(custody, "TRUSTED_CUSTODIANS", registry), (
             patch.object(custody, "utc_calendar_today",
                          return_value=date(2026, 11, 4))):
            with self.assertRaisesRegex(
                ValueError, "^H356_EXTERNAL_SEALED_OOD_CUSTODY_NOT_ATTESTED$"
            ):
                custody.verify_sealed_ood_commitment(sign(claim, key, kid))

    def test_signed_future_utc_day_is_refused_despite_local_clock(self):
        claim, key, kid, registry = invented()
        claim["signature_date_utc"] = "2026-11-04"
        claim["expiry_date_utc"] = "2026-11-05"
        with patch.object(custody, "TRUSTED_CUSTODIANS", registry), (
             patch.object(custody, "utc_calendar_today",
                          return_value=date(2026, 11, 3))):
            with self.assertRaisesRegex(
                ValueError, "^H356_EXTERNAL_SEALED_OOD_CUSTODY_NOT_ATTESTED$"
            ):
                custody.verify_sealed_ood_commitment(sign(claim, key, kid))


if __name__ == "__main__":
    unittest.main()
