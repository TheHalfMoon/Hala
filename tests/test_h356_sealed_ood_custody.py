"""H356 invented-only sealed OOD custody tests.

No source/gold/benchmark tasks/real external keys are created or published.
"""
from __future__ import annotations

import copy
from base64 import b64encode
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import unittest
from unittest.mock import patch

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    serialization = None
    Ed25519PrivateKey = None

from h356 import sealed_ood_custody as custody
from h355.signed_rights_intake import TRUSTED_REVIEWERS
from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS


def invented():
    now = datetime.now(timezone.utc).date()
    claim = {
        "schema": custody.SCHEMA,
        "custodian_identity": "INVENTED-ONLY-INDEPENDENT-CUSTODIAN",
        "signature_date_utc": now.isoformat(),
        "expiry_date_utc": (now + timedelta(days=30)).isoformat(),
        "source_family_commitments_sha256": [chr(n)*64 for n in (97, 98, 99, 100)],
        "sealed_task_set_sha256": "e"*64,
        "sealed_gold_sha256": "f"*64,
        "source_rights_evidence_sha256": "0"*64,
        "independence_evidence_sha256": "1"*64,
        "nonoverlap_audit_sha256": "2"*64,
        "locked_protocol_sha256": "3"*64,
        "language_case_counts": {"en": 20, "ar": 20},
        "task_count": 40,
        "source_family_count": 4,
        "restricted_metadata_claims": {
            "no_unseen_content_disclosed": True,
            "custodian_not_train_or_prompt_author": True,
            "source_family_disjoint_from_all_known_dev_and_consumed_ood": True,
            "rendered_image_eval_rights_reviewed": True,
            "previously_unscored_and_unseen": True,
            "gold_held_by_custodian": True,
            "aggregate_only_release": True,
            "no_patient_or_real_account_data": True,
            "no_training_from_this_evaluation": True,
        },
    }
    private = Ed25519PrivateKey.generate()
    pub = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    key = sha256(pub).hexdigest()
    registry = {key: {"identity": claim["custodian_identity"],
                      "public_key_bytes": pub}}
    return claim, private, key, registry


def sign(claim, private, key):
    signature = private.sign(custody.canonical(claim))
    return json.dumps({
        "schema": custody.ENVELOPE, "custodian_key_id": key,
        "payload": claim, "signature_b64": b64encode(signature).decode("ascii"),
    }, separators=(",", ":"), sort_keys=True).encode("utf-8")


@unittest.skipIf(Ed25519PrivateKey is None, "Requires optional pinned cryptography in H356 CI")
class H356SealedCustodyTests(unittest.TestCase):
    def test_production_no_trusted_custodian_no_test_or_train_admission(self):
        claim, private, kid, registry = invented()
        self.assertEqual(len(custody.TRUSTED_CUSTODIANS), 0)
        self.assertEqual(len(TRUSTED_REVIEWERS), 0)
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())
        with self.assertRaises(ValueError):
            custody.verify_sealed_ood_commitment(sign(claim, private, kid))

    def test_patched_invented_key_authenticates_only_metadata_not_ood(self):
        claim, private, kid, registry = invented()
        with patch.object(custody, "TRUSTED_CUSTODIANS", registry):
            result = custody.verify_sealed_ood_commitment(sign(claim, private, kid))
        self.assertTrue(result["external_custodian_signature_verified"])
        self.assertEqual((result["declared_en_cases"], result["declared_ar_cases"]), (20, 20))
        self.assertEqual(result["declared_family_count"], 4)
        for name in ("actual_family_nonoverlap_proven", "rights_proven",
                     "claim_substantively_independently_audited",
                     "blinded_ood_scoring_admitted", "new_training_authorized",
                     "model_gain_proven", "can_release"):
            self.assertFalse(result[name])
        self.assertNotIn("sealed_gold_sha256", result)

    def test_rejects_forged_signature_and_mutated_claim(self):
        claim, private, kid, registry = invented()
        valid = json.loads(sign(claim, private, kid))
        variants = []
        changed = copy.deepcopy(valid)
        changed["payload"]["task_count"] = 41
        variants.append(changed)
        changed = copy.deepcopy(valid)
        changed["signature_b64"] = b64encode(bytes(64)).decode()
        variants.append(changed)
        changed = copy.deepcopy(valid)
        changed["custodian_key_id"] = "a"*64
        variants.append(changed)
        changed = copy.deepcopy(valid)
        changed["custodian_public_key"] = "user-supplied"
        variants.append(changed)
        with patch.object(custody, "TRUSTED_CUSTODIANS", registry):
            for candidate in variants:
                with self.subTest(case=sha256(json.dumps(candidate).encode()).hexdigest()[:8]):
                    with self.assertRaises(ValueError):
                        custody.verify_sealed_ood_commitment(json.dumps(candidate).encode())

    def test_rejects_raw_hidden_case_and_unapproved_metadata(self):
        claim, private, kid, registry = invented()
        for key, value in (
            ("source_family_names", ["PrivateApp"]),
            ("private_gold_points", [[120, 64]]),
            ("raw_screenshot_bytes", "data"),
            ("score", {"ar": 1, "en": 2}),
            ("source_examples", [{"prompt": "secret"}]),
        ):
            modified = copy.deepcopy(claim)
            modified[key] = value
            with self.subTest(key=key), patch.object(
                custody, "TRUSTED_CUSTODIANS", registry), self.assertRaises(ValueError):
                custody.verify_sealed_ood_commitment(sign(modified, private, kid))

    def test_expired_and_future_and_unbalanced_counts_rejected(self):
        claim, private, kid, registry = invented()
        changes = []
        altered = copy.deepcopy(claim)
        altered["expiry_date_utc"] = (datetime.now(timezone.utc).date()-timedelta(days=1)).isoformat()
        altered["signature_date_utc"] = (datetime.now(timezone.utc).date()-timedelta(days=32)).isoformat()
        changes.append(altered)
        altered = copy.deepcopy(claim)
        altered["signature_date_utc"] = (datetime.now(timezone.utc).date()+timedelta(days=2)).isoformat()
        changes.append(altered)
        altered = copy.deepcopy(claim)
        altered["language_case_counts"] = {"en": 31, "ar": 9}
        changes.append(altered)
        altered = copy.deepcopy(claim)
        altered["language_case_counts"] = {"en": 20, "ar": True}
        changes.append(altered)
        for candidate in changes:
            with patch.object(custody, "TRUSTED_CUSTODIANS", registry), self.assertRaises(ValueError):
                custody.verify_sealed_ood_commitment(sign(candidate, private, kid))

    def test_duplicate_family_commitment_and_self_asserted_scope(self):
        claim, private, kid, registry = invented()
        candidates = []
        altered = copy.deepcopy(claim)
        altered["source_family_commitments_sha256"][1] = altered["source_family_commitments_sha256"][0]
        candidates.append(altered)
        altered = copy.deepcopy(claim)
        altered["restricted_metadata_claims"]["gold_held_by_custodian"] = False
        candidates.append(altered)
        altered = copy.deepcopy(claim)
        altered["source_family_count"] = 5
        candidates.append(altered)
        altered = copy.deepcopy(claim)
        altered["source_family_commitments_sha256"][0] = "not-a-hash"
        candidates.append(altered)
        for candidate in candidates:
            with patch.object(custody, "TRUSTED_CUSTODIANS", registry), self.assertRaises(ValueError):
                custody.verify_sealed_ood_commitment(sign(candidate, private, kid))

    def test_untrusted_identity_and_bad_trust_key_cannot_be_enrolled(self):
        claim, private, kid, registry = invented()
        wrong = copy.deepcopy(registry)
        wrong[kid]["identity"] = "DIFFERENT INDEPENDENT CUSTODIAN"
        with patch.object(custody, "TRUSTED_CUSTODIANS", wrong), self.assertRaises(ValueError):
            custody.verify_sealed_ood_commitment(sign(claim, private, kid))
        wrong = copy.deepcopy(registry)
        wrong[kid]["public_key_bytes"] = bytes(32)
        with patch.object(custody, "TRUSTED_CUSTODIANS", wrong), self.assertRaises(ValueError):
            custody.verify_sealed_ood_commitment(sign(claim, private, kid))

    def test_duplicate_json_nonfinite_oversized_and_extra_fields_rejected(self):
        claim, private, kid, registry = invented()
        for raw in (b'{"schema":1,"schema":2}', b'{"a":NaN}',
                    b"x"*(custody.MAX_RECEIPT_BYTES+1),
                    b'{"payload":{"raw_gold":7}}', bytes([255])*100):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                custody.verify_sealed_ood_commitment(raw)


if __name__ == "__main__":
    unittest.main()
