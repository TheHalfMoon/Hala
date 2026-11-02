"""H355 invented keys only, adversarial signed rights handoff tests.

NEVER generate or commit a real trust root, signed approval or permission.
Tests are about verifier behavior. Production trusted key registry stays EMPTY.
"""
from __future__ import annotations

from base64 import b64encode
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    # The repository-wide stdlib-only suite has no cryptography dependency.
    # The dedicated H355 4-way CI matrix installs the pinned package and
    # MUST run every adversarial test without skips.
    serialization = None
    Ed25519PrivateKey = None

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h343.independent_rights_review_request import build_public_review_request
from h355 import signed_rights_intake as intake

ROOT = Path(__file__).resolve().parents[1]


def material(*, permission: bool = True):
    req = build_public_review_request(ROOT)
    issued = date.today()
    payload = {
        "schema": intake.SCHEMA,
        "review_request_sha256": req["request_sha256"],
        "reviewer_identity": "INVENTED-ONLY-EXTERNAL-TEST-REVIEWER",
        "reviewer_independence_evidence_sha256": "a" * 64,
        "evidence_custody_sha256": "b" * 64,
        "signed_date_utc": issued.isoformat(),
        "expiry_date_utc": (issued + timedelta(days=30)).isoformat(),
        "review_scope": {
            "purpose": "rendered_gui_screenshot_vision_model",
            "commercial_use": False,
            "output_weight_distribution": permission,
            "training_use": permission,
            "third_party_asset_exclusions": True,
        },
        "reviewed_families": [
            {
                "family": family["family"],
                "source_git_blob": family["pinned_git_blob"],
                "source_introduction_commit": family["introducing_commit"],
                "copyright_title_evidence_sha256": "c" * 64,
                "rendered_screenshot_rights_granted": permission,
                "derivative_weights_granted": permission,
                "reviewer_title_confirmed": permission,
                "license_restrictions_resolved": permission,
            } for family in req["families"]
        ],
    }
    ephemeral = Ed25519PrivateKey.generate()
    pub = ephemeral.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)
    id_ = sha256(pub).hexdigest()
    registry = {id_: {
        "identity": payload["reviewer_identity"],
        "public_key_bytes": pub,
    }}
    return payload, ephemeral, registry, id_


def envelope(payload, private, key_id):
    signature = private.sign(intake.canonical(payload))
    return json.dumps({
        "schema": intake.ENVELOPE_SCHEMA, "key_id": key_id,
        "payload": payload, "signature_b64": b64encode(signature).decode("ascii"),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")


@unittest.skipIf(Ed25519PrivateKey is None,
                 "H355 signed key tests require optional cryptography; dedicated CI installs it")
class H355RightsSignatureTests(unittest.TestCase):
    def test_trusted_registry_is_empty_and_train_is_still_denied(self):
        self.assertEqual(len(intake.TRUSTED_REVIEWERS), 0)
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())
        payload, key, registry, key_id = material()
        with self.assertRaisesRegex(ValueError, "NOT_AUTHENTICATED"):
            intake.verify_signed_review(envelope(payload, key, key_id), ROOT)

    def test_invented_registered_key_checks_signature_but_never_grants_training(self):
        payload, key, registry, key_id = material()
        with patch.object(intake, "TRUSTED_REVIEWERS", registry):
            got = intake.verify_signed_review(envelope(payload, key, key_id), ROOT)
        self.assertTrue(got["external_signature_verified"])
        self.assertTrue(got["specified_training_scope_reviewed"])
        self.assertEqual(got["source_family_count"], 2)
        self.assertFalse(got["official_h199_train_admitted"])
        self.assertFalse(got["can_train"])
        self.assertFalse(got["can_release"])
        from training.preflight import require_admitted_gui_training_manifest
        with self.assertRaisesRegex(ValueError, "NOT_ADMITTED"):
            require_admitted_gui_training_manifest(b'{"rights_cleared":true}')

    def test_modified_signed_content_wrong_key_and_signature_refused(self):
        payload, key, registry, key_id = material()
        good = envelope(payload, key, key_id)
        doc = json.loads(good)
        variants = []
        edited = json.loads(good)
        edited["payload"]["reviewed_families"][0][
            "rendered_screenshot_rights_granted"] = False
        variants.append(json.dumps(edited).encode())
        edited = json.loads(good)
        edited["key_id"] = "f"*64
        variants.append(json.dumps(edited).encode())
        edited = json.loads(good)
        edited["signature_b64"] = b64encode(b"x"*64).decode("ascii")
        variants.append(json.dumps(edited).encode())
        edited = json.loads(good)
        edited["payload"]["review_request_sha256"] = "0"*64
        variants.append(json.dumps(edited).encode())
        with patch.object(intake, "TRUSTED_REVIEWERS", registry):
            for candidate in variants:
                with self.subTest(candidate_sha=sha256(candidate).hexdigest()[:8]):
                    with self.assertRaises(ValueError):
                        intake.verify_signed_review(candidate, ROOT)

    def test_forged_source_and_expired_or_future_review_refused_even_when_signed(self):
        payload, key, registry, key_id = material()
        variants = []
        foreign = json.loads(json.dumps(payload))
        foreign["reviewed_families"][0]["source_git_blob"] = "0"*40
        variants.append(foreign)
        stale = json.loads(json.dumps(payload))
        stale["expiry_date_utc"] = (date.today()-timedelta(days=1)).isoformat()
        stale["signed_date_utc"] = (date.today()-timedelta(days=30)).isoformat()
        variants.append(stale)
        future = json.loads(json.dumps(payload))
        future["signed_date_utc"] = (date.today()+timedelta(days=2)).isoformat()
        future["expiry_date_utc"] = (date.today()+timedelta(days=40)).isoformat()
        variants.append(future)
        invalid = json.loads(json.dumps(payload))
        invalid["reviewed_families"].reverse()
        variants.append(invalid)
        with patch.object(intake, "TRUSTED_REVIEWERS", registry):
            for item in variants:
                with self.assertRaises(ValueError):
                    intake.verify_signed_review(envelope(item, key, key_id), ROOT)

    def test_nonapproval_review_never_becomes_approved(self):
        payload, key, registry, key_id = material(permission=False)
        with patch.object(intake, "TRUSTED_REVIEWERS", registry):
            result = intake.verify_signed_review(envelope(payload, key, key_id), ROOT)
        self.assertTrue(result["external_signature_verified"])
        self.assertFalse(result["specified_training_scope_reviewed"])
        self.assertFalse(result["can_train"])

    def test_wrong_identity_or_untrusted_public_key_never_trusted(self):
        payload, key, registry, key_id = material()
        registry[key_id]["identity"] = "OTHER-REVIEWER"
        with patch.object(intake, "TRUSTED_REVIEWERS", registry), self.assertRaises(ValueError):
            intake.verify_signed_review(envelope(payload, key, key_id), ROOT)
        payload, key, registry, key_id = material()
        registry[key_id]["public_key_bytes"] = bytes(32)
        with patch.object(intake, "TRUSTED_REVIEWERS", registry), self.assertRaises(ValueError):
            intake.verify_signed_review(envelope(payload, key, key_id), ROOT)

    def test_ambiguous_json_duplicate_keys_nonfinite_extra_fields_refused(self):
        for raw in (
            b'{"schema":1,"schema":2}',
            b'{"payload":{"x":NaN}}',
            b'{"payload":{"a":1,"a":2}}',
            b'{"schema":1,"x":0}',
            b"x" * (intake.MAX_BYTES + 1),
            b'{"schema":"oops"}',
        ):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                intake.verify_signed_review(raw, ROOT)

    def test_self_signed_envelope_cannot_enroll_own_public_key(self):
        payload, key, registry, key_id = material()
        extra = json.loads(envelope(payload, key, key_id))
        extra["reviewer_public_key"] = b64encode(
            registry[key_id]["public_key_bytes"]).decode()
        with self.assertRaises(ValueError):
            intake.verify_signed_review(json.dumps(extra).encode(), ROOT)


if __name__ == "__main__":
    unittest.main()
