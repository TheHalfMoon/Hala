"""H356 private benchmark custodian commitment intake, inert in production.

This verifies an externally trusted signature over public-safe *claims* only.
A signature is NOT evidence of actual source novelty, ownership, consent,
unseen tasks or custodian independence. No reviewer key is enrolled, so all
production submissions currently fail closed. Never accept hidden examples,
scores, model predictions, raw filenames, IDs or screenshots via this API.
"""
from __future__ import annotations

from base64 import b64decode
from datetime import date, timedelta
from hashlib import sha256
import json
import re
from types import MappingProxyType

from h355.signed_rights_intake import canonical, unique_object, nonfinite

SCHEMA = "hala.external-sealed-ood-custody.v1"
ENVELOPE = "hala.external-sealed-ood-signed-envelope.v1"
ROLE = "H356_SIGNED_METADATA_ONLY_NO_EVALUATION_ADMISSION"
MAX_RECEIPT_BYTES = 8192
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z0-9._:@ -]{8,120}\Z")
# Independent external custodian keys require separate human review.
# No key is trusted by this implementation. Never trust an envelope key.
TRUSTED_CUSTODIANS = MappingProxyType({})
PERMITTED = frozenset({"en", "ar"})
PUBLIC_EXPOSED_OR_CONSUMED = frozenset({
    "railkit", "signaldesk", "beaconboard", "tingle", "tiny-slider",
    "choices", "nouislider", "flatpickr", "awesomplete", "pell",
    "pikaday", "todomvc", "dnd-kit", "calculator", "sortablejs",
})


def require(condition: bool) -> None:
    if not condition:
        raise ValueError("H356_EXTERNAL_SEALED_OOD_CUSTODY_NOT_ATTESTED")


def dated(value: object) -> date:
    require(type(value) is str and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))
    try:
        day = date.fromisoformat(value)
    except ValueError:
        raise ValueError("H356_EXTERNAL_SEALED_OOD_CUSTODY_NOT_ATTESTED") from None
    require(day.isoformat() == value)
    return day


def read_envelope(raw: bytes) -> dict:
    require(type(raw) is bytes and 50 <= len(raw) <= MAX_RECEIPT_BYTES)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                           parse_constant=nonfinite)
    except (ValueError, UnicodeError, RecursionError, TypeError):
        raise ValueError("H356_EXTERNAL_SEALED_OOD_CUSTODY_NOT_ATTESTED") from None
    require(type(value) is dict and set(value) ==
            {"schema", "custodian_key_id", "payload", "signature_b64"}
            and value["schema"] == ENVELOPE
            and type(value["custodian_key_id"]) is str
            and HEX64.fullmatch(value["custodian_key_id"])
            and type(value["signature_b64"]) is str
            and len(value["signature_b64"]) == 88
            and type(value["payload"]) is dict)
    return value


def verify_sealed_ood_commitment(raw: bytes) -> dict:
    """Requires externally pre-registered custody identity, never test data.

    This interface cannot score, unseal or admit an OOD dataset. Candidate-
    supplied assertions are not independently audited and are never promoted.
    """
    envelope = read_envelope(raw)
    claim = envelope["payload"]
    require(set(claim) == {
        "schema", "custodian_identity", "signature_date_utc",
        "expiry_date_utc", "source_family_commitments_sha256",
        "sealed_task_set_sha256", "sealed_gold_sha256",
        "source_rights_evidence_sha256", "independence_evidence_sha256",
        "nonoverlap_audit_sha256", "locked_protocol_sha256",
        "language_case_counts", "task_count",
        "source_family_count", "restricted_metadata_claims",
    })
    require(claim["schema"] == SCHEMA and
            type(claim["custodian_identity"]) is str and
            NAME.fullmatch(claim["custodian_identity"]))
    today = date.today()
    signed = dated(claim["signature_date_utc"])
    expiry = dated(claim["expiry_date_utc"])
    require(signed <= today <= expiry and
            timedelta(days=1) <= expiry - signed <= timedelta(days=180))
    for name in ("sealed_task_set_sha256", "sealed_gold_sha256",
                 "source_rights_evidence_sha256",
                 "independence_evidence_sha256", "nonoverlap_audit_sha256",
                 "locked_protocol_sha256"):
        require(type(claim[name]) is str and HEX64.fullmatch(claim[name]))
    commitments = claim["source_family_commitments_sha256"]
    require(type(commitments) is list and 3 <= len(commitments) <= 12
            and all(type(key) is str and HEX64.fullmatch(key) for key in commitments)
            and len(set(commitments)) == len(commitments))
    count = claim["language_case_counts"]
    require(type(count) is dict and set(count) == PERMITTED
            and all(type(count[lang]) is int and 16 <= count[lang] <= 128
                    for lang in PERMITTED))
    require(type(claim["source_family_count"]) is int
            and claim["source_family_count"] == len(commitments))
    require(type(claim["task_count"]) is int
            and claim["task_count"] == sum(count.values())
            and 32 <= claim["task_count"] <= 256)
    assertions = claim["restricted_metadata_claims"]
    require(type(assertions) is dict and set(assertions) == {
        "no_unseen_content_disclosed",
        "custodian_not_train_or_prompt_author",
        "source_family_disjoint_from_all_known_dev_and_consumed_ood",
        "rendered_image_eval_rights_reviewed",
        "previously_unscored_and_unseen",
        "gold_held_by_custodian",
        "aggregate_only_release",
        "no_patient_or_real_account_data",
        "no_training_from_this_evaluation",
    } and all(assertions[key] is True for key in assertions))
    # Only an independently enrolled key is relevant; the envelope cannot
    # bring its own anchor or use a H355 legal reviewer's separate trust.
    kid = envelope["custodian_key_id"]
    require(kid in TRUSTED_CUSTODIANS)
    trusted = TRUSTED_CUSTODIANS[kid]
    require(type(trusted) is dict and set(trusted) ==
            {"identity", "public_key_bytes"}
            and trusted["identity"] == claim["custodian_identity"]
            and type(trusted["public_key_bytes"]) is bytes
            and len(trusted["public_key_bytes"]) == 32
            and sha256(trusted["public_key_bytes"]).hexdigest() == kid)
    try:
        signature = b64decode(envelope["signature_b64"], validate=True)
        require(len(signature) == 64)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        Ed25519PublicKey.from_public_bytes(
            trusted["public_key_bytes"]).verify(signature, canonical(claim))
    except (ValueError, TypeError, ImportError):
        raise ValueError("H356_EXTERNAL_SEALED_OOD_CUSTODY_NOT_ATTESTED") from None
    except Exception as exc:
        if exc.__class__.__name__ != "InvalidSignature":
            raise
        raise ValueError("H356_EXTERNAL_SEALED_OOD_CUSTODY_NOT_ATTESTED") from None
    # Names/digests are commitments, NOT evidence of actual disjointness.
    # No hidden family name, gold, prompt, target or source is returned.
    return {
        "role": ROLE,
        "external_custodian_signature_verified": True,
        "declared_family_count": len(commitments),
        "declared_en_cases": count["en"],
        "declared_ar_cases": count["ar"],
        "claim_substantively_independently_audited": False,
        "actual_family_nonoverlap_proven": False,
        "rights_proven": False,
        "blinded_ood_scoring_admitted": False,
        "new_training_authorized": False,
        "model_gain_proven": False,
        "can_release": False,
    }
