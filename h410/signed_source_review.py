"""H410 independently enrolled Ed25519 signed visual-source approval gate.

PRODUCTION FAIL-CLOSED: no external reviewer public key is enrolled. Merely
having a signature does not establish perceptual correctness or independence.
No runtime generation, trust-on-first-use, public key from candidate, signing
secrets, training data, model weights or screenshots are persisted here.
"""
from __future__ import annotations

from base64 import b64decode
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
import json
import math
import re
import secrets
import time
from typing import Any, Callable

from h199.native_action_sft import WIDTH,HEIGHT
from h407.reviewed_action_bridge import (
    VisualTargetReview,run_source_reviewed_policy,action_digest,
)

SCHEMA="hala.h410.signed-visible-unique-target.v1"
PURPOSE="visually_verified_single_target"
DECLINE="H410_AUTHENTICATED_SOURCE_REVIEW_REQUIRED"
# Registration MUST require separate externally qualified human governance;
# no key is enrolled in the private Hala source.
TRUSTED_SOURCE_REVIEWERS=MappingProxyType({})
HEX64=re.compile(r"[0-9a-f]{64}\Z")
HEX32=re.compile(r"[0-9a-f]{32}\Z")
_FIELDS=frozenset({
    "schema","source_review_purpose","nonce","registered_url",
    "png_sha256","task_sha256","action_sha256",
    "reviewer_identity","matching_targets","visible_box",
    "issued_unix","expires_unix",
})

@dataclass(frozen=True)
class SignedSourceReview:
    key_id:str
    payload:dict
    signature_b64:str

def reject():
    raise ValueError(DECLINE)

def canonical_payload(value: object) -> bytes:
    try:
        encoded=json.dumps(value,sort_keys=True,separators=(",",":"),
                           ensure_ascii=True,allow_nan=False).encode("ascii")
    except (TypeError,ValueError,UnicodeError,OverflowError):
        reject()
    if not 0<len(encoded)<=4096:
        reject()
    return encoded

def _check_signed(
    message: object, *, image: bytes, task: str, action: dict,
    nonce: str,registered_url: str,
) -> VisualTargetReview:
    if type(message) is not SignedSourceReview:
        reject()
    if (type(message.key_id) is not str
        or HEX64.fullmatch(message.key_id) is None
        or message.key_id not in TRUSTED_SOURCE_REVIEWERS
        or type(message.payload) is not dict
        or set(message.payload)!=_FIELDS):
        reject()
    trusted=TRUSTED_SOURCE_REVIEWERS[message.key_id]
    if (type(trusted) is not dict or set(trusted)!={"identity","public_key_bytes"}
        or type(trusted["identity"]) is not str
        or type(trusted["public_key_bytes"]) is not bytes
        or len(trusted["public_key_bytes"])!=32
        or sha256(trusted["public_key_bytes"]).hexdigest()!=message.key_id):
        reject()
    p=message.payload
    expected={
        "schema":SCHEMA,
        "source_review_purpose":PURPOSE,
        "nonce":nonce,
        "registered_url":registered_url,
        "png_sha256":sha256(image).hexdigest(),
        "task_sha256":sha256(task.encode("utf8")).hexdigest(),
        "action_sha256":action_digest(action),
        "reviewer_identity":trusted["identity"],
        "matching_targets":1,
    }
    if any(type(p.get(key)) is not type(value) or p[key]!=value
           for key,value in expected.items()):
        reject()
    if type(nonce) is not str or HEX32.fullmatch(nonce) is None:
        reject()
    issued,expires=p["issued_unix"],p["expires_unix"]
    now=int(time.time())
    if (type(issued) is not int or type(expires) is not int
        or not issued<=now<=expires
        or not 0<expires-issued<=30):
        reject()
    box=p["visible_box"]
    if (type(box) is not list or len(box)!=4
        or any(type(v) not in (int,float) or not math.isfinite(v)
               for v in box)):
        reject()
    left,top,right,bottom=box
    if (not (0<=left<right<WIDTH and 0<=top<bottom<HEIGHT)
        or not (left<=action["point"][0]<=right
                and top<=action["point"][1]<=bottom)):
        reject()
    if (type(message.signature_b64) is not str
        or len(message.signature_b64)!=88):
        reject()
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        signature=b64decode(message.signature_b64,validate=True)
        if len(signature)!=64:reject()
        Ed25519PublicKey.from_public_bytes(
            trusted["public_key_bytes"]).verify(
                signature,canonical_payload(p))
    except Exception:
        # Includes InvalidSignature, decode errors and absent crypto library;
        # never leak raw signer private metadata to the model/browser.
        reject()
    return VisualTargetReview(
        png_sha256=p["png_sha256"],
        task_sha256=p["task_sha256"],
        action_sha256=p["action_sha256"],
        matching_targets=1,
        visible_box=tuple(box),
    )

def run_authenticated_reviewed_policy(
    page:Any,task:str,registered_url:str,
    predict:Callable[[bytes,str],str],
    sign_review:Callable[[bytes,str,dict,str],SignedSourceReview],
) -> dict:
    """Research-only signed reviewer gate, never default agent execution.

    The model does NOT receive the nonce or trusted private signing key.
    Every click needs a fresh random nonce; a preapproved signed message
    cannot be replayed into another transaction. H407 additionally enforces
    fresh same-pixel screenshot and registered source URL before dispatch.
    Source rights, reviewer independence, policy correctness and source
    authorization must be vetted OUTSIDE this cryptographic envelope.
    """
    if not TRUSTED_SOURCE_REVIEWERS:
        reject()
    if not callable(sign_review) or not callable(predict):
        reject()
    nonce=secrets.token_hex(16)

    def _review(png,user_task,action):
        try:
            signed=sign_review(png,user_task,action,nonce)
            return _check_signed(
                signed,image=png,task=user_task,action=action,
                nonce=nonce,registered_url=registered_url)
        except Exception:
            reject()
    try:
        result=run_source_reviewed_policy(
            page,task,registered_url,predict,_review)
    except ValueError as exc:
        # h407 suppresses all reviewer exceptions; retain H410's own
        # sanitized categorical failure without exposing untrusted details.
        if str(exc)=="H407 independent source review declined":
            reject()
        raise
    return {**result,
            "signed_source_review_verified":bool(
                result["kind"]=="click" and result["local_review_bound_to_png"]),
            "independent_ood_qualified":False,
            "real_gui_safety_certified":False,
            "official_model_rights_admitted":False,
            "authenticated_external_independence_proven":False,
            }
