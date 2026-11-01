"""H343 public source-bound independent screenshot ML rights REVIEW REQUEST.

No legal approval, H199 TRAIN admission, private media or model evidence.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h329.railkit_source_provenance import (
    SOURCE as RAIL_PATH, SOURCE_BLOB as RAIL_BLOB,
    ORIGIN as RAIL_ORIGIN, audit_original_html,
)
from h339.original_source_rights_evidence import (
    SOURCE as SIGNAL_PATH, SOURCE_BLOB as SIGNAL_BLOB,
    ORIGIN as SIGNAL_ORIGIN, audit_signaldesk_html,
)

ROLE = "H343_INDEPENDENT_RIGHTS_REVIEW_REQUIRED_NOT_TRAIN_ADMITTED"
SCHEMA = "hala.original-screenshot-ml-rights-request.v1"
FAMILIES = (
    ("RailKit", RAIL_PATH, RAIL_BLOB, RAIL_ORIGIN, 16, 32),
    ("SignalDesk", SIGNAL_PATH, SIGNAL_BLOB, SIGNAL_ORIGIN, 128, 256),
)
QUESTIONS = (
    "independent_copyright_title_and_authorship",
    "complete_derivative_and_contributor_history",
    "permission_to_render_and_reproduce_source",
    "permission_to_train_on_rendered_screenshots",
    "permission_for_resulting_adapter_and_weight_distribution",
    "noncommercial_vs_commercial_scope_and_restrictions",
    "third_party_asset_and_privacy_exclusions",
    "reviewer_independence_identity_and_evidence_custody",
)


def need(condition: bool) -> None:
    if not condition:
        raise ValueError("H343_EXTERNAL_RIGHTS_DECISION_REQUIRED")


def _canonical(doc: dict) -> bytes:
    return json.dumps(doc, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def build_public_review_request(root: Path) -> dict:
    """Inspect only exact original public HTML; do not read/render images."""
    need(isinstance(root, Path) and not ADMITTED_TRAIN_MANIFEST_SHAS)
    try:
        need(not root.is_symlink())
        root = root.resolve(strict=True)
        need(root.is_dir())
        families = []
        for name, relative, blob, origin, scenes, actions in FAMILIES:
            path = root / relative
            need(path.is_file() and not path.is_symlink()
                 and not path.parent.is_symlink()
                 and path.resolve(strict=True).is_relative_to(root))
            raw = path.read_bytes()
            if name == "RailKit":
                audit_original_html(raw)
            else:
                audit_signaldesk_html(raw)
            families.append({
                "family": name, "source_path": relative,
                "pinned_git_blob": blob, "introducing_commit": origin,
                "exposed_candidate_scenes": scenes,
                "exposed_candidate_action_stages": actions,
                "independent_title_approved": False,
                "screenshot_ml_training_rights_approved": False,
            })
    except (OSError, ValueError, RuntimeError):
        raise ValueError("H343_EXTERNAL_RIGHTS_DECISION_REQUIRED") from None
    payload = {
        "schema": SCHEMA, "role": ROLE, "families": families,
        "questions_for_independent_reviewer": list(QUESTIONS),
        "requested_use": "rendered_original_GUI_screenshots_for_Qwen_VLM_training",
        "review_status": "NO_INDEPENDENT_LEGAL_DECISION",
        "official_h199_train_admitted": False,
        "blind_ood_qualified": False, "adapted_gain_proven": False,
        "optimizer_steps": 0, "can_train": False, "can_release": False,
    }
    return {**payload, "request_sha256": sha256(_canonical(payload)).hexdigest()}


def verify_request_integrity_only(packet: object, root: Path) -> dict:
    """A valid REQUEST is never a rights grant, regardless of claimed flags."""
    need(type(packet) is dict and packet == build_public_review_request(root))
    return {
        "request_integrity": True,
        "independent_rights_approval": False,
        "official_train_admission": False,
        "can_train": False, "can_release": False,
    }
