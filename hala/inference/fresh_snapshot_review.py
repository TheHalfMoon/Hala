"""H39 strict CPU-only proposal/snapshot consistency review; never executes.

The caller must independently obtain BOTH a trustworthy requested goal AND a
new trusted capture from the host. This helper cannot authenticate that origin,
prove atomic freshness, or authorize an action; success returns None, NOT a
permit token.
"""
from __future__ import annotations

from hala.contracts.proposal import Action, validate_action
from hala.contracts.validation import ContractError, contains, object_fields, require
from hala.inference import ServiceError
from hala.inference.explicit_label_guard import unique_exact_label_match
from hala.inference.vision_candidate import BINDINGS, bindings, parse_request

PROPOSAL_FIELDS = ("schema", *BINDINGS, "media_sha256", "actions",
                   "confidence", "status", "execution_authorized",
                   "verification_required", "provenance")
PROVENANCE_FIELDS = ("engine", "model_id", "model_revision",
                     "processor_revision", "precision", "prompt_revision",
                     "qualified", "latency_ms", "serving_config_revision")


def review_same_fresh_snapshot(original: dict, proposal: dict,
                               recaptured: dict, *,
                               trusted_goal: str) -> None:
    """Refuse if candidate differs from independently obtained fresh host state.

    NOTE: caller-provided objects aren't themselves proof of independence or
    an authenticated user. A host must enforce identity, permissions,
    atomicity/locks, postconditions and action execution separately.
    """
    require(type(original) is dict and type(recaptured) is dict,
            "Two validated capture objects required")
    require(type(trusted_goal) is str and 0 < len(trusted_goal) <= 1024,
            "A bounded separately trusted goal is required")
    object_fields(original, ("schema", "policy_request", "media"))
    object_fields(recaptured, ("schema", "policy_request", "media"))
    initial_policy = object_fields(original["policy_request"],
                                   ("schema", "request_id", "tenant_id", "task",
                                    "device", "observation", "allowed_actions",
                                    "method_ceiling"),
                                   ("history", "cancelled"))
    tenant = initial_policy["tenant_id"]
    require(type(tenant) is str, "Tenant must be validated before capture")
    try:
        earlier, _, _, digest = parse_request(original, tenant)
        current, _, _, current_digest = parse_request(recaptured, tenant)
    except ServiceError:
        raise ContractError('Captured request is no longer eligible') from None
    require(earlier.task.goal == trusted_goal and current.task.goal == trusted_goal,
            "Independent trusted intent is missing or changed")
    # Equality includes all immutable request records (window, locale, name,
    # bounds, budget, request/session/generation, redactions, truncation).
    require(earlier == current, "Host observation, policy or task changed")
    require(original["media"] == recaptured["media"] and digest == current_digest,
            "Fresh PNG bytes, metadata or observation hash changed")
    require(earlier.observation.truncation.elements_omitted == 0,
            "Partial accessibility tree cannot prove label uniqueness")
    require(type(proposal) is dict, "Untrusted proposal must be a bounded object")
    object_fields(proposal, PROPOSAL_FIELDS)
    require(proposal["schema"] == "hala.click_proposal.v2" and
            proposal["status"] == "continue" and
            proposal["execution_authorized"] is False and
            proposal["verification_required"] is True and
            proposal["confidence"] is None,
            "Proposal cannot claim authority, completion or confidence")
    require(all(proposal[field] == value for field, value in bindings(current).items()),
            "Proposal identity/generation differs from fresh host capture")
    require(proposal["media_sha256"] == current_digest,
            "Proposal screenshot is not the fresh host screenshot")
    provenance = object_fields(proposal["provenance"], PROVENANCE_FIELDS)
    require(provenance["qualified"] is False,
            "Unqualified inference must not claim qualification")
    actions = proposal["actions"]
    require(type(actions) is list and len(actions) == 1,
            "Exactly one click proposal required")
    action = Action.from_dict(actions[0])
    require(action.type == "click" and
            type(action.point) is tuple and len(action.point) == 2 and
            all(type(v) is int for v in action.point),
            "Click coordinates must be bounded original integer pixels")
    validate_action(action, current)
    # Re-check geometry against the WHOLE observed set, not just targets
    # eligible after redaction/protection.
    observed = [e for e in current.observation.accessibility
                if contains(e.bounds, action.point)]
    require(len(observed) == 1 and observed[0].element_ref == action.element_ref_hint,
            "Point binding is ambiguous or no longer matches the target")
    require(unique_exact_label_match(trusted_goal, observed[0].name,
                                     [e.name for e in current.observation.accessibility]),
            "Selected target has no unique literal match to trusted intent")
    # Deliberately return no authorization, token, action, capability or handle.
