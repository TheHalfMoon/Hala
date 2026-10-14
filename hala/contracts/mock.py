"""Deterministic fixture behavior, not a trained model or an executor."""
from hashlib import sha256
import re

from .proposal import Action, Metadata, Postcondition, Proposal, validate_proposal, validate_target
from .request import Request
from .responses import Decision, Plan, PlanStep, validate_decision, validate_plan
from .validation import ContractError

MOCK_REVISION = sha256(b"hala-offline-fixture-mock-v1").hexdigest()


def metadata():
    # Zero is an explicit synthetic timing value, not a measured inference claim.
    return Metadata("hala-offline-fixture-mock", MOCK_REVISION, "fixture-prompts-v1", "offline-mock-v1", 0.0, "deterministic_mock", False)


def request_record(value):
    return value if type(value) is Request else Request.from_dict(value)


def binding(request):
    return {"request_id": request.request_id, "tenant_id": request.tenant_id, "task_id": request.task.task_id, "device_id": request.device.device_id, "session_id": request.device.session_id, "observation_generation": request.observation.generation}


def target_named(request, name):
    # Only an exact, unique label match is usable. Observation text is never instructions.
    matches = [item for item in request.observation.accessibility if item.name.casefold() == name.casefold()]
    if len(matches) != 1:
        return None
    try:
        return validate_target(matches[0].element_ref, request)
    except ContractError:
        return None


def fixture_action(request):
    goal = request.task.goal.strip()
    click = re.fullmatch(r"(?:click|انقر) (.+)", goal, flags=re.IGNORECASE)
    if click:
        target = target_named(request, click.group(1))
        if target is None:
            return None
        if "semantic_invoke" in request.allowed_actions:
            return Action("semantic_invoke", element_ref=target.element_ref)
        if "click" in request.allowed_actions and request.method_ceiling == "WINDOW_TARGETED_INPUT":
            x, y, width, height = target.bounds
            return Action("click", point=(x + width / 2, y + height / 2), coordinate_space=request.observation.image.coordinate_space, image_scale=request.observation.image.scale, element_ref_hint=target.element_ref)
        return None
    entry = re.fullmatch(r"set (.+?) to (.*)", goal, flags=re.IGNORECASE)
    if entry:
        target = target_named(request, entry.group(1))
        if target is None or target.role.casefold() not in ("edit", "textbox", "textinput"):
            return None
        if "set_value" in request.allowed_actions:
            return Action("set_value", element_ref=target.element_ref, text=entry.group(2), secret=False)
        if "type" in request.allowed_actions and request.method_ceiling == "WINDOW_TARGETED_INPUT":
            return Action("type", element_ref=target.element_ref, text=entry.group(2), secret=False)
        return None
    if goal.casefold() == "observe" and "observe" in request.allowed_actions:
        return Action("observe")
    if goal.casefold() == "check completion" and "done_proposed" in request.allowed_actions:
        return Action("done_proposed")
    return None


def mock_action(value):
    request = request_record(value)
    common = {"schema": "hala.proposal.v1", **binding(request), "confidence": 0.0, "metadata": metadata()}
    if request.cancelled:
        result = Proposal(**common, actions=(), status="cancelled", rationale_short="User cancellation prevents proposals.")
    elif request.budget_exhausted:
        result = Proposal(**common, actions=(), status="budget_exhausted", rationale_short="Task step budget is exhausted.")
    else:
        action = fixture_action(request)
        if action is None:
            reason = "Offline mock could not uniquely ground this fixture intent; no action is recommended."
            actions = (Action("abstain", reason=reason),) if "abstain" in request.allowed_actions else ()
            result = Proposal(**common, actions=actions, status="abstain", rationale_short=reason)
        elif action.type == "done_proposed":
            result = Proposal(**common, actions=(action,), status="verification_recommended", rationale_short="Recommend independent local final-state verification; this is not a success receipt.", expected_postcondition=Postcondition("final_state_check"))
        else:
            result = Proposal(**common, actions=(action,), status="continue", rationale_short="Deterministic synthetic fixture proposal; Aladdin local policy remains authoritative.")
    return validate_proposal(result, request)


def mock_plan(value):
    request = request_record(value)
    proposal = mock_action(request)
    if proposal.status in ("cancelled", "budget_exhausted", "abstain"):
        steps = ()
    else:
        steps = tuple(PlanStep(request.task.step_index + index, "Request local evaluation of this bounded proposal.", action.type, action.element_ref or action.element_ref_hint) for index, action in enumerate(proposal.actions))
    return validate_plan(Plan("hala.plan.v1", **binding(request), steps=steps, status=proposal.status, confidence=0.0, rationale_short=proposal.rationale_short, metadata=metadata()), request)


def mock_decision(value):
    request = request_record(value)
    if request.cancelled:
        route, reason = "abstain", "User cancellation requires abstention."
    elif request.budget_exhausted:
        route, reason = "budget_exhausted", "Task step budget is exhausted."
    elif request.observation.window.protected:
        route, reason = "conservative_review", "Protected observation requires independent local review."
    else:
        action = fixture_action(request)
        if action is None:
            route, reason = "mind", "Unrecognized fixture intent requires qualified reasoning or user clarification."
        elif action.type in ("semantic_invoke", "set_value"):
            route, reason = "deterministic_tool", "A bounded structured proposal is available; this route grants no approval."
        elif action.type == "observe":
            route, reason = "observe", "A fresh local observation is recommended."
        elif action.type == "done_proposed":
            route, reason = "conservative_review", "Independent local final-state verification is recommended."
        else:
            route, reason = "type", "Qualified visual grounding may be required; local policy still decides."
    return validate_decision(Decision("hala.decision.v1", **binding(request), route=route, confidence=0.0, abstained=route in ("abstain", "budget_exhausted"), reason=reason, advisory_only=True, metadata=metadata()), request)
