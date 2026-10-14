"""Immutable planning and routing advice with contextual input bindings."""
from dataclasses import dataclass

from .proposal import INPUT_ACTIONS, Metadata, STATUSES, validate_binding, validate_target
from .request import ACTION_TYPES, MAX_ACTIONS
from .validation import Record, boolean, frozen_sequence, number, object_fields, require, sequence, string

ROUTES = ("deterministic_tool", "mind", "type", "observe", "ask_user", "abstain", "escalate", "budget_exhausted", "conservative_review")
BINDING_FIELDS = ("request_id", "tenant_id", "task_id", "device_id", "session_id", "observation_generation")


@dataclass(frozen=True, slots=True)
class PlanStep(Record):
    index: int
    intent: str
    action_type: str
    element_ref: str | None = None

    def __post_init__(self):
        number(self.index, 0, 59, integer=True)
        string(self.intent, 512)
        require(type(self.action_type) is str and self.action_type in ACTION_TYPES, "Unsupported planned action")
        if self.element_ref is not None:
            string(self.element_ref)

    @classmethod
    def from_dict(cls, value):
        return cls(**object_fields(value, ("index", "intent", "action_type"), ("element_ref",)))


@dataclass(frozen=True, slots=True)
class Plan(Record):
    schema: str
    request_id: str
    tenant_id: str
    task_id: str
    device_id: str
    session_id: str
    observation_generation: str
    steps: tuple
    status: str
    confidence: float
    rationale_short: str
    metadata: Metadata

    def __post_init__(self):
        require(self.schema == "hala.plan.v1", "Unsupported plan schema")
        for field in BINDING_FIELDS:
            string(getattr(self, field))
        require(self.status in STATUSES, "Unsupported plan status")
        number(self.confidence, 0, 1)
        string(self.rationale_short, 200)
        require(type(self.metadata) is Metadata, "Invalid planning metadata")
        require(all(type(step) is PlanStep for step in frozen_sequence(self.steps, MAX_ACTIONS)), "Invalid plan steps")
        if self.status in ("cancelled", "budget_exhausted", "abstain"):
            require(not self.steps, "Inactive or abstaining plan cannot include steps")
        else:
            require(bool(self.steps), "Active plan needs steps")
        if self.status == "verification_recommended":
            require(len(self.steps) == 1 and self.steps[0].action_type == "done_proposed", "Completion only recommends verification")
        if self.status == "ask_user":
            require(len(self.steps) == 1 and self.steps[0].action_type == "ask_user", "Question plan can only ask the user")
        if self.status == "continue":
            require(all(step.action_type not in ("done_proposed", "ask_user", "abstain") for step in self.steps), "Terminal labels cannot continue a plan")

    @classmethod
    def from_dict(cls, value, request):
        data = object_fields(value, ("schema", *BINDING_FIELDS, "steps", "status", "confidence", "rationale_short", "metadata"))
        result = cls(**{**data, "steps": tuple(PlanStep.from_dict(item) for item in sequence(data["steps"], MAX_ACTIONS)), "metadata": Metadata.from_dict(data["metadata"])})
        return validate_plan(result, request)


def validate_plan(plan, request):
    require(type(plan) is Plan, "Expected a validated plan")
    validate_binding(plan, request)
    if request.cancelled:
        require(plan.status == "cancelled" and not plan.steps, "Cancelled task cannot be planned")
    elif request.budget_exhausted:
        require(plan.status == "budget_exhausted" and not plan.steps, "Task budget exhausted")
    else:
        require(plan.status not in ("cancelled", "budget_exhausted"), "Incorrect inactive plan status")
    require(len(plan.steps) <= request.task.max_steps - request.task.step_index, "Plan exceeds remaining task budget")
    for offset, step in enumerate(plan.steps):
        require(step.index == request.task.step_index + offset, "Plan indices must be contiguous from current step")
        require(step.action_type in request.allowed_actions, "Planned action outside vocabulary")
        require(step.action_type not in INPUT_ACTIONS or request.method_ceiling == "WINDOW_TARGETED_INPUT", "Planned action exceeds method ceiling")
        if step.action_type in ("semantic_invoke", "set_value", "click", "type", "key", "scroll"):
            require(step.element_ref is not None, "Planned interaction needs an observed target")
        if step.element_ref is not None:
            validate_target(step.element_ref, request)
    return plan


@dataclass(frozen=True, slots=True)
class Decision(Record):
    schema: str
    request_id: str
    tenant_id: str
    task_id: str
    device_id: str
    session_id: str
    observation_generation: str
    route: str
    confidence: float
    abstained: bool
    reason: str
    advisory_only: bool
    metadata: Metadata

    def __post_init__(self):
        require(self.schema == "hala.decision.v1", "Unsupported decision schema")
        for field in BINDING_FIELDS:
            string(getattr(self, field))
        require(self.route in ROUTES, "Unknown advisory route")
        number(self.confidence, 0, 1)
        boolean(self.abstained)
        boolean(self.advisory_only)
        require(self.advisory_only, "Routing cannot grant authority")
        require(self.abstained == (self.route in ("abstain", "budget_exhausted")), "Abstention must match route")
        string(self.reason, 200)
        require(type(self.metadata) is Metadata, "Invalid routing metadata")

    @classmethod
    def from_dict(cls, value, request):
        data = object_fields(value, ("schema", *BINDING_FIELDS, "route", "confidence", "abstained", "reason", "advisory_only", "metadata"))
        result = cls(**{**data, "metadata": Metadata.from_dict(data["metadata"])})
        return validate_decision(result, request)


def validate_decision(decision, request):
    require(type(decision) is Decision, "Expected a validated decision")
    validate_binding(decision, request)
    if request.cancelled:
        require(decision.route == "abstain", "Cancellation cannot be overridden by routing")
    elif request.budget_exhausted:
        require(decision.route == "budget_exhausted", "Budget exhaustion cannot be overridden by routing")
    else:
        require(decision.route != "budget_exhausted", "Incorrect budget exhaustion route")
    return decision
