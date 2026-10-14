"""Untrusted bounded proposals; validation never grants permission to execute."""
from dataclasses import dataclass
import re

from .request import ACTION_TYPES, MAX_ACTIONS, Request
from .validation import Record, boolean, contains, frozen_sequence, number, object_fields, overlaps, require, sequence, string

STATUSES = ("continue", "abstain", "ask_user", "cancelled", "budget_exhausted", "verification_recommended")
INPUT_ACTIONS = frozenset({"click", "type", "key", "scroll"})
ACTION_FIELDS = {
    "semantic_invoke": ("element_ref",),
    "set_value": ("element_ref", "text", "secret"),
    "click": ("point", "coordinate_space", "image_scale", "element_ref_hint"),
    "type": ("element_ref", "text", "secret"),
    "key": ("element_ref", "keys"),
    "scroll": ("element_ref", "delta_x", "delta_y"),
    "wait": ("duration_ms",), "observe": (), "ask_user": ("question",),
    "abstain": ("reason",), "done_proposed": (), "tool_select": ("tool",),
}
NAVIGATION_KEYS = frozenset({"ENTER", "TAB", "ESC", "BACKSPACE", "DELETE", "UP", "DOWN", "LEFT", "RIGHT", "HOME", "END", "PAGEUP", "PAGEDOWN", "SPACE"})


@dataclass(frozen=True, slots=True)
class Metadata(Record):
    model_id: str
    model_revision: str
    prompt_template_revision: str
    serving_config_revision: str
    latency_ms: float
    backend: str
    qualified: bool

    def __post_init__(self):
        for value in (self.model_id, self.model_revision, self.prompt_template_revision, self.serving_config_revision):
            string(value)
        require(re.fullmatch(r"[0-9a-f]{64}", self.model_revision) is not None, "Model revision must be an immutable SHA-256 digest")
        number(self.latency_ms, 0, 120000)
        require(self.backend == "deterministic_mock", "Only the offline mock backend is currently supported")
        boolean(self.qualified)
        require(not self.qualified, "Mock inference cannot claim model qualification")

    @classmethod
    def from_dict(cls, value):
        return cls(**object_fields(value, ("model_id", "model_revision", "prompt_template_revision", "serving_config_revision", "latency_ms", "backend", "qualified")))


@dataclass(frozen=True, slots=True)
class Action(Record):
    type: str
    element_ref: str | None = None
    text: str | None = None
    secret: bool | None = None
    point: tuple | None = None
    coordinate_space: str | None = None
    image_scale: float | None = None
    element_ref_hint: str | None = None
    keys: tuple | None = None
    delta_x: float | None = None
    delta_y: float | None = None
    duration_ms: int | None = None
    question: str | None = None
    reason: str | None = None
    tool: str | None = None

    def __post_init__(self):
        require(type(self.type) is str and self.type in ACTION_TYPES, "Unsupported action")
        data = self.to_dict()
        object_fields(data, ("type", *ACTION_FIELDS[self.type]))
        for field in ("element_ref", "element_ref_hint", "tool"):
            if getattr(self, field) is not None:
                string(getattr(self, field))
        for field in ("text", "question", "reason"):
            if getattr(self, field) is not None:
                string(getattr(self, field), 4096 if field == "text" else 512, empty=field == "text")
        if self.secret is not None:
            boolean(self.secret)
            require(not self.secret, "Secret entry is outside this mock contract")
        if self.point is not None:
            frozen_sequence(self.point, 2)
            require(len(self.point) == 2, "Point needs two coordinates")
            for coordinate in self.point:
                number(coordinate, 0, 8192)
            require(self.coordinate_space in ("window", "viewport"), "Unsupported action coordinate space")
            number(self.image_scale, 0.25, 4)
        if self.keys is not None:
            frozen_sequence(self.keys, 2)
            require(bool(self.keys) and all(type(key) is str for key in self.keys), "Invalid key sequence")
            require((len(self.keys) == 1 and self.keys[0] in NAVIGATION_KEYS) or (len(self.keys) == 2 and self.keys[0] == "CTRL" and self.keys[1] in ("A", "C", "V", "X", "Z")), "Key sequence outside bounded vocabulary")
        for delta in (self.delta_x, self.delta_y):
            if delta is not None:
                number(delta, -1200, 1200)
        if self.type == "scroll":
            require(self.delta_x != 0 or self.delta_y != 0, "Scroll cannot have zero displacement")
        if self.duration_ms is not None:
            number(self.duration_ms, 1, 60000, integer=True)
        if self.tool is not None:
            require(self.tool in ("semantic_invoke", "set_value", "click", "type", "key", "scroll", "wait", "observe"), "Unsupported selected tool")

    @classmethod
    def from_dict(cls, value):
        object_fields(value, ("type",), tuple(field for field in cls.__dataclass_fields__ if field != "type"))
        action_type = value["type"]
        require(type(action_type) is str and action_type in ACTION_TYPES, "Unsupported action")
        object_fields(value, ("type", *ACTION_FIELDS[action_type]))
        data = dict(value)
        for name in ("point", "keys"):
            if name in data:
                data[name] = sequence(data[name], 2)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class Postcondition(Record):
    kind: str
    role: str | None = None
    name: str | None = None
    element_ref: str | None = None
    value: str | None = None

    def __post_init__(self):
        fields = {"element_exists": ("role", "name"), "element_absent": ("element_ref",), "element_value": ("element_ref", "value"), "observation_changed": (), "final_state_check": ()}
        require(type(self.kind) is str and self.kind in fields, "Unsupported postcondition kind")
        object_fields(self.to_dict(), ("kind", *fields[self.kind]))
        for name in fields[self.kind]:
            string(getattr(self, name), 4096 if name == "value" else 512, empty=name in ("name", "value"))

    @classmethod
    def from_dict(cls, value):
        return cls(**object_fields(value, ("kind",), ("role", "name", "element_ref", "value")))


def validate_binding(record, request):
    require(type(request) is Request, "Expected a validated request")
    expected = {"request_id": request.request_id, "tenant_id": request.tenant_id, "task_id": request.task.task_id, "device_id": request.device.device_id, "session_id": request.device.session_id, "observation_generation": request.observation.generation}
    for name, value in expected.items():
        require(getattr(record, name) == value, "Response binding or observation generation mismatch")


def validate_target(reference, request):
    target = next((item for item in request.observation.accessibility if item.element_ref == reference), None)
    require(target is not None, "Target was not observed in this generation")
    require(not request.observation.window.protected and not target.protected, "Protected target cannot be proposed")
    require(not any(overlaps(target.bounds, redaction) for redaction in request.observation.image.redactions), "Target overlaps redacted data")
    return target


def validate_action(action, request):
    require(type(action) is Action, "Invalid action record")
    require(action.type in request.allowed_actions, "Action outside request vocabulary")
    if action.type in INPUT_ACTIONS:
        require(request.method_ceiling == "WINDOW_TARGETED_INPUT", "Action exceeds method ceiling")
    if action.element_ref is not None:
        validate_target(action.element_ref, request)
    if action.type == "click":
        target = validate_target(action.element_ref_hint, request)
        image = request.observation.image
        require(action.coordinate_space == image.coordinate_space and action.image_scale == image.scale, "Coordinate space or DPI mismatch")
        require(action.point[0] < image.width and action.point[1] < image.height, "Point outside captured frame")
        require(contains(target.bounds, action.point), "Click point is outside its observed target")
    if action.type == "tool_select":
        require(action.tool in request.allowed_actions, "Selected tool is not allowed")
        require(action.tool not in INPUT_ACTIONS or request.method_ceiling == "WINDOW_TARGETED_INPUT", "Selected tool exceeds method ceiling")


@dataclass(frozen=True, slots=True)
class Proposal(Record):
    schema: str
    request_id: str
    tenant_id: str
    task_id: str
    device_id: str
    session_id: str
    observation_generation: str
    actions: tuple
    status: str
    confidence: float
    rationale_short: str
    metadata: Metadata
    expected_postcondition: Postcondition | None = None

    def __post_init__(self):
        require(self.schema == "hala.proposal.v1", "Unsupported proposal schema")
        for value in (self.request_id, self.tenant_id, self.task_id, self.device_id, self.session_id, self.observation_generation):
            string(value)
        require(self.status in STATUSES, "Unsupported proposal status")
        number(self.confidence, 0, 1)
        string(self.rationale_short, 200)
        require(type(self.metadata) is Metadata, "Invalid inference metadata")
        require(self.expected_postcondition is None or type(self.expected_postcondition) is Postcondition, "Invalid postcondition record")
        require(all(type(action) is Action for action in frozen_sequence(self.actions, MAX_ACTIONS)), "Invalid actions")
        kinds = tuple(action.type for action in self.actions)
        if self.status in ("cancelled", "budget_exhausted"):
            require(not kinds, "Inactive task cannot have proposed actions")
        elif self.status == "verification_recommended":
            require(kinds == ("done_proposed",), "Completion only recommends independent verification")
        elif self.status == "abstain":
            require(kinds in ((), ("abstain",)), "Abstention cannot carry executable actions")
        elif self.status == "ask_user":
            require(kinds == ("ask_user",), "Question status needs a question only")
        else:
            require(bool(kinds) and not any(kind in ("done_proposed", "abstain", "ask_user") for kind in kinds), "Continue requires bounded nonterminal actions")

    @classmethod
    def from_dict(cls, value, request):
        required = ("schema", "request_id", "tenant_id", "task_id", "device_id", "session_id", "observation_generation", "actions", "status", "confidence", "rationale_short", "metadata")
        data = object_fields(value, required, ("expected_postcondition",))
        parsed = cls(**{**data, "actions": tuple(Action.from_dict(item) for item in sequence(data["actions"], MAX_ACTIONS)), "metadata": Metadata.from_dict(data["metadata"]), "expected_postcondition": Postcondition.from_dict(data["expected_postcondition"]) if data.get("expected_postcondition") is not None else None})
        validate_proposal(parsed, request)
        return parsed


def validate_proposal(proposal, request):
    """Validate advice against current input; Aladdin still decides authority."""
    require(type(proposal) is Proposal, "Expected a validated proposal")
    validate_binding(proposal, request)
    if request.cancelled:
        require(proposal.status == "cancelled" and not proposal.actions, "Cancelled task cannot continue")
    elif request.budget_exhausted:
        require(proposal.status == "budget_exhausted" and not proposal.actions, "Task budget exhausted")
    else:
        require(proposal.status not in ("cancelled", "budget_exhausted"), "Response claims an inactive task incorrectly")
    require(len(proposal.actions) <= request.task.max_steps - request.task.step_index, "Proposal exceeds remaining task budget")
    for action in proposal.actions:
        validate_action(action, request)
    if proposal.expected_postcondition is not None and proposal.expected_postcondition.element_ref is not None:
        validate_target(proposal.expected_postcondition.element_ref, request)
    return proposal
