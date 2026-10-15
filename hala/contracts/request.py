"""Hala-side bounded v1 observation adapter for proposed Aladdin contracts."""
from dataclasses import dataclass

from .validation import Record, boolean, frozen_sequence, in_frame, number, object_fields, parse_json, rect, require, sequence, string

ALADDIN_REFERENCE_SHA = "0c3c86c8acfb57e3dd3ce83c83ee8cfd3b5035c1"
ACTION_TYPES = frozenset({"semantic_invoke", "set_value", "click", "type", "key", "scroll", "wait", "observe", "ask_user", "abstain", "done_proposed", "tool_select"})
METHOD_CEILINGS = ("STRUCTURED_ONLY", "WINDOW_TARGETED_INPUT")
MAX_ACTIONS = 8


@dataclass(frozen=True, slots=True)
class Task(Record):
    task_id: str
    goal: str
    step_index: int
    max_steps: int

    def __post_init__(self):
        string(self.task_id)
        string(self.goal, 4096)
        number(self.max_steps, 1, 60, integer=True)
        number(self.step_index, 0, self.max_steps, integer=True)

    @classmethod
    def from_dict(cls, value):
        return cls(**object_fields(value, ("task_id", "goal", "step_index", "max_steps")))


@dataclass(frozen=True, slots=True)
class Device(Record):
    device_id: str
    session_id: str
    os: str
    locale: str

    def __post_init__(self):
        for value in (self.device_id, self.session_id, self.os, self.locale):
            string(value)

    @classmethod
    def from_dict(cls, value):
        return cls(**object_fields(value, ("device_id", "session_id", "os", "locale")))


@dataclass(frozen=True, slots=True)
class Window(Record):
    window_ref: str
    title: str
    process: str
    protected: bool = False

    def __post_init__(self):
        string(self.window_ref)
        string(self.title, 512, empty=True)
        string(self.process, 128)
        boolean(self.protected)

    @classmethod
    def from_dict(cls, value):
        return cls(**object_fields(value, ("window_ref", "title", "process"), ("protected",)))


@dataclass(frozen=True, slots=True)
class Element(Record):
    element_ref: str
    role: str
    name: str
    bounds: tuple
    protected: bool = False

    def __post_init__(self):
        string(self.element_ref)
        string(self.role)
        string(self.name, 512, empty=True)
        frozen_sequence(self.bounds, 4)
        rect(self.bounds)
        boolean(self.protected)

    @classmethod
    def from_dict(cls, value):
        data = object_fields(value, ("element_ref", "role", "name", "bounds"), ("protected",))
        return cls(**{**data, "bounds": rect(data["bounds"])})


@dataclass(frozen=True, slots=True)
class Image(Record):
    format: str
    width: int
    height: int
    coordinate_space: str
    scale: float
    redactions: tuple

    def __post_init__(self):
        require(self.format in ("png", "jpeg"), "Unsupported image format")
        number(self.width, 1, 8192, integer=True)
        number(self.height, 1, 8192, integer=True)
        require(self.coordinate_space in ("window", "viewport"), "Unsupported coordinate space")
        number(self.scale, 0.25, 4)
        for bounds in frozen_sequence(self.redactions, 128):
            frozen_sequence(bounds, 4)
            rect(bounds)
            require(in_frame(bounds, self.width, self.height), "Redaction outside captured frame")

    @classmethod
    def from_dict(cls, value):
        data = object_fields(value, ("format", "width", "height", "coordinate_space", "scale", "redactions"))
        return cls(**{**data, "redactions": tuple(rect(item) for item in sequence(data["redactions"], 128))})


@dataclass(frozen=True, slots=True)
class Truncation(Record):
    elements_omitted: int

    def __post_init__(self):
        number(self.elements_omitted, 0, 100000, integer=True)

    @classmethod
    def from_dict(cls, value):
        return cls(**object_fields(value, ("elements_omitted",)))


@dataclass(frozen=True, slots=True)
class Observation(Record):
    generation: str
    window: Window
    accessibility: tuple
    image: Image
    truncation: Truncation

    def __post_init__(self):
        string(self.generation)
        require(type(self.window) is Window and type(self.image) is Image and type(self.truncation) is Truncation, "Invalid observation records")
        elements = frozen_sequence(self.accessibility, 256)
        require(all(type(item) is Element for item in elements), "Invalid element records")
        require(len({item.element_ref for item in elements}) == len(elements), "Duplicate element reference")
        require(all(in_frame(item.bounds, self.image.width, self.image.height) for item in elements), "Element outside captured frame")

    @classmethod
    def from_dict(cls, value):
        data = object_fields(value, ("generation", "window", "accessibility", "image", "truncation"))
        return cls(data["generation"], Window.from_dict(data["window"]), tuple(Element.from_dict(item) for item in sequence(data["accessibility"], 256)), Image.from_dict(data["image"]), Truncation.from_dict(data["truncation"]))


@dataclass(frozen=True, slots=True)
class HistoryEntry(Record):
    step: int
    action: str
    target: str
    result: str
    postcondition: str

    def __post_init__(self):
        number(self.step, 0, 59, integer=True)
        require(type(self.action) is str and self.action in ACTION_TYPES, "Unsupported history action")
        string(self.target, empty=True)
        require(self.result in ("completed", "failed", "cancelled", "unknown"), "Invalid historical result")
        require(self.postcondition in ("passed", "failed", "unknown"), "Invalid historical postcondition")

    @classmethod
    def from_dict(cls, value):
        return cls(**object_fields(value, ("step", "action", "target", "result", "postcondition")))


@dataclass(frozen=True, slots=True)
class Request(Record):
    schema: str
    request_id: str
    tenant_id: str
    task: Task
    device: Device
    observation: Observation
    allowed_actions: tuple
    method_ceiling: str
    history: tuple = ()
    cancelled: bool = False

    def __post_init__(self):
        require(self.schema == "hala.request.v1", "Unsupported request schema")
        string(self.request_id)
        string(self.tenant_id)
        require(type(self.task) is Task and type(self.device) is Device and type(self.observation) is Observation, "Invalid nested request records")
        allowed = frozen_sequence(self.allowed_actions, len(ACTION_TYPES))
        require(all(type(item) is str and item in ACTION_TYPES for item in allowed), "Unsupported allowed action")
        require(len(set(allowed)) == len(allowed), "Duplicate allowed action")
        require(self.method_ceiling in METHOD_CEILINGS, "Unsupported method ceiling")
        boolean(self.cancelled)
        history = frozen_sequence(self.history, 60)
        require(all(type(item) is HistoryEntry for item in history), "Invalid history records")
        require(all(item.step < self.task.step_index for item in history), "History must precede current task step")
        require(len({item.step for item in history}) == len(history), "Duplicate history step")

    @property
    def budget_exhausted(self):
        return self.task.step_index >= self.task.max_steps

    @classmethod
    def from_dict(cls, value):
        data = object_fields(value, ("schema", "request_id", "tenant_id", "task", "device", "observation", "allowed_actions", "method_ceiling"), ("history", "cancelled"))
        return cls(data["schema"], data["request_id"], data["tenant_id"], Task.from_dict(data["task"]), Device.from_dict(data["device"]), Observation.from_dict(data["observation"]), sequence(data["allowed_actions"], len(ACTION_TYPES)), data["method_ceiling"], tuple(HistoryEntry.from_dict(item) for item in sequence(data.get("history", []), 60)), data.get("cancelled", False))

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(parse_json(value))
