"""H155 strict image-to-native-GUI action contract; no model inference here.

Existing frozen H142 2B LoRA supports START/visible_column_heading only.
It must NOT silently be treated as capable of MiniWoB clicks/forms.
"""
import json
import math
import re
from dataclasses import dataclass
from typing import Any

W, H = 960, 720
KINDS = ("click", "type", "select", "submit", "abstain")
FROZEN_H142_POINT_ROLES = ("start", "visible_column_heading")
H155_SUPPORTED_TASKS = {
    "click-test": ("click",),
    "enter-text": ("click", "type", "submit"),
    "choose-list": ("select", "submit"),
}
MAX_STEPS = 5
MAX_TEXT = 128


@dataclass(frozen=True)
class GuiAction:
    kind: str
    point: tuple[float, float] | None = None
    text: str | None = None

    def as_dict(self):
        result = {"kind": self.kind}
        if self.point is not None:
            result["point"] = list(self.point)
        if self.text is not None:
            result["text"] = self.text
        return result


def action_from_dict(data: Any) -> GuiAction:
    if not isinstance(data, dict) or type(data.get("kind")) is not str:
        raise ValueError("Action must be a structured object with kind")
    kind = data["kind"]
    if kind not in KINDS:
        raise ValueError("Unknown GUI action (must abstain)")
    keys = ({"kind", "point"} if kind in ("click", "submit") else
            {"kind", "text"} if kind in ("type", "select") else {"kind"})
    if set(data) != keys:
        raise ValueError("Unexpected fields or missing typed action arguments")
    if kind in ("click", "submit"):
        point = data["point"]
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(type(value) not in (int, float)
                       or not math.isfinite(value)
                       for value in point)
                or not 0 <= point[0] < W or not 0 <= point[1] < H):
            raise ValueError("Action point outside screenshot or nonfinite")
        return GuiAction(kind, (float(point[0]), float(point[1])))
    if kind in ("type", "select"):
        value = data["text"]
        if (type(value) is not str or not 0 < len(value) <= MAX_TEXT
                or any(ord(char) < 32 or ord(char) == 127 for char in value)):
            raise ValueError("Typed action value invalid or control characters")
        return GuiAction(kind, text=value)
    return GuiAction("abstain")


def from_neural_response(raw: str, *, task: str, source: str) -> tuple[GuiAction, ...]:
    """Parse genuine model-native JSON, never eval text or silently repair."""
    if source != "h155-new-trained-model":
        raise ValueError("Frozen H142 point-only LoRA is NOT qualified for H155 actions")
    if task not in H155_SUPPORTED_TASKS:
        raise ValueError("New task not preregistered (abstain)")
    if not isinstance(raw, str) or len(raw) > 1600:
        raise ValueError("Neural output absent/too long")
    try:
        document = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Neural output is not exact JSON") from exc
    if not isinstance(document, dict) or set(document) != {"actions"}:
        raise ValueError("Expected exactly actions array")
    data = document["actions"]
    if not isinstance(data, list) or not 1 <= len(data) <= MAX_STEPS:
        raise ValueError("Action count out of bounds")
    actions = tuple(action_from_dict(item) for item in data)
    if any(action.kind == "abstain" for action in actions):
        if actions != (GuiAction("abstain"),):
            raise ValueError("Abstention must be the only action")
        return actions
    allowed = H155_SUPPORTED_TASKS[task]
    if any(item.kind not in allowed for item in actions):
        raise ValueError("Task requested unsupported native action")
    if task == "click-test" and len(actions) != 1:
        raise ValueError("Click-test requires one click")
    if task == "enter-text" and tuple(x.kind for x in actions) != (
            "click", "type", "submit"):
        raise ValueError("Enter-text requires click,type,submit sequence")
    if task == "choose-list" and tuple(x.kind for x in actions) != (
            "select", "submit"):
        raise ValueError("Choose-list requires select,submit sequence")
    return actions


def model_capability_status(model_role: str, task: str) -> dict:
    if task not in H155_SUPPORTED_TASKS:
        raise ValueError("No model transfer task registered")
    if model_role in FROZEN_H142_POINT_ROLES:
        return {
            "status": "UNSUPPORTED_ACTION_ROLE_ABSTAIN",
            "model_actions_attempted": 0,
            "model_task_success_claim": None,
            "source_reference_result_is_not_model_result": True,
        }
    if model_role != "h155-new-trained-model":
        raise ValueError("Unknown model capability role")
    return {
        "status": "ACTION_SCHEMA_ADMITTED_NOT_EVALUATED",
        "model_actions_attempted": 0,
        "model_task_success_claim": None,
        "source_reference_result_is_not_model_result": True,
    }
