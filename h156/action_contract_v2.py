"""H156 native screenshot-to-action contract v2.

A dropdown action must contain BOTH visible coordinates and an option text.
Controller uses mouse/keyboard only, never a DOM selector or hidden gold.
"""
import json
from dataclasses import dataclass
from h155.action_contract import (
    action_from_dict as old_action, GuiAction, MAX_STEPS, MAX_TEXT,
    H155_SUPPORTED_TASKS
)


def action_from_dict_v2(item):
    if isinstance(item,dict) and item.get("kind") == "select":
        if set(item) != {"kind","point","text"}:
            raise ValueError("Select requires visible list point AND option text")
        # Reuse strict finite native-point validation from original parser.
        point = old_action({"kind":"click","point":item["point"]}).point
        text = old_action({"kind":"type","text":item["text"]}).text
        return GuiAction("select",point,text)
    if isinstance(item,dict) and item.get("kind") == "abstain":
        return old_action(item)
    if isinstance(item,dict) and item.get("kind") in ("click","type","submit"):
        return old_action(item)
    raise ValueError("Unknown or invalid GUI action")


def parse_v2(raw, *, task, model_checkpoint_sha):
    if not (isinstance(model_checkpoint_sha,str) and
            len(model_checkpoint_sha)==64 and
            all(ch in "0123456789abcdef" for ch in model_checkpoint_sha)):
        raise ValueError("Action-producing checkpoint must be SHA-pinned")
    if task not in H155_SUPPORTED_TASKS:
        raise ValueError("Unregistered GUI task")
    if not isinstance(raw,str) or len(raw)>1600:
        raise ValueError("Bounded model output only")
    try:
        doc=json.loads(raw)
    except (ValueError,TypeError) as exc:
        raise ValueError("Invalid exact model JSON") from exc
    if not isinstance(doc,dict) or set(doc)!={"actions"}:
        raise ValueError("Expected only actions")
    steps=doc["actions"]
    if not isinstance(steps,list) or not 1<=len(steps)<=MAX_STEPS:
        raise ValueError("Invalid action steps count")
    actions=tuple(action_from_dict_v2(step) for step in steps)
    if any(a.kind=="abstain" for a in actions):
        if len(actions)!=1:
            raise ValueError("Abstain must be standalone")
        return actions
    if tuple(a.kind for a in actions)!=H155_SUPPORTED_TASKS[task]:
        raise ValueError("Unexpected action sequence for task")
    return actions


def native_mouse_keyboard(page, actions):
    """Execute only model-provided points/strings, no DOM reads, no JS."""
    if not actions or any(a.kind=="abstain" for a in actions):
        return {"status":"ABSTAINED","actions_executed":0}
    for action in actions:
        if action.kind in ("click","submit"):
            page.mouse.click(*action.point)
        elif action.kind=="type":
            page.keyboard.type(action.text,delay=10)
        elif action.kind=="select":
            page.mouse.click(*action.point)
            page.keyboard.press("Home")
            page.keyboard.type(action.text,delay=25)
            page.keyboard.press("Enter")
        else:
            raise ValueError("Unrecognized type")
    return {"status":"ACTIONS_SENT_NATIVE","actions_executed":len(actions)}
