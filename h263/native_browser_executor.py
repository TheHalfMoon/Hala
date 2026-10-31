"""H263: execute one validated native H199 action in a real local browser.

This module is an actuator, NOT an evaluator or a model. It does not read
screenshots, reference labels, private files, model weights or DOM task gold.
Caller must separately enforce split admission and sealed blind evaluation.
No raw action, text, point, task, screenshot or per-case receipt is logged.
"""
from __future__ import annotations

from h199.native_action_sft import HEIGHT, WIDTH, parse_actions


def new_offline_context(browser):
    """Use a fixed screenshot coordinate system and block network access."""
    return browser.new_context(
        viewport={"width": WIDTH, "height": HEIGHT},
        offline=True,
        service_workers="block",
        accept_downloads=False,
        permissions=[],
    )


def _editable_is_focused(page) -> bool:
    # Safety check for a type_text action, never exposed to a model.
    return page.evaluate("""() => {
        const e = document.activeElement;
        if (!e || e.disabled || e.readOnly) return false;
        if (e.isContentEditable) return true;
        if (e.tagName === "TEXTAREA") return true;
        return e.tagName === "INPUT" &&
            ["text", "search", "email", "url", "tel", "password", "number"]
                .includes(e.type);
    }""")


def execute_one(page, model_json: str) -> dict[str, object]:
    """Execute at most one strict action, returning safe aggregate metadata.

    Real task success MUST be assessed by a separate private state oracle;
    executed=True only means a browser API call completed.
    """
    parsed = parse_actions(model_json, training_target=False)
    if page.viewport_size != {"width": WIDTH, "height": HEIGHT}:
        raise ValueError("H263 requires a fixed 960x720 viewport")
    action = parsed["actions"][0]
    kind = action["kind"]
    if kind == "abstain":
        return {"kind": kind, "executed": False}
    if kind == "click":
        page.mouse.click(*action["point"])
    elif kind == "drag":
        page.mouse.move(*action["start"])
        page.mouse.down()
        try:
            page.mouse.move(*action["end"], steps=12)
        finally:
            page.mouse.up()
    elif kind == "key_press":
        page.mouse.click(*action["point"])
        page.keyboard.press(action["key"])
    elif kind == "type_text":
        if not _editable_is_focused(page):
            raise ValueError("H263 type_text requires an already focused editable field")
        page.keyboard.insert_text(action["text"])
    elif kind == "scroll":
        page.mouse.move(*action["point"])
        page.mouse.wheel(0, action["dy"])
    else:
        raise ValueError("H263 unsupported action")
    return {"kind": kind, "executed": True}
