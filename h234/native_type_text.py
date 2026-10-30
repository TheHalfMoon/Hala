"""H234: bounded bilingual screenshot-native text entry, CPU-only contract.

Extends the frozen H199 action vocabulary without modifying H199 or opening
any owner media, historical OOD, gold, models, network or training assets.
This is NOT training/data admission or a production browser policy.
"""
from __future__ import annotations

import json
import unicodedata
from urllib.parse import urlsplit

from h199.native_action_sft import (
    MAX_JSON_BYTES, _point, _reject_json_constant, _unique_json_object,
    parse_actions as parse_h199_actions,
    require, source_task_prompt,
)

KIND = "type_text"
MAX_TEXT_CODEPOINTS = 160
MAX_TEXT_UTF8_BYTES = 360


def _text(value: object) -> str:
    require(type(value) is str and 1 <= len(value) <= MAX_TEXT_CODEPOINTS,
            "type_text must contain bounded nonempty Unicode text")
    require(len(value.encode("utf-8", errors="surrogatepass")) <= MAX_TEXT_UTF8_BYTES,
            "type_text UTF-8 bytes exceed safe bound")
    require(value == value.strip() and value == unicodedata.normalize("NFC", value),
            "type_text must be trimmed NFC Unicode")
    require(all(not unicodedata.category(ch).startswith("C") for ch in value),
            "type_text contains control, format, surrogate or private-use character")
    require(any(not ch.isspace() for ch in value),
            "type_text cannot be whitespace-only")
    return value


def parse_actions_v4(value: str, *, training_target: bool = False) -> dict:
    """Accept H199 exact one-action forms plus bounded, visible-point type_text.

    Retains strict duplicate-key and non-JSON constant rejection at all depths.
    A valid output is one action only, with no DOM id/selector, macros or
    instructions to execute code, navigate, paste or invoke shell commands.
    """
    require(type(value) is str and len(value.encode("utf-8")) <= MAX_JSON_BYTES,
            "v4 output must be bounded UTF-8 JSON")
    try:
        doc = json.loads(value, object_pairs_hook=_unique_json_object,
                         parse_constant=_reject_json_constant)
    except (ValueError, TypeError) as exc:
        raise ValueError("H234 malformed native JSON") from exc
    require(type(doc) is dict and set(doc) == {"actions"} and
            type(doc["actions"]) is list and len(doc["actions"]) == 1,
            "H234 requires exactly one action")
    action = doc["actions"][0]
    require(type(action) is dict and type(action.get("kind")) is str,
            "H234 action must be exact object")
    if action["kind"] != KIND:
        return parse_h199_actions(value, training_target=training_target)
    require(set(action) == {"kind", "point", "text"},
            "type_text allows only kind, screenshot point and text")
    # Reuse the H199 source coordinate envelope 960x720.
    _point(action["point"])
    _text(action["text"])
    return {"actions": [dict(action)]}


def source_task_prompt_v4(task: str) -> str:
    """Add the new action to the H199 screenshot-only prompt vocabulary."""
    original = source_task_prompt(task)
    needle = "If unsafe or uncertain, output"
    require(original.count(needle) == 1, "H199 prompt contract unexpectedly changed")
    insertion = (
        'To enter text into a visible editable field, use '
        '{"kind":"type_text","point":[x,y],"text":"..."}. '
        "Use only the user's visible task and screenshot, not hidden DOM, "
        "labels or page source. Never generate shortcuts or multiple actions. "
    )
    return original.replace(needle, insertion + needle, 1)


def synthetic_sft_messages_v4(instruction: str, one_native_action_json: str):
    """Create a synthetic-only multimodal message shape, not an admitted batch."""
    native = parse_actions_v4(one_native_action_json, training_target=True)
    user = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": source_task_prompt_v4(instruction)},
        ],
    }]
    completion = [{
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": json.dumps(native, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")),
        }],
    }]
    return user, user + completion


def replay_local_type_text(page, one_native_action_json: str) -> dict:
    """Offline, screenshot-point-only browser action; no DOM-selector targeting.

    The caller must supply an already isolated Playwright page on a file://
    fixture and must separately verify outcome; this does not select a source
    or grant permission to read private owner images.
    """
    parsed = parse_actions_v4(one_native_action_json)
    action = parsed["actions"][0]
    require(action["kind"] == KIND, "offline text replay requires type_text")
    require(urlsplit(page.url).scheme == "file",
            "native text replay only allowed on offline file fixtures")
    x, y = action["point"]
    page.mouse.click(x, y)
    page.keyboard.insert_text(action["text"])
    return {
        "status": "H234_OFFLINE_NATIVE_TYPE_TEXT_ONLY_NOT_DATASET_ADMISSION",
        "action": KIND,
        "model_calls": 0,
        "optimizer_steps": 0,
        "training_approved": False,
        "ood_evaluation_approved": False,
    }
