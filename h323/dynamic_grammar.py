"""H323 prospective model-time native grammar for PUBLIC ORIGINAL BeaconBoard.

Distinct from H297 FIELD INPUT/FIC field protocol: first click a STATION
BUTTON, then type a SIM-NN synthetic code into the revealed input.
Neither correct button coordinates nor correct SIM digits are given to
the decoder beyond the user's own visible task text.
"""
from __future__ import annotations

import re
from typing import Any

from h199.native_action_sft import parse_actions
from h273.relative_native_bridge import (
    ABSTAIN, NORMALIZED_REGEX, build_normalized_token_filter,
    normalize_strict_action,
)

STAGE_CLICK = "click_button"
STAGE_TYPE = "type_revealed"
STAGE_DONE = "done"
STAGES = frozenset({STAGE_CLICK, STAGE_TYPE})
STRICT_TYPE = r'\{"actions":\[\{"kind":"type_text","text":"SIM-[0-9]{2}"\}\]\}'
TYPE_REGEX = "(" + ABSTAIN + "|" + STRICT_TYPE + ")"
CLICK_PROMPT = (
    " Observe the 960x720 screenshot. Locate the STATION BUTTON named "
    "in the user task, not a text input. Emit a relative native click "
    "using x and y in 0..999 from the screenshot ONLY; do not use DOM "
    "or gold reference coordinates. Return exactly one JSON click "
    "or abstain action; do not type in this stage."
)
TYPE_PROMPT = (
    " After the previously predicted button click, the input is now "
    "revealed and focused. Observe this NEW screenshot. Return exactly "
    "one JSON type_text with a fictional SIM-NN code matching the "
    "user-visible task, or abstain. Do not click again. "
    "No DOM, reference labels or hidden source hints."
)


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H323 STRICT_DYNAMIC_MODEL_GRAMMAR_REFUSED")


def model_visible_prompt(task: str, stage: str) -> str:
    need(type(task) is str and 10 <= len(task) <= 500
         and not any(c in task for c in "\x00\r\n")
         and stage in STAGES)
    return task + (CLICK_PROMPT if stage == STAGE_CLICK else TYPE_PROMPT)


class DynamicPolicyState:
    def __init__(self) -> None:
        self.stage = STAGE_CLICK
        self.generations = 0
        self.native_clicks = 0
        self.native_types = 0
        self.abstained = False

    def consume_model_text(self, raw: str) -> str:
        need(self.stage in STAGES and self.generations < 2)
        if self.stage == STAGE_CLICK:
            native = normalize_strict_action(raw)
        else:
            need(type(raw) is str and len(raw.encode("utf-8")) <= 120
                 and re.fullmatch(TYPE_REGEX, raw) is not None)
            native = raw
        kind = parse_actions(native)["actions"][0]["kind"]
        if self.stage == STAGE_CLICK:
            need(kind in ("click", "abstain"))
            self.native_clicks += int(kind == "click")
            self.stage = STAGE_TYPE if kind == "click" else STAGE_DONE
        else:
            need(kind in ("type_text", "abstain"))
            self.native_types += int(kind == "type_text")
            self.stage = STAGE_DONE
        self.abstained |= kind == "abstain"
        self.generations += 1
        return native

    def categorical_state(self) -> dict:
        return {
            "generations": self.generations,
            "focus_clicks": self.native_clicks,
            "type_calls": self.native_types,
            "abstained": self.abstained,
            "terminal": self.stage == STAGE_DONE,
        }


def model_time_filter(tokenizer: Any, stage: str):
    """Actual LM-format-enforcer constraints, not posthoc text correction."""
    need(stage in STAGES)
    if stage == STAGE_CLICK:
        return build_normalized_token_filter(tokenizer)
    from lmformatenforcer import (
        RegexParser, TokenEnforcer, TokenEnforcerTokenizerData,
    )
    need(callable(getattr(tokenizer, "encode", None))
         and callable(getattr(tokenizer, "decode", None))
         and getattr(tokenizer, "eos_token_id", None) is not None)
    size = len(tokenizer)
    need(type(size) is int and 0 < size <= 300000)
    anchors = tokenizer.encode("0")
    need(type(anchors) is list and bool(anchors)
         and type(anchors[-1]) is int)
    anchor = anchors[-1]
    special = set(tokenizer.all_special_ids)
    tokens = []
    for i in range(size):
        if i in special:
            continue
        with_anchor = tokenizer.decode([anchor, i])[1:]
        alone = tokenizer.decode([i])
        tokens.append((i, with_anchor, len(with_anchor) > len(alone)))
    data = TokenEnforcerTokenizerData(
        tokens, lambda ids: tokenizer.decode(ids).rstrip("\ufffd"),
        tokenizer.eos_token_id, False, size)
    enforcer = TokenEnforcer(data, RegexParser(TYPE_REGEX))

    def prefix(batch_id, sent):
        need(batch_id == 0)
        ids = sent.tolist()
        need(type(ids) is list and bool(ids))
        allowed = enforcer.get_allowed_tokens(ids).allowed_tokens
        need(bool(allowed))
        return allowed
    return prefix
