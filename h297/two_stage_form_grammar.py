"""H297: strict TWO-STAGE prospective form action grammar for H296 FieldPort.

First Qwen action: normalized 0..999 click to focus or abstain.
Second Qwen action: native bounded fictional type_text or abstain.
Never combine actions, never patch an invalid output, never consult
source DOM/gold, and never claim real model inference/learned skill.
Only a model-time token filter plus safe conversion primitive is built.
"""
from __future__ import annotations

import re
from typing import Any

from h199.native_action_sft import parse_actions
from h273.relative_native_bridge import (
    ABSTAIN, NORMALIZED_REGEX, build_normalized_token_filter,
    normalize_strict_action,
)

STAGE_FOCUS = "focus"
STAGE_TYPE = "type"
STAGE_DONE = "done"
STAGES = frozenset({STAGE_FOCUS, STAGE_TYPE})
TYPE_TEXT = r'\{"actions":\[\{"kind":"type_text","text":"FIC-[0-9]{2}"\}\]\}'
STRICT_TYPE_REGEX = "(" + ABSTAIN + "|" + TYPE_TEXT + ")"
FROZEN_PREFIX = (
    " You are seeing a 960x720 screenshot. For the visible form INPUT "
    "specified in the user's task, output Qwen RELATIVE point x,y 0..999 "
    "where (0,0) is top-left and (999,999) is bottom-right. "
    "Choose the CENTER of the visible input ONLY from the screenshot. "
    "Never output pixel coordinates or hidden DOM/reference points. "
    'Return exactly one JSON object: {"actions":[{"kind":"click","point":[x,y]}]} '
    'or {"actions":[{"kind":"abstain"}]}. No Markdown or other actions.'
)
FROZEN_TYPE_SUFFIX = (
    " The input was just focused in a PREVIOUS step. "
    "Observe the current screenshot. Type ONLY the fictional code "
    "requested in the user's task into the already-focused editable field. "
    "Allowed synthetic code FORMAT is FIC-00 through FIC-99, but this "
    "format does NOT reveal the correct code. "
    "Return exactly one JSON object: "
    '{"actions":[{"kind":"type_text","text":"FIC-NN"}]} '
    'with NN as two digits, or {"actions":[{"kind":"abstain"}]}. '
    "Do not click, combine actions, infer hidden DOM or add Markdown."
)


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H297 prospective two-stage model grammar refused: " + reason)


def model_visible_prompt(user_task: str, stage: str) -> str:
    need(type(user_task) is str and 10 <= len(user_task) <= 500
         and not any(c in user_task for c in "\x00\r\n"),
         "bounded real user-visible task required")
    need(type(stage) is str and stage in STAGES,
         "only a future focus or type model-time step allowed")
    suffix = FROZEN_PREFIX if stage == STAGE_FOCUS else FROZEN_TYPE_SUFFIX
    # The task is user-visible input. No gold coordinates/source IDs
    # are supplied to the native model-time grammar.
    return user_task + suffix


def strict_type_native(raw: str) -> str:
    """Accept original, already-compliant type_text JSON; NO rewriting."""
    need(type(raw) is str and len(raw.encode("utf-8")) <= 120
         and re.fullmatch(STRICT_TYPE_REGEX, raw) is not None,
         "strict prospective type/abstain model-time format")
    parsed = parse_actions(raw)
    need(parsed["actions"][0]["kind"] in ("type_text", "abstain"),
         "no other type phase action or combined focus+type")
    return raw


class TwoStepFormPolicyState:
    """Only predicted ACTION kind changes state; never source gold/DOM."""
    def __init__(self) -> None:
        self.stage = STAGE_FOCUS
        self.generations = 0
        self.native_clicks = 0
        self.native_types = 0
        self.abstained = False

    def consume_model_text(self, raw: str) -> str:
        need(self.stage in STAGES and self.generations < 2,
             "one or two original model generations only")
        stage = self.stage
        native = (normalize_strict_action(raw)
                  if stage == STAGE_FOCUS else strict_type_native(raw))
        kind = parse_actions(native)["actions"][0]["kind"]
        if stage == STAGE_FOCUS:
            need(kind in ("click", "abstain"),
                 "first step must focus with click or abstain")
            self.native_clicks += int(kind == "click")
            self.stage = STAGE_TYPE if kind == "click" else STAGE_DONE
        else:
            need(kind in ("type_text", "abstain"),
                 "second step type or abstain only")
            self.native_types += int(kind == "type_text")
            self.stage = STAGE_DONE
        self.abstained = self.abstained or kind == "abstain"
        self.generations += 1
        return native

    def categorical_state(self) -> dict:
        return {
            "generations": self.generations,
            "focus_clicks": self.native_clicks,
            "type_calls": self.native_types,
            "abstained": self.abstained,
            "terminal": self.stage == STAGE_DONE,
            "actual_pretrained_model_generation_proven": False,
            "independent_blind_ood_or_train_admitted": False,
        }


def model_time_filter(tokenizer: Any, stage: str):
    """Actual LM-format-enforcer token gating; not posthoc JSON cleaning.

    This function is NOT a real run itself. The caller must prove the
    exact T4 owner-private process, real Qwen generation and source
    screenshot/task state independently before reporting model scores.
    """
    need(type(stage) is str and stage in STAGES,
         "cannot generate a third/mixed or unregistered stage")
    if stage == STAGE_FOCUS:
        return build_normalized_token_filter(tokenizer)

    from lmformatenforcer import (
        RegexParser, TokenEnforcer, TokenEnforcerTokenizerData,
    )
    need(callable(getattr(tokenizer, "encode", None))
         and callable(getattr(tokenizer, "decode", None))
         and getattr(tokenizer, "eos_token_id", None) is not None,
         "model-time tokenizer contract")
    size = len(tokenizer)
    need(type(size) is int and 0 < size <= 300000,
         "model-time bounded vocabulary")
    anchors = tokenizer.encode("0")
    need(type(anchors) is list and bool(anchors)
         and type(anchors[-1]) is int,
         "model-time token anchor must exist")
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
    enforcer = TokenEnforcer(data, RegexParser(STRICT_TYPE_REGEX))

    def prefix(batch_id, sent):
        need(batch_id == 0, "one model-time generation sequence only")
        ids = sent.tolist()
        need(type(ids) is list and bool(ids),
             "bounded model-time context required")
        allowed = enforcer.get_allowed_tokens(ids).allowed_tokens
        need(bool(allowed),
             "real model-time grammar cannot continue")
        return allowed
    return prefix
