"""H273 strict prospective Qwen3-VL relative-point to native-pixel bridge.

Qwen3-VL official grounding uses a 0..999/1000 relative coordinate
frame. Hala's H199 native browser uses 960x720 absolute pixels.
This TRUSTED deterministic bridge is applied only to prospective
NORMALIZED model outputs, before the unchanged H199 validator. It is
NOT posthoc repair of H267/H268/H269/H270/H271 predictions.
No screenshot, gold DOM or target knowledge enters conversion.
"""
from __future__ import annotations
import json
import re

WIDTH=960
HEIGHT=720
NMAX=999
NUMBER=r"(0|[1-9][0-9]?|[1-9][0-9][0-9])"
ABSTAIN=r'\{"actions":\[\{"kind":"abstain"\}\]\}'
CLICK=r'\{"actions":\[\{"kind":"click","point":\['+NUMBER+","+NUMBER+r'\]\}\]\}'
NORMALIZED_REGEX="("+ABSTAIN+"|"+CLICK+")"

FIRST_STOCK_TASK="Click the +1 button of the FIRST visible inventory item to add one unit."
RELATIVE_PROMPT=(
    FIRST_STOCK_TASK+
    " You are seeing a 960x720 pixel screenshot. "
    "For the point use Qwen3-VL's native RELATIVE coordinate grid, "
    "where x and y are each integers from 0 to 999, with (0,0) "
    "at the top-left and (999,999) at the bottom-right. "
    "Do NOT return pixel coordinates. Choose the point solely from "
    "the screenshot, aiming at the CENTER of the requested control. "
    "Return ONLY one JSON object in exactly one of these formats: "
    '{"actions":[{"kind":"click","point":[x,y]}]} or '
    '{"actions":[{"kind":"abstain"}]}. '
    "Never return Markdown or any extra text. If unsure, abstain."
)


def normalized_to_pixel(coordinate:int, extent:int)->int:
    if (type(coordinate) is not int or not 0<=coordinate<=NMAX
            or type(extent) is not int or not 1<=extent<=4096):
        raise ValueError("H273 invalid normalized point or screen extent")
    return min(extent-1,(coordinate*extent)//1000)


def normalize_strict_action(raw:str)->str:
    """Accept only exact allowed normalized JSON, convert to valid H199.

    No strings/markdown/newlines/floats/extra action keys/gold. This is
    preregistered coordinate-frame conversion and never loosens H199.
    """
    if type(raw) is not str or len(raw)>128 or re.fullmatch(
            NORMALIZED_REGEX,raw) is None:
        raise ValueError("H273 invalid strict normalized model format")
    try:
        data=json.loads(raw)
        actions=data["actions"]
        action=actions[0]
        if action["kind"]=="abstain":
            native={"actions":[{"kind":"abstain"}]}
        elif action["kind"]=="click":
            x,y=action["point"]
            native={"actions":[{"kind":"click","point":[
                normalized_to_pixel(x,WIDTH),
                normalized_to_pixel(y,HEIGHT)]}]}
        else:
            raise ValueError("unsupported strict model action")
    except (KeyError,IndexError,TypeError,ValueError) as error:
        raise ValueError("H273 unexpected model action") from None
    text=json.dumps(native,separators=(",",":"))
    from h199.native_action_sft import parse_actions
    parse_actions(text)
    return text


def build_normalized_token_filter(tokenizer):
    """Pinned LMFE core constrained DURING generation, never JSON cleanup.

    Matches original H268 integration's true vocabulary, decoder and
    transformer prefix callback but with fixed H273 normalized grammar.
    """
    from lmformatenforcer import (
        RegexParser,TokenEnforcer,TokenEnforcerTokenizerData)
    if (not callable(getattr(tokenizer,"encode",None))
            or not callable(getattr(tokenizer,"decode",None))
            or getattr(tokenizer,"eos_token_id",None) is None):
        raise ValueError("H273 tokenizer incompatible with exact grammar")
    size=len(tokenizer)
    if type(size) is not int or not 0<size<=300000:
        raise ValueError("H273 tokenizer size invalid")
    a=tokenizer.encode("0")
    if not a or type(a[-1]) is not int:
        raise ValueError("H273 missing tokenizer anchor")
    anchor=a[-1]
    special=set(tokenizer.all_special_ids)
    tokens=[]
    for i in range(size):
        if i in special:
            continue
        with_anchor=tokenizer.decode([anchor,i])[1:]
        alone=tokenizer.decode([i])
        tokens.append((i,with_anchor,len(with_anchor)>len(alone)))
    data=TokenEnforcerTokenizerData(
        tokens,lambda ids:tokenizer.decode(ids).rstrip("\ufffd"),
        tokenizer.eos_token_id,False,size)
    enforcer=TokenEnforcer(data,RegexParser(NORMALIZED_REGEX))
    def prefix(batch_id,sent):
        if batch_id!=0:
            raise ValueError("H273 only single private source-owned sequence")
        ids=sent.tolist()
        if not isinstance(ids,list) or not ids:
            raise ValueError("H273 missing input tokens")
        allowed=enforcer.get_allowed_tokens(ids).allowed_tokens
        if not allowed:
            raise ValueError("H273 token grammar cannot continue")
        return allowed
    return prefix


def bind_relative_policy(pretrained_policy):
    """Explicit fixed frame task, with no reference point or DOM access.

    Caller controls screenshot freshness and H199 strict parsing.
    """
    if not callable(pretrained_policy):
        raise ValueError("H273 requires real preloaded visual policy")
    from h199.native_action_sft import source_task_prompt
    expected=source_task_prompt(FIRST_STOCK_TASK)
    def predict(png,prompt):
        if prompt != expected:
            raise ValueError("H273 source task prompt identity drift")
        # The raw response exists only in this volatile callback.
        result=pretrained_policy(png,RELATIVE_PROMPT)
        return normalize_strict_action(result)
    return predict
