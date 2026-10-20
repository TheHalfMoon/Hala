"""Two-arm within-GPU-VM, paired prompt test on exposed H34 synthetic DEV.

No fresh blind TEST, execution authority, or neural improvement claimed.
"""
from __future__ import annotations

import hashlib
import json

from h34.corpus import build_cases, case_sha256
from h34.runner import MODEL, PROMPT as H34_ORIGINAL_PROMPT, REVISION

# Exactly the historical H34 "revised" prompt; source unchanged.
NUMERIC_EXAMPLE = H34_ORIGINAL_PROMPT
# Same task, syntax, abstention, range, and normalized-center semantics.
# No concrete example coordinate pair or digit sequences indicating a point.
NO_NUMERIC_EXAMPLE = NUMERIC_EXAMPLE.replace(
    'For a choice, use exactly this form: {"point":[0.25,0.75]} with normalized center coordinates ',
    'For a choice, output a JSON object with a point key and two numerical entries '
    'with normalized center coordinates ',
)
if NO_NUMERIC_EXAMPLE == NUMERIC_EXAMPLE:
    raise ValueError("H36 no-numeric-example arm did not change historic prompt")
ARMS = ("numeric_example", "no_numeric_example")
PROMPTS = {ARMS[0]: NUMERIC_EXAMPLE, ARMS[1]: NO_NUMERIC_EXAMPLE}
PROTOCOL = "hala.h36.paired_h34_synthetic_dev.v1"
RAW_FIELDS = frozenset({"case_id", "language", "arm", "image_sha256",
                        "prompt_sha256", "raw", "error"})


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def prompt_sha(arm: str) -> str:
    if arm not in PROMPTS:
        raise ValueError("unknown pre-registered prompt arm")
    return sha_bytes(PROMPTS[arm].encode("utf-8"))


def plan() -> list[dict]:
    """Counterbalance prompt order across EN/AR and four UI layouts."""
    cases = build_cases()
    if len(cases) != 32 or case_sha256() != "4b7f849a624caac23dd857adb19ed4293d93d0693dbdbd2921d4423b666a6a3a":
        raise ValueError("H34 frozen case scope changed")
    out = []
    for index, case in enumerate(cases):
        first = ARMS[index % 2]
        for arm in (first, ARMS[1 - index % 2]):
            out.append({"case_id": case["id"], "language": case["language"],
                        "arm": arm, "prompt_sha256": prompt_sha(arm)})
    if len(out) != 64 or any(sum(x["arm"] == arm for x in out) != 32 for arm in ARMS):
        raise ValueError("unexpected paired protocol size")
    return out


def plan_sha() -> str:
    return sha_bytes((json.dumps(plan(), sort_keys=True, separators=(",", ":")) + "\n").encode())


def instructions(case: dict, arm: str) -> str:
    return case["goal"] + "\n" + PROMPTS[arm]
