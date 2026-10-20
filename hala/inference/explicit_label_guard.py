"""Experimental host-observed literal-label gate, not a semantic model.

Only explicitly quoted English/Arabic target-label requests are supported. This
is an optional filter on unqualified click *proposals*, not host authorization,
OCR, LLM improvement, or a general natural-language action policy.
"""
from __future__ import annotations

import re
import unicodedata

_EN = re.compile(r"Select the action labeled '([^'\r\n]{1,512})' on the page\.", re.ASCII)
_AR = re.compile(r"اختر الإجراء المكتوب عليه «([^«»\r\n]{1,512})» في الصفحة\.")


def _normalized(text: str) -> str | None:
    if type(text) is not str or not text or len(text) > 512:
        return None
    if text != text.strip() or any(unicodedata.category(c) in ("Cf", "Cc", "Cs") for c in text):
        return None
    return unicodedata.normalize("NFC", text)


def explicit_label(goal: str) -> str | None:
    """Extract a whole-goal *literal* target; reject anything else."""
    if type(goal) is not str or len(goal) > 1024:
        return None
    match = _EN.fullmatch(goal) or _AR.fullmatch(goal)
    if match is None:
        return None
    return _normalized(match.group(1))


def unique_exact_label_match(goal: str, selected: str, names: list[str]) -> bool:
    """Fail closed on ambiguity, unknown goals, bidi controls and duplicates.

    All observed names, including protected/unavailable ones, count toward
    uniqueness; the model does not see the target oracle, only the screenshot.
    """
    expected = explicit_label(goal)
    if expected is None or type(names) is not list or not 1 <= len(names) <= 256:
        return False
    # A malformed or bidi-spoofed observed name makes the entire snapshot
    # unsafe for this *experimental* exact-name gate; never ignore it.
    normalized_names = [_normalized(name) for name in names]
    if any(name is None for name in normalized_names):
        return False
    if _normalized(selected) != expected:
        return False
    return normalized_names.count(expected) == 1
