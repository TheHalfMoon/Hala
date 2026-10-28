"""H173 exact Git commit pin provided outside the immutable source tree.

Embedding the commit SHA *inside* a file that forms that same commit is
self-referential and cannot establish a stable identity. The owner must
pin a completed published code commit in the PRIVATE kernel environment.
"""
import os
import re

SOURCE_ENV = "HALA_H173_PINNED_SOURCE_SHA"


def validate_source_commit(value):
    """Require an immutable 40-hex Git commit, never a mutable branch/tag."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("H173 owner must explicitly pin exact 40-hex Git source commit")
    if value == "0" * 40:
        raise ValueError("H173 zero source commit is not an actual frozen revision")
    return value


def pinned_source_commit():
    return validate_source_commit(os.environ.get(SOURCE_ENV, ""))
