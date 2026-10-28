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


def verify_clean_local_source_checkout(source_root=None, expected_source_sha=None):
    """Refuse to spend the one-shot OOD benchmark on altered local code.

    Owner supplies the final immutable Git SHA outside the source tree.
    Ignore untracked owner-private files, but reject modified/staged
    tracked source or a checkout pointing to a different Git commit.
    """
    import subprocess
    from pathlib import Path

    expected = validate_source_commit(
        expected_source_sha
        if expected_source_sha is not None
        else pinned_source_commit()
    )
    root = (Path(source_root) if source_root is not None
            else Path(__file__).resolve().parent.parent).resolve(strict=True)
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True, timeout=30, stderr=subprocess.DEVNULL
        ).strip()

    if (
        Path(git("rev-parse", "--show-toplevel")).resolve(strict=True) != root
        or git("rev-parse", "HEAD") != expected
        or git("status", "--porcelain", "--untracked-files=no")
    ):
        raise ValueError(
            "H173 owner native OOD requires exact clean immutable Git source commit"
        )
    return expected
