"""H387: stage exact pinned public processor bytes for HF to open.

The owner-supplied cache may change after H218 preflight. Load only
byte-verified copies in a short-lived private staging directory on the
SAME drive, instead of reopening the owner-supplied mutable cache.
No network, weights, model, GPU, private screenshots or legal admission.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from h218.pinned_qwen_processor_bytes import (
    PINNED_PUBLIC_FILES, SNAPSHOT_REVISION, verify_pinned_snapshot,
)


def _require(ok: bool) -> None:
    if not ok:
        raise ValueError("H387_PINNED_PROCESSOR_BYTES_NOT_STABLE")


@contextmanager
def verified_processor_handoff(source: Path) -> Iterator[Path]:
    """Offer HF a verified, temporary copy, never the mutable source path.

    Initial H218 check is not enough: HF reparses individual files later.
    Verify each source byte string again while COPYING, then independently
    verify the staged directory. The HF call must occur INSIDE this context.
    The context and its files are removed on success AND on exceptions.

    Isolation protects against changes to the *original owner cache*.
    It is not protection from an attacker who can modify our own private
    staging directory under the same operating-system credentials.
    """
    _require(isinstance(source, Path) and source.is_absolute()
             and source.name == SNAPSHOT_REVISION
             and source.is_dir() and not source.is_symlink())
    initial = verify_pinned_snapshot(source)
    _require(initial["verified_files"] == 8)
    # Use the owner cache drive, avoiding large copies on the crowded C:.
    with TemporaryDirectory(prefix=".hala-h387-public-", dir=source.parent) as tmp:
        staged = Path(tmp) / SNAPSHOT_REVISION
        staged.mkdir()
        _require(len(PINNED_PUBLIC_FILES) == 8)
        for name, (size, digest) in sorted(PINNED_PUBLIC_FILES.items()):
            original = source / name
            _require(original.is_file() and not original.is_symlink())
            with original.open("rb") as reader:
                raw = reader.read(size + 1)
            _require(len(raw) == size and sha256(raw).hexdigest() == digest)
            (staged / name).write_bytes(raw)
        receipt = verify_pinned_snapshot(staged)
        _require(receipt["verified_files"] == 8
                 and receipt["verified_total_bytes"] ==
                     sum(size for size, _ in PINNED_PUBLIC_FILES.values()))
        yield staged
