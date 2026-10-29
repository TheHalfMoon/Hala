"""H218: exact-byte offline Qwen3VL processor provenance gate.

Reads ONLY the 8 previously observed PUBLIC Qwen3-VL processor files in a
user-specified snapshot root. Source SHA-256 values are pinned from owner's
Windows read-only cached processor revision. No model weights, trainer,
private image/gold, other filesystem walks, networking, or historical OOD.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

SNAPSHOT_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
# Verified owner Windows local public HF cache, 2026-10-30, exactly 8 files.
# No secret paths, data or private project files are embedded here.
PINNED_PUBLIC_FILES = MappingProxyType({
    "chat_template.json": (5502, "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4"),
    "config.json": (1505, "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b"),
    "merges.txt": (1671839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "preprocessor_config.json": (390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
    "tokenizer.json": (7032403, "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"),
    "tokenizer_config.json": (10868, "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399"),
    "video_preprocessor_config.json": (385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
})
MAX_FILES = 8
MAX_PUBLIC_FILE_BYTES = 8 * 1024 * 1024


def _deny(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H218 public Qwen processor bytes blocked: " + reason)


def verify_files(
    snapshot: Path, expected: Mapping[str, tuple[int, str]],
) -> dict[str, object]:
    """Strict source-only reusable matcher, allowing synthetic test fixtures.

    Only production verify_pinned_snapshot uses fixed built-in digests.
    The expected mapping is *not* a runtime user-override in the preflight.
    """
    _deny(isinstance(snapshot, Path) and snapshot.is_absolute(),
          "absolute explicit snapshot Path required")
    _deny(snapshot.is_dir() and not snapshot.is_symlink(),
          "snapshot missing or linked")
    _deny(type(expected) in (dict, MappingProxyType)
          and len(expected) == MAX_FILES,
          "exactly eight explicitly pinned public files required")
    _deny(all(
        type(name) is str and name and name not in {".", ".."}
        and "/" not in name and "\\" not in name and not name.startswith(".")
        and type(size) is int and 0 < size <= MAX_PUBLIC_FILE_BYTES
        and type(digest) is str and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest)
        for name, (size, digest) in expected.items()
    ), "invalid trusted file specification")
    # Fail closed on unknown files and directories including model weights:
    # extra safetensors, URLs, symlinks or arbitrary owner media are forbidden.
    observed = list(snapshot.iterdir())
    _deny(len(observed) == MAX_FILES
          and {entry.name for entry in observed} == set(expected),
          "unknown, missing, duplicated or extra model/processor file")
    for name, (size, expected_sha) in sorted(expected.items()):
        p = snapshot / name
        _deny(p.is_file() and not p.is_symlink(),
              "processor file missing, invalid or linked")
        _deny(p.stat().st_size == size,
              "processor file byte length drift")
        h = sha256()
        count = 0
        with p.open("rb") as reader:
            while chunk := reader.read(1024 * 1024):
                count += len(chunk)
                _deny(count <= size, "processor byte stream length drift")
                h.update(chunk)
        _deny(count == size and h.hexdigest() == expected_sha,
              "processor file SHA-256 mismatch")
    return {
        "status": "H218_ALL_EIGHT_PUBLIC_QWEN_PROCESSOR_BYTES_PINNED",
        "verified_files": MAX_FILES,
        "verified_total_bytes": sum(pair[0] for pair in expected.values()),
        "model_weights_loaded": False,
        "optimizer_steps": 0,
        "owner_private_media_read": 0,
    }


def verify_pinned_snapshot(snapshot: Path) -> dict[str, object]:
    """The ONLY real preflight entrypoint, with non-overrideable SHA receipt."""
    _deny(isinstance(snapshot, Path) and snapshot.is_absolute()
          and snapshot.name == SNAPSHOT_REVISION,
          "Qwen3-VL revision directory identity changed")
    return verify_files(snapshot, PINNED_PUBLIC_FILES)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Verify exact 8 already-cached public Qwen3-VL processor files"
    )
    parser.add_argument("--local-processor-snapshot", required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = verify_pinned_snapshot(args.local_processor_snapshot)
    except (OSError, ValueError, TypeError):
        # Report no paths, contents, or hashes if qualification fails.
        raise SystemExit("H218_PINNED_PUBLIC_QWEN_PROCESSOR_BYTES_BLOCKED")
    print(
        receipt["status"],
        "files="+str(receipt["verified_files"]),
        "public_bytes="+str(receipt["verified_total_bytes"]),
        "weights=0 model_calls=0 private_data=0 training=0",
    )


if __name__ == "__main__":
    main()
