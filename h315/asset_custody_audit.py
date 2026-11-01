"""H315: offline, fail-closed asset custody preflight. No data admission.

Reads only a supplied metadata JSON; never dereferences paths or URLs, reads
media bytes, downloads assets, or authorizes TRAIN/DEV/OOD usage.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_SHA = re.compile(r"^[a-f0-9]{40}$")
_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}/[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def _canonical_repo(value: object) -> bool:
    if not isinstance(value, str) or not _REPO.fullmatch(value):
        return False
    return all(not part.endswith(".") for part in value.split("/"))


def _canonical_asset_path(value: object) -> bool:
    """Conservative POSIX-relative identifier, never resolved or opened."""
    if not isinstance(value, str) or not 1 <= len(value) <= 512:
        return False
    for part in value.split("/"):
        if (not part or part in {".", ".."} or part.startswith(" ")
                or part.endswith((" ", "."))):
            return False
        if not all(ch.isalnum() or ch in "._- +@" for ch in part):
            return False
    return True


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict:
    """Reject duplicate JSON keys at every level, including the envelope."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON metadata key")
        result[key] = value
    return result

_ALLOWED = {"source_repo", "source_commit", "asset_path", "asset_sha256",
            "license_spdx", "license_evidence_commit", "author_attestation",
            "rendered_digest", "historical_provenance_review",
            "independent_rights_review"}
_REQUIRED = _ALLOWED


def audit(payload: object) -> dict:
    """Return safe aggregate only. No source paths or identifying contents."""
    if not isinstance(payload, dict) or set(payload) != {"assets"}:
        raise ValueError("invalid manifest envelope")
    assets = payload["assets"]
    if not isinstance(assets, list) or len(assets) > 1000:
        raise ValueError("invalid asset collection")
    blocked = 0
    seen = set()
    seen_bytes = set()
    seen_rendered = set()
    for item in assets:
        if not isinstance(item, dict) or set(item) != _REQUIRED:
            raise ValueError("invalid asset metadata")
        if any(not isinstance(item[k], str) or not item[k] or len(item[k]) > 512
               for k in ("source_repo", "asset_path", "license_spdx")):
            raise ValueError("invalid asset identifier")
        if not _canonical_repo(item["source_repo"]):
            raise ValueError("invalid repository identifier")
        if not _canonical_asset_path(item["asset_path"]):
            raise ValueError("invalid relative asset identifier")
        if any(not isinstance(item[k], str) or not _SHA.fullmatch(item[k])
               for k in ("source_commit", "license_evidence_commit")):
            raise ValueError("invalid source identity")
        if any(not isinstance(item[k], str) or not _DIGEST.fullmatch(item[k])
               for k in ("asset_sha256", "rendered_digest")):
            raise ValueError("invalid content digest")
        if any(type(item[k]) is not bool for k in
               ("author_attestation", "historical_provenance_review",
                "independent_rights_review")):
            raise ValueError("invalid evidence flag")
        identity = (item["source_repo"], item["source_commit"], item["asset_path"])
        if identity in seen:
            raise ValueError("duplicate asset identity")
        seen.add(identity)
        # Same bytes under another repo/path must never bypass custody review.
        # Also reject reused rendered captures from different source identities.
        if item["asset_sha256"] in seen_bytes:
            raise ValueError("duplicate asset content")
        if item["rendered_digest"] in seen_rendered:
            raise ValueError("duplicate rendered content")
        seen_bytes.add(item["asset_sha256"])
        seen_rendered.add(item["rendered_digest"])
        # This audit can NEVER authorize admission, even for all-true evidence.
        blocked += 1
    return {"schema": "hala-h315-custody-v1", "asset_count": len(assets),
            "quarantined_count": blocked, "admitted_count": 0,
            "decision": "NO_ADMISSION"}


def main() -> None:
    parser = argparse.ArgumentParser(description="H315 source metadata custody check")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    try:
        # Fixed upper bound; metadata only, no path traversal from values.
        if args.manifest.stat().st_size > 1_000_000:
            raise ValueError("oversized manifest")
        payload = json.loads(args.manifest.read_text(encoding="utf-8"),
                             object_pairs_hook=_strict_json_object)
        result = audit(payload)
        print(json.dumps(result, sort_keys=True))
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        print('{"decision":"NO_ADMISSION","status":"INVALID_METADATA"}')
        raise SystemExit(2)


if __name__ == "__main__":
    main()
