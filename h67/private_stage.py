"""H67: publish one SHA-bound TRAIN stage to the owner's PRIVATE provider.

This is storage, NOT provider-side Kaggle GPU training. The sole large member
is a flat root-level ZIP so even a provider without bulk archive support can
be byte-verified through single-member authenticated readback. Never delete
the only local copy on index-only or partial-byte verification.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from h60 import publish
from h67.stage import SCHEMA, _check_zip, digest


def _full_readback(receipt, files) -> bool:
    """Require actual data bytes for EVERY uploaded member, not just metadata."""
    if receipt.get("provider_readback_files") == len(files) and receipt.get(
            "provider_readback_bytes_mismatched") == 0:
        return True
    if (receipt.get("provider_inventory_complete") is True
            and receipt.get("provider_readback_members_byte_verified_count") == len(files)
            and receipt.get("provider_readback_members_size_verified_count") == 0):
        return True
    return False


def publish_private(bundle: Path, sha: str, git_sha: str, token: str, output: Path,
                    *, provider=publish, delete_host_copy: bool = False) -> dict:
    """Actual provider publication requires authentication + exact byte readback."""
    if not re.fullmatch(r"[a-f0-9]{64}", sha or ""):
        raise ValueError("expected CPU stage SHA256 required")
    if not re.fullmatch(r"[a-f0-9]{40}", git_sha or ""):
        raise ValueError("expected pinned Git SHA required")
    if not token or len(token) < 10:
        raise ValueError("authenticated owner storage token required")
    bundle = Path(bundle).resolve()
    output = Path(output).resolve()
    if not bundle.is_file() or digest(bundle) != sha:
        raise ValueError("CPU stage bundle does not match pinned SHA256")
    import zipfile
    with zipfile.ZipFile(bundle) as z:
        index, _ = _check_zip(z, pinned_sha=git_sha)
    ref = f"{provider.OWNER}/hala-h67-train-{sha[:16]}-private"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output, prefix=".h67-provider-") as td:
        staging = Path(td) / "upload"
        staging.mkdir()
        filename = "hala-h67-train-stage.zip"
        shutil.copyfile(bundle, staging / filename)
        metadata = {
            "title": f"Hala H67 private verified CPU TRAIN {sha[:12]}",
            "id": ref,
            "licenses": [{"name": "Apache-2.0"}],
            "isPrivate": True,
            "subtitle": "CPU-prepared TRAIN source only; no model weights or DEV answers",
        }
        metadata_path = staging / "dataset-metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf8")
        files = {p.name: p.stat().st_size for p in (staging / filename, metadata_path)}
        digests = {p.name: digest(p) for p in (staging / filename, metadata_path)}
        provider.create_private_dataset(staging, ref, metadata["title"], metadata["subtitle"])
        provider.wait_until_visible(ref, token)
        privacy = provider.verify_private(ref, token)
        if (privacy.get("provider_is_private") is not True
                or privacy.get("anonymous_view_http_status") != 403):
            raise ValueError("private provider access was not proved")
        details = provider.verify_readback(ref, token, files, digests, Path(td) / "readback")
        if not _full_readback(details, files):
            raise ValueError("provider did not return every uploaded member byte-exact")
    receipt = {
        "schema": SCHEMA, "status": "PRIVATE_STAGE_BYTE_VERIFIED",
        "provider_ref": ref, "source_git_sha": git_sha,
        "bundle_sha256": sha, "dataset_sha256": index["dataset_sha256"],
        "provider_is_private": True,
        "provider_readback_verified": True,
        "provider_member_hashes_verified": len(files),
        "host_deleted": False,
    }
    receipt_path = output / f"h67-private-{sha[:16]}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf8")
    if delete_host_copy:
        # ONLY the explicitly given archive can be deleted, never CPU source,
        # checkpoints, or unrelated files; provider still holds exact bytes.
        bundle.unlink()
        receipt["host_deleted"] = True
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf8")
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--git-sha", required=True)
    p.add_argument("--receipt-dir", type=Path, required=True)
    p.add_argument("--token-env", default="KAGGLE_API_TOKEN")
    p.add_argument("--delete-archive-after-readback", action="store_true")
    args = p.parse_args(argv)
    receipt = publish_private(args.bundle, args.sha256, args.git_sha,
                              os.environ.get(args.token_env, ""), args.receipt_dir,
                              delete_host_copy=args.delete_archive_after_readback)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
