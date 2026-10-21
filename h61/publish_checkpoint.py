"""H61 publish one trained checkpoint bundle and its receipts to private provider storage.

The training run publishes an immutable checkpoint bundle on the remote
session, but the remote session disk is temporary. The founder's rule is that
a model checkpoint and its receipts must live in authorised private storage
and must not depend on a temporary local folder as the only copy.

This module takes the downloaded bundle, creates a NEW private provider
dataset that holds it, and proves the copy: the provider reports the dataset
as private, an unauthenticated request for it is refused with HTTP 403, the
provider index lists exactly the published member names and sizes, and every
published member is read back and matched against the locally computed
SHA-256.

The H60 shard namespace is never touched: this module only ever writes inside
its own H61 namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from h60 import publish

SCHEMA = "hala.h61.checkpoint-publication.v1"
NAMESPACE = "hala-h61-model-checkpoint"
BLOCK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(BLOCK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(source: Path) -> dict[str, Path]:
    """Every file of the bundle, keyed by its relative path."""
    source = Path(source)
    members = {}
    for path in sorted(source.rglob("*")):
        if path.is_file():
            members[str(path.relative_to(source)).replace("\\", "/")] = path
    if not members:
        raise ValueError("An empty checkpoint bundle cannot be published")
    return members


def publish_bundle(source: Path, ref: str, title: str, subtitle: str, token: str,
                   work_dir: Path) -> dict:
    """Publish the bundle privately and prove the provider holds the same bytes."""
    members = collect(source)
    sizes = {name: path.stat().st_size for name, path in members.items()}
    shas = {name: sha256_file(path) for name, path in members.items()}
    upload = Path(work_dir) / "upload"
    if upload.exists():
        shutil.rmtree(upload, ignore_errors=True)
    upload.mkdir(parents=True)
    for name, path in members.items():
        target = upload / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    receipt = {
        "schema": SCHEMA,
        "provider_ref": ref,
        "published_members": len(members),
        "published_bytes": sum(sizes.values()),
        "member_sha256": shas,
        "provider_published": False,
        "provider_readback_verified": False,
    }
    publish.create_private_dataset(upload, ref, title, subtitle)
    publish.wait_until_visible(ref, token)
    privacy = publish.verify_private(ref, token)
    receipt["provider_is_private"] = privacy["provider_is_private"]
    receipt["anonymous_provider_http_status"] = privacy["anonymous_view_http_status"]
    receipt["provider_version"] = privacy["provider_version"]
    receipt["provider_view_total_bytes"] = privacy["provider_view_total_bytes"]
    if not privacy["provider_is_private"]:
        raise ValueError("the provider did not report the checkpoint dataset as private")
    if privacy["anonymous_view_http_status"] != 403:
        raise ValueError("an unauthenticated checkpoint request was not refused with HTTP 403")
    readback = publish.verify_readback(ref, token, sizes, shas, Path(work_dir) / "readback")
    receipt.update(readback)
    receipt["provider_published"] = True
    receipt["provider_readback_verified"] = True
    receipt["published_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    ref = f"{publish.OWNER}/{NAMESPACE}-{args.tag}-private"
    result = publish_bundle(
        Path(args.source),
        ref,
        f"Hala H61 model checkpoint {args.tag} PRIVATE",
        "H61 trained adapter checkpoint and its receipts, private verified bundle",
        args.token,
        Path(args.work),
    )
    Path(args.receipt).write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps({key: result[key] for key in (
        "provider_ref", "published_members", "published_bytes",
        "provider_is_private", "anonymous_provider_http_status",
        "provider_readback_verified")}, sort_keys=True))
