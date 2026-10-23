"""H70 private original-source domain publication with actual byte-level readback.

Only reads a verified H69 CPU source archive. Never starts GPUs, accesses
held-out answers, or claims model accuracy. The CLI defaults to private;
privacy + owner-authenticated exact-file readback are both mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from h69.domain_archive import EXPECTED_IMAGES, INDEX, unpack, sha

SCHEMA = "hala.h70.private-domain-readback.v1"
OWNER = "abdulazizshehri"
PRIVATE_DOMAINS = frozenset(EXPECTED_IMAGES) - {"slider"}
HEX = re.compile(r"^[0-9a-f]{64}$")
GITSHA = re.compile(r"^[0-9a-f]{40}$")


def provider_command(args, *, cli=subprocess.run):
    completed = cli(
        ["kaggle", *args], capture_output=True, text=True, timeout=3600,
        errors="replace",
    )
    if completed.returncode:
        raise RuntimeError(f"owner Kaggle command failed: {args[1] if len(args)>1 else 'unknown'}")
    return completed


def anonymous_view(ref: str, *, opener=urllib.request.urlopen) -> int:
    request = urllib.request.Request(
        "https://www.kaggle.com/api/v1/datasets/view/" + ref,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with opener(request, timeout=30) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def publish(bundle: Path, domain: str, expected_sha: str, source_git_sha: str,
            receipt_dir: Path, *, cli=subprocess.run, opener=urllib.request.urlopen,
            unpacker=unpack, resume_existing: bool = False,
            sleep=time.sleep) -> dict:
    """Leave input bytes intact even on failure; require full rederivation."""
    if domain not in PRIVATE_DOMAINS:
        raise ValueError("H70 only handles three not-yet-published original TRAIN domains")
    if not isinstance(expected_sha, str) or not HEX.fullmatch(expected_sha):
        raise ValueError("explicit exact archive SHA256 required")
    if not isinstance(source_git_sha, str) or not GITSHA.fullmatch(source_git_sha):
        raise ValueError("reviewed pinned H70 Git SHA required")
    bundle = Path(bundle).resolve()
    if not bundle.is_file() or sha(bundle) != expected_sha:
        raise ValueError("source archive differs from the reviewed SHA256")
    name = f"hala-h70-{domain}-2500.hala.bin"
    # Kaggle dataset slugs allow alphanumeric and hyphens, not underscores.
    # Preserve the canonical domain key in source bytes and file name.
    ref = f"{OWNER}/hala-h70-original-{domain.replace('_', '-')}-2500-private"
    receipt_dir = Path(receipt_dir).resolve()
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if (receipt_dir / f"{domain}-readback.json").exists():
        raise FileExistsError("H70 publication receipt already exists; refuse repeat create")
    # Local preflight before invoking any provider mutation: H69 must fully
    # rederive real upstream TRAIN provenance, not merely trust a ZIP filename.
    with tempfile.TemporaryDirectory(prefix=".h70-preflight-", dir=receipt_dir) as pre:
        pre_root = Path(pre)
        qualified = unpacker(
            bundle, expected_sha, pre_root / "original", source_git_sha, domain)
        if (qualified.get("status") != "VERIFIED_CPU_SOURCE_RESTORED"
                or qualified.get("images") != EXPECTED_IMAGES[domain]):
            raise ValueError("H69 original source proof failed")
    with tempfile.TemporaryDirectory(prefix=".h70-publish-", dir=receipt_dir) as temp:
        staging = Path(temp) / "upload"
        staging.mkdir()
        import shutil
        shutil.copyfile(bundle, staging / name)
        meta = {
            "id": ref,
            "title": f"Hala H70 {domain.replace('_', ' ').title()} TRAIN Private",
            "subtitle": "Pinned original Drag TRAIN screenshots and actions; no model weights",
            "licenses": [{"name": "Apache-2.0"}],
            "isPrivate": True,
        }
        (staging / "dataset-metadata.json").write_text(
            json.dumps(meta, sort_keys=True), encoding="utf8")
        if not resume_existing:
            provider_command(["datasets", "create", "-p", str(staging), "-q", "-r", "skip"],
                             cli=cli)
        # Kaggle can accept an asynchronous upload but return 404/403 for
        # the file index until server-side processing is complete. Never
        # re-create an accepted dataset just because a fresh list failed.
        ready = False
        for attempt in range(18):
            try:
                listed = provider_command(
                    ["datasets", "files", ref, "--csv", "--page-size", "20"], cli=cli)
                ready = name in listed.stdout and str(bundle.stat().st_size) in listed.stdout
            except (RuntimeError, subprocess.TimeoutExpired, OSError):
                ready = False
            if ready:
                break
            if attempt < 17:
                sleep(10)
        if not ready:
            raise RuntimeError("private provider file index not yet available; use explicit resume")
        for attempt in range(12):
            if anonymous_view(ref, opener=opener) == 403:
                break
            if attempt == 11:
                raise ValueError("private dataset access not demonstrated")
            sleep(10)
        readback = Path(temp) / "authenticated-readback"
        readback.mkdir()
        provider_command(
            ["datasets", "download", ref, "-f", name, "-p", str(readback), "-q", "-o"],
            cli=cli)
        returned = readback / name
        if not returned.is_file() or returned.stat().st_size != bundle.stat().st_size:
            raise ValueError("provider returned missing or short source archive")
        if sha(returned) != expected_sha:
            raise ValueError("provider byte SHA256 mismatch")
        recovery = unpacker(
            returned, expected_sha, Path(temp) / "verified-restore",
            source_git_sha, domain)
        if (recovery.get("status") != "VERIFIED_CPU_SOURCE_RESTORED"
                or recovery.get("images") != EXPECTED_IMAGES[domain]
                or recovery.get("actions") != qualified["actions"]
                or recovery.get("labels_sha256") != qualified["labels_sha256"]):
            raise ValueError("provider-returned archive failed H69 source revalidation")
    result = {
        "schema": SCHEMA, "status": "PRIVATE_BYTE_READBACK_VERIFIED",
        "provider_dataset": ref, "domain": domain, "original_images": qualified["images"],
        "original_native_drag_actions": qualified["actions"],
        "source_revision": qualified["source_revision"],
        "source_git_sha": source_git_sha, "archive_sha256": expected_sha,
        "archive_bytes": bundle.stat().st_size,
        "labels_sha256": qualified["labels_sha256"],
        "anonymous_http_status": 403,
        "authenticated_full_byte_readback": True,
        "recovered_source_validator": "VERIFIED_CPU_SOURCE_RESTORED",
        "training_executed": False,
        "gpu_allocated": False,
        "host_archives_deleted": False,
    }
    receipt = receipt_dir / f"{domain}-readback.json"
    receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                       encoding="utf8")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--domain", choices=sorted(PRIVATE_DOMAINS), required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--git-sha", required=True)
    ap.add_argument("--receipts", type=Path, required=True)
    ap.add_argument("--resume-existing", action="store_true",
                    help="verify an already-created owner dataset without re-creating it")
    a = ap.parse_args(argv)
    print(json.dumps(publish(a.bundle, a.domain, a.sha256, a.git_sha, a.receipts,
                             resume_existing=a.resume_existing),
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
