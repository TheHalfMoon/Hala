"""H65: upload one verified host-staged checkpoint to approved PRIVATE Kaggle.

Storage is separate from Kaggle GPU training. The provider's authenticated
download is verified byte-for-byte before any optional host cleanup. Tokens
come from an environment variable, never from a committed file or CLI argv.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from h60.publish import OWNER
from h61.publish_checkpoint import publish_bundle, collect, sha256_file
from training.checkpoints import load_checkpoint


SCHEMA = "hala.h65.private-checkpoint-readback.v1"


def publish_latest(root: Path, token: str, work: Path, *,
                   uploader=publish_bundle, remove_host_copy=False) -> dict:
    root = Path(root).resolve()
    if not token or len(token) < 10:
        raise ValueError("authenticated provider token required")
    version, state = load_checkpoint(root, None)
    pointer = json.loads((root / "latest.json").read_text(encoding="utf8"))
    checkpoint_sha = pointer["checkpoint_sha256"]
    ref = (f"{OWNER}/hala-h65-checkpoint-step-{state['step']:08d}-"
           f"{checkpoint_sha[:12]}-private")
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work, prefix=".private-publish-") as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        shutil.copy2(root / "latest.json", stage / "latest.json")
        target = stage / pointer["version"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(version, target)
        # H83: do not claim an authenticated provider readback of a bundle
        # that was inconsistent during the host-to-publish copy.
        staged_version, staged_state = load_checkpoint(stage, None)
        staged_pointer = json.loads((stage / "latest.json").read_text("utf8"))
        if (staged_pointer["checkpoint_sha256"] != checkpoint_sha
                or staged_state != state or staged_version != target.resolve()):
            raise ValueError("staged checkpoint copy differs from verified host identity")
        expected_files = collect(stage)
        expected_hashes = {
            name: sha256_file(path) for name, path in expected_files.items()
        }
        expected_bytes = sum(path.stat().st_size for path in expected_files.values())
        receipt = uploader(
            stage, ref,
            f"Hala verified resume step {state['step']:08d} PRIVATE",
            "Private, byte-verified model adapter and optimizer resume state",
            token, Path(temp) / "provider-work",
        )
    if (not isinstance(receipt, dict)
            or receipt.get("provider_ref") != ref
            or receipt.get("member_sha256") != expected_hashes
            or type(receipt.get("published_members")) is not int
            or receipt["published_members"] != len(expected_hashes)
            or type(receipt.get("published_bytes")) is not int
            or receipt["published_bytes"] != expected_bytes
            or receipt.get("provider_published") is not True
            or receipt.get("provider_is_private") is not True
            or receipt.get("anonymous_provider_http_status") != 403
            or receipt.get("provider_readback_verified") is not True):
        raise ValueError("provider private publication and exact bundle readback NOT verified")
    _, unchanged_host = load_checkpoint(root, None)
    unchanged_pointer = json.loads((root / "latest.json").read_text("utf8"))
    if (unchanged_host != state
            or unchanged_pointer["checkpoint_sha256"] != checkpoint_sha):
        raise ValueError("host checkpoint identity changed during private readback")
    result = {
        "schema": SCHEMA,
        "private_provider_ref": ref,
        "step": state["step"],
        "checkpoint_sha256": checkpoint_sha,
        "provider_published": True,
        "provider_is_private": True,
        "provider_readback_verified": True,
        "host_copy_removed": False,
    }
    receipt_path = work / "private-readback-receipt.json"
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                            encoding="utf8")
    if remove_host_copy:
        # Remove only after authenticated private provider byte-readback. Never
        # delete the persistent provider receipt living outside root.
        shutil.rmtree(root / "checkpoints")
        (root / "latest.json").unlink()
        result["host_copy_removed"] = True
        receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                encoding="utf8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--token-env", default="KAGGLE_API_TOKEN")
    parser.add_argument("--delete-host-after-readback", action="store_true")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    result = publish_latest(args.checkpoint_dir, token, args.work,
                            remove_host_copy=args.delete_host_after_readback)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
