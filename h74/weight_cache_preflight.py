"""H74: read-only, offline, pinned model-cache inventory (not model qualification).

Never downloads weights, creates a GPU session, imports torch, or trains.
A cache candidate still requires on-device load and a real optimizer step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
SCHEMA = "hala.h74.offline-model-cache.v1"
MAX_INDEX_BYTES = 16 * 1024 * 1024
MAX_SHARDS = 128


def inspect_snapshot(snapshot: Path) -> dict:
    """Reject partial, wrong-revision, or redirecting model weight snapshots."""
    snapshot = Path(snapshot)
    if (snapshot.name != REVISION or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != "models--Qwen--Qwen3-VL-2B-Instruct"
            or not snapshot.is_dir() or snapshot.is_symlink()):
        raise ValueError("pinned model revision snapshot absent or mismatched")
    repo_root = snapshot.parent.parent.resolve()

    def checked_file(relative: str) -> Path:
        name = PurePosixPath(relative)
        if (not relative or len(relative) > 256 or name.is_absolute()
                or any(part in ("", ".", "..") for part in name.parts)
                or "\\" in relative or ":" in relative):
            raise ValueError("untrusted model cache weight filename")
        candidate = snapshot.joinpath(*name.parts)
        if (not candidate.is_file() or candidate.stat().st_size <= 0
                or not candidate.resolve().is_relative_to(repo_root)):
            raise ValueError("missing, empty, or escaping cached weight")
        return candidate

    config = checked_file("config.json")
    if config.stat().st_size > MAX_INDEX_BYTES:
        raise ValueError("oversized model configuration")
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        index = checked_file("model.safetensors.index.json")
        if index.stat().st_size > MAX_INDEX_BYTES:
            raise ValueError("oversized weight index")
        parsed = json.loads(index.read_text("utf8"))
        weights = parsed.get("weight_map") if isinstance(parsed, dict) else None
        if not isinstance(weights, dict) or not weights:
            raise ValueError("missing indexed weight map")
        if len(weights) > 500_000 or any(not isinstance(v, str) for v in weights.values()):
            raise ValueError("invalid indexed weight filenames")
        shards = set(weights.values())
        if len(shards) > MAX_SHARDS:
            raise ValueError("unbounded weight index")
    else:
        shards = {"model.safetensors"}
    weights_bytes = sum(checked_file(name).stat().st_size for name in shards)
    return {
        "schema": SCHEMA, "status": "CACHED_WEIGHT_FILES_CANDIDATE_ONLY",
        "model_id": MODEL_ID, "revision": REVISION,
        "weight_shards_present": len(shards), "weight_bytes_present": weights_bytes,
        "network_calls": 0, "model_loaded": False, "optimizer_steps": 0,
        "checkpoint_produced": False, "runtime_qualified": False,
    }


def local_only_probe(cache_dir: Path, *, snapshot_fetch=None) -> dict:
    """Use the provider's cache API with explicit no-download contract."""
    if snapshot_fetch is None:
        from huggingface_hub import snapshot_download
        snapshot_fetch = snapshot_download
    snapshot = snapshot_fetch(
        repo_id=MODEL_ID, revision=REVISION,
        cache_dir=str(cache_dir), local_files_only=True)
    return inspect_snapshot(Path(snapshot))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/tmp/hala-model-cache/hub"))
    args = parser.parse_args(argv)
    try:
        report = local_only_probe(args.cache_dir)
    except (OSError, ValueError, KeyError, ImportError) as error:
        report = {"schema": SCHEMA, "status": "BLOCKED_NO_VERIFIED_WARM_SNAPSHOT",
                  "model_id": MODEL_ID, "revision": REVISION,
                  "reason_class": type(error).__name__, "network_calls": 0,
                  "model_loaded": False, "optimizer_steps": 0,
                  "checkpoint_produced": False, "runtime_qualified": False}
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "CACHED_WEIGHT_FILES_CANDIDATE_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
