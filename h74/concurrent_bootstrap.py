"""H74: overlap H72 TRAIN transfer and pinned model prefetch on one approved T4.

No sessions are allocated here. Both independent SHA proofs are required,
and output explicitly forbids training claims. Use --cpu-smoke only with
a tiny synthetic archive to test provider API scheduling, never to qualify
the actual Qwen weights.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from h72.chunk_transfer import transfer, executable, _run, _marker
from h62.run_h61_diagnostic import SESSION_LINE
from h74.remote_prefetch import MODEL_ID, MODEL_REVISION, WEIGHTS_SHA


def trusted_session(session: str, *, cpu_smoke: bool = False, runner=None):
    cli = executable()
    output = (_run(cli, "sessions") if runner is None
              else _run(cli, "sessions", runner=runner))
    hardware = None
    for line in output.splitlines():
        match = SESSION_LINE.match(line.strip())
        if match and match.group("name") == session:
            hardware = match.group("hardware").upper()
            break
    if hardware != "T4" and not (cpu_smoke and hardware == "CPU"):
        raise ValueError("H74 requires an existing permitted free T4; no auto-allocation")
    return hardware


def prefetch(session, *, mode="full", runner=None):
    if mode not in ("metadata", "full"):
        raise ValueError("H74 may only prefetch metadata or full pinned model")
    cli = executable()
    remote = Path(__file__).with_name("remote_prefetch.py")
    if not remote.is_file():
        raise FileNotFoundError("pinned remote model verifier absent")
    args = [
        "exec", "-s", session, "-f", str(remote),
        "--timeout", "1200",
        "--env", "HALA_H74_MODE=" + mode,
        "--env", "HALA_FREE_COLAB_RESEARCH=1",
        "--env", "HALA_NO_PAID_RUNTIME_CONFIRMED=1",
    ]
    output = (_run(cli, *args, timeout=1260) if runner is None
              else _run(cli, *args, timeout=1260, runner=runner))
    result = _marker(output, "H74_PREFETCH")
    expected_status = ("H74_MODEL_SHA_VERIFIED" if mode == "full"
                       else "H74_METADATA_VERIFIED")
    if (not isinstance(result, dict) or result.get("status") != expected_status
            or result.get("model_id") != MODEL_ID
            or result.get("revision") != MODEL_REVISION
            or result.get("gpu_training_executed") is not False):
        raise ValueError("Colab did not verify exact pinned model source")
    if mode == "full" and (
            result.get("model_weights_verified") is not True
            or result.get("model_weights_sha256") != WEIGHTS_SHA):
        raise ValueError("Colab did not SHA-verify complete Qwen model weights")
    return result


def bootstrap(source, zip_sha, git_sha, session, *,
              cpu_smoke=False, workers=3,
              uploader=transfer, model_prefetch=prefetch,
              session_check=trusted_session):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("bounded 1..4 chunk workers required")
    session_check(session, cpu_smoke=cpu_smoke)
    mode = "metadata" if cpu_smoke else "full"
    # One prefetch and one source transfer on the same authorized session:
    # useful work overlaps instead of serially paying both cold starts.
    with ThreadPoolExecutor(max_workers=2) as pool:
        model_future = pool.submit(model_prefetch, session, mode=mode)
        source_future = pool.submit(uploader, source, zip_sha, git_sha,
                                    session, cpu_test=cpu_smoke, workers=workers)
        # Both must complete; a partial model cache or upload is NO-GO.
        model = model_future.result()
        stage = source_future.result()
    if (stage.get("status") != "REMOTE_ZIP_SHA_VERIFIED"
            or stage.get("zip_sha256") != zip_sha or stage.get("git_sha") != git_sha
            or stage.get("gpu_training_executed") is not False):
        raise ValueError("H72 transport proof missing or incorrect")
    if model.get("revision") != MODEL_REVISION:
        raise ValueError("H74 model revision proof missing")
    return {
        "schema": "hala.h74.concurrent-bootstrap.v1",
        "status": ("METADATA_TRANSPORT_SMOKE_VERIFIED" if cpu_smoke else
                   "MODEL_AND_TRAIN_ZIP_SHA_VERIFIED"),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "weights_sha256": (None if cpu_smoke else WEIGHTS_SHA),
        "zip_sha256": zip_sha, "source_git_sha": git_sha,
        "uploaded_chunks": stage["uploaded_chunks"],
        "resumed_verified_chunks": stage["resumed_verified_chunks"],
        "gpu_optimizer_steps": 0,
        "remote_h67_rederivation_executed": False,
        "checkpoint_published": False,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--sha256",required=True)
    parser.add_argument("--git-sha",required=True)
    parser.add_argument("--session",required=True)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--cpu-smoke",action="store_true")
    args=parser.parse_args()
    result=bootstrap(args.source,args.sha256,args.git_sha,args.session,
                     cpu_smoke=args.cpu_smoke,workers=args.workers)
    print("H74_BOOTSTRAP=" + json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
