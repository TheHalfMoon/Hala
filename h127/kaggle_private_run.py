"""H127 private, noncommercial, CPU-only Kaggle blind media export."""
from pathlib import Path
import json
import os
import subprocess
import sys

SOURCE_SHA = "__SOURCE_COMMIT_SHA__"
REPO = Path("/kaggle/tmp/hala-h127-source-bound")
OUTPUT = Path("/kaggle/working/hala-h127-private-blind-review")
RECEIPT = Path("/kaggle/working/hala-h127-attempt.json")


def record(obj):
    tmp = RECEIPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", "utf8")
    tmp.replace(RECEIPT)


def checked(*cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError("H127 frozen Git source unavailable or rejected")
    return result.stdout.strip()


def main():
    state = {"schema": "hala.h127.owner-private-blind-source-CPU-attempt.v1",
             "status": "NOT_QUALIFIED", "model_optimizer_steps": 0,
             "new_model_inference_calls": 0, "paid_compute_used": False,
             "commercial_use": False, "original_EVAL_or_TEST_used": False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H127 requires fresh owner private Kaggle runtime")
        REPO.parent.mkdir(parents=True, exist_ok=True)
        checked("git", "init", "-q", str(REPO))
        checked("git", "-C", str(REPO), "remote", "add", "origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git", "-C", str(REPO), "fetch", "--depth", "1", "origin", SOURCE_SHA)
        checked("git", "-C", str(REPO), "checkout", "--detach", "-q", "FETCH_HEAD")
        if checked("git", "-C", str(REPO), "rev-parse", "HEAD") != SOURCE_SHA:
            raise ValueError("H127 frozen source commit mismatch")
        sys.path.insert(0, str(REPO))
        from h73.kaggle_h73_cpu import require_cpu
        require_cpu()
        from h127.private_export import run
        state["stage"] = "PRIVATE_ORIGINAL_TRAIN64_BLIND_PACK"
        record(state)
        result = run(Path("/kaggle/tmp/hala-h127-stage"), OUTPUT)
        if result.get("status") != "PASS_H127_PRIVATE_BLIND_PACK_CREATED_NOT_REVIEWED":
            raise ValueError("H127 owner private blind source bundle not qualified")
        state.update(result)
        record(state)
    except Exception as exc:
        state.update(status="FAIL_H127_SOURCE_BLIND_PACK_NOT_QUALIFIED",
                     failure_class=type(exc).__name__)
        record(state)
        raise


if __name__ == "__main__":
    main()
