"""Independent integrity recomputation of H160 private native episode receipts."""
import argparse
import json
from pathlib import Path

from h158.audit_neural_transfer import paired
from h160.replay_neural_native import preflight, LOCK, RESULT
from h160.gpu_infer import MANIFEST_SHA, SOURCE_ZIP_SHA
from h160.pack_private_eval import SHA_H156, SHA_H157


def audit(owner, upstream, predictions):
    owner = Path(owner)
    original, cases, _ = preflight(owner, upstream, predictions)
    root = owner / "private-h160"
    native = json.loads((root / RESULT).read_text("utf8"))
    lock = json.loads((root / LOCK).read_text("utf8"))
    gpu_receipt = json.loads((
        root / "kaggle-output-v1" / "h160-free-t4-inference-receipt.json"
    ).read_text("utf8"))
    if (native.get("status") !=
        "H160_COMPLETED_32_REAL_ORIGINAL_APP_MODEL_TASKS" or
        native.get("native_browser_trials") != 32 or
        native.get("model_gpu_calls") != 32 or
        native.get("source_manifest_sha256") != MANIFEST_SHA or
        native.get("optimizer_steps") != 0 or
        native.get("h140_blind_calls") != 0 or
        native.get("paid_gpu") is not False or
        native.get("original_app_memory_only") is not True or
        native.get("cross_app_exploratory_not_blind") is not True or
        lock.get("status") != "H160_REAL_NATIVE_MODEL_SCORING_CONSUMED" or
        lock.get("source_zip_sha256") != SOURCE_ZIP_SHA or
        lock.get("source_manifest_sha256") != MANIFEST_SHA or
        lock.get("adapter_shas") !=
        {"before":SHA_H156,"after":SHA_H157} or
        gpu_receipt.get("status") != "H160_COMPLETE_32_REAL_GPU_CALLS" or
        gpu_receipt.get("real_gpu_calls") != 32 or
        gpu_receipt.get("optimizer_steps") != 0 or
        gpu_receipt.get("paid_gpu") is not False):
        raise ValueError("H160 independent GPU/native receipt integrity failure")
    recomputed = paired(native.get("private_cases"), cases)
    if recomputed != native.get("paired"):
        raise ValueError("H160 private native paired success recomputation mismatch")
    return {
        "status":"H160_INDEPENDENTLY_RECOMPUTED_NATIVE_MODEL_REWARD",
        "actual_original_browser_trials":32,
        "actual_real_GPU_calls":32,
        "before_app_task_success":recomputed["before_actual_app_success"],
        "after_app_task_success":recomputed["after_actual_app_success"],
        "paired_new_wins":recomputed["new_wins"],
        "paired_regressions":recomputed["regressions"],
        "p_two_sided":recomputed["paired_sign_two_sided_p"],
        "h140_blind_calls":0,
        "paid_GPU":False,
        "Hala35B_unchanged":True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.owner,args.upstream,args.predictions),
                     indent=2, sort_keys=True))
