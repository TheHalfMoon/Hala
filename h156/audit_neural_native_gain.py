"""H156 independent real-native DEV paired gain audit; never a BLIND claim."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from h156.audit_train_dev import audit
from h156.train_gpu import ADAPTER_SHA
from h156.replay_model_dev import RESULT, LOCK


def paired_episodes(episodes, expected_cases):
    """Require exactly the same 12 distinct source tasks in both model arms."""
    if not isinstance(episodes, list) or len(episodes) != 24:
        raise ValueError("H156 real native comparison requires 24 episodes")
    expected = {(r["image_sha256"], r["task"]) for r in expected_cases}
    if len(expected) != 12:
        raise ValueError("Frozen DEV requires 12 distinct image/task pairs")
    by_key = {}
    needed = {"role", "task", "image_sha", "parse_valid", "abstained",
              "model_actions_executed", "native_success",
              "native_reward_positive"}
    for r in episodes:
        if not isinstance(r, dict) or set(r) != needed:
            raise ValueError("Native task record missing authoritative fields")
        if (r["role"] not in ("before", "after") or
            type(r["native_success"]) is not bool or
            type(r["native_reward_positive"]) is not bool or
            r["native_success"] != r["native_reward_positive"] or
            type(r["parse_valid"]) is not bool or
            type(r["abstained"]) is not bool or
            type(r["model_actions_executed"]) is not int or
            not 0 <= r["model_actions_executed"] <= 5 or
            (r["abstained"] and (r["native_success"] or r["model_actions_executed"] != 0)) or
            (r["native_success"] and not r["parse_valid"])):
            raise ValueError("Native outcome/abstention contradicts execution")
        key = (r["image_sha"], r["task"])
        if key not in expected:
            raise ValueError("H156 native result outside sealed DEV split")
        joined = (key, r["role"])
        if joined in by_key:
            raise ValueError("H156 duplicated native task arm")
        by_key[joined] = r
    if len(by_key) != 24:
        raise ValueError("Missing paired native model arm")
    stats = Counter()
    per_task = {}
    for case in expected_cases:
        key = (case["image_sha256"], case["task"])
        a = by_key[(key, "before")]["native_success"]
        b = by_key[(key, "after")]["native_success"]
        stats["before"] += int(a)
        stats["after"] += int(b)
        stats["new_wins"] += int(b and not a)
        stats["regressions"] += int(a and not b)
        stats["both_success"] += int(a and b)
        stats["both_fail"] += int(not a and not b)
        count = per_task.setdefault(case["task"], Counter())
        count["tasks"] += 1
        count["before"] += int(a)
        count["after"] += int(b)
    if set(per_task) != {"click-test", "enter-text", "choose-list"} or any(
            v["tasks"] != 4 for v in per_task.values()):
        raise ValueError("Native DEV action families not balanced")
    discordant = stats["new_wins"] + stats["regressions"]
    # Exact two-sided paired sign test conditional on discordant pairs.
    p = (min(1.0, 2.0 * sum(math.comb(discordant, k) for k in range(
         min(stats["new_wins"], stats["regressions"]) + 1)) / (2 ** discordant))
         if discordant else 1.0)
    return {
        "dev_total": 12,
        "before_native_success": stats["before"],
        "after_native_success": stats["after"],
        "gain_tasks": stats["after"]-stats["before"],
        "new_wins": stats["new_wins"],
        "regressions": stats["regressions"],
        "both_success": stats["both_success"],
        "both_fail": stats["both_fail"],
        "paired_exact_sign_p_two_sided": p,
        "per_action_family": {t: dict(c) for t, c in sorted(per_task.items())},
        "same_app_dev_not_blind": True,
        "independent_multisite_transfer_established": False,
    }


def audit_result(owner):
    owner = Path(owner)
    source_receipt = audit(owner)
    directory = owner/"private-h156"
    result_path = directory/RESULT
    lock_path = directory/LOCK
    doc = json.loads(result_path.read_text("utf8"))
    lock = json.loads(lock_path.read_text("utf8"))
    summary = doc.get("summary")
    if (not isinstance(summary, dict) or
        summary.get("schema") != "hala.h156.true-native-neural-dev-replay.v1" or
        summary.get("status") != "COMPLETE_MODEL_DRIVEN_ORIGINAL_MINIWOB_DEV_REPLAY" or
        summary.get("dev_tasks") != 12 or
        summary.get("real_model_native_browser_trials") != 24 or
        summary.get("frozen_initial_lora_sha") != ADAPTER_SHA or
        not isinstance(summary.get("new_lora_sha"), str) or
        len(summary["new_lora_sha"]) != 64 or
        summary["new_lora_sha"] == ADAPTER_SHA or
        summary.get("dev_not_blind") is not True or
        summary.get("cross_app_transfer_proven") is not False or
        summary.get("h140_blind_calls") != 0 or
        lock.get("source_manifest_sha256") != source_receipt["source_sha256"] or
        lock.get("model_checkpoint_sha256") != summary["new_lora_sha"]):
        raise ValueError("Native model DEV provenance receipt invalid")
    cases = json.loads((directory/"miniwob-action-train-dev-v2"/"manifest.json")
                       .read_text("utf8"))["private_cases"]
    dev_cases = [r for r in cases if r["split"] == "DEV"]
    result = paired_episodes(doc.get("owner_private_cases"), dev_cases)
    if (result["before_native_success"] != summary["model_before_native_success"] or
        result["after_native_success"] != summary["model_after_native_success"]):
        raise ValueError("Aggregate model successes contradict verified task cases")
    return {
        "schema": "hala.h156.independently-audited-native-dev-pair.v1",
        "status": "VERIFIED_NATIVE_MODEL_DEV_PAIRED_RESULT",
        "source_sha256": source_receipt["source_sha256"],
        **result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-root", required=True)
    arguments = parser.parse_args()
    print(json.dumps(audit_result(arguments.owner_root), indent=2, sort_keys=True))
