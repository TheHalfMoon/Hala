"""Independent proof of H173 one-shot genuinely unseen-app native OOD reward.

No models, donor GUI interaction, or private reference coordinates are used
by this audit. It verifies the immutable owner result and consumed lock.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from h173.infer_gpu import OOD_ZIP_SHA
from h172.infer_gpu import VISION_WEIGHTS_SHA
from h173.native_reward import (
    LOCK_NAME, REWARD_SCHEMA, exact_mcnemar, sha
)
from h173.source_preflight import OOD_COUNT, preregister_provisional_cases


def audit(output, owner):
    path = Path(output)
    record = json.loads(path.read_text("utf8"))
    lock = json.loads(
        (Path(owner) / "private-h173" / LOCK_NAME).read_text("utf8")
    )
    if len(OOD_ZIP_SHA) != 64:
        raise ValueError("H173 cannot score unqualified unseen OOD SHA")
    if (
        record.get("schema") != REWARD_SCHEMA
        or record.get("status")
        != "H173_ONE_SHOT_GENUINELY_UNSEEN_ORIGINAL_OOD_CONSUMED"
        or record.get("cases") != OOD_COUNT
        or record.get("real_original_native_browser_rewards") != 64
        or record.get("source_ood_zip_sha256") != OOD_ZIP_SHA
        or record.get("trained_vision_sha256") != VISION_WEIGHTS_SHA
        or record.get("real_native_state_not_gold_coordinate_proximity")
        is not True
        or record.get("owner_gold_on_gpu") is not False
        or record.get("old_consumed_eval_used") is not False
        or record.get("optimizer_steps") != 0
        or record.get("paid_gpu") is not False
        or record.get("flagship_35b_unchanged") is not True
        or lock.get("schema") != "hala.h173.owner-private-ood-consumption-lock.v1"
        or lock.get("status") != "H173_ONE_SHOT_FINAL_PERMANENTLY_CONSUMED"
        or lock.get("source_ood_zip_sha256") != OOD_ZIP_SHA
        or lock.get("gpu_proposals_sha256")
        != record.get("gpu_proposals_sha256")
        or lock.get("total_browser_rewards_so_far") != 64
        or lock.get("private_reward_sha256") != sha(path.read_bytes())
    ):
        raise ValueError("H173 original OOD one-shot private receipt/lock mismatch")
    scenes = preregister_provisional_cases()
    outcomes = record.get("private_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 64:
        raise ValueError("H173 all native one-shot paired outcomes required")
    seen, paired, counts = set(), {}, Counter()
    same_case_image = {}
    image_owner = {}
    index = {scene["index"]: scene for scene in scenes}
    for row in outcomes:
        ix, role = row.get("index"), row.get("role")
        case = index.get(ix)
        key = (ix, role)
        if (
            case is None
            or role not in ("before", "after")
            or key in seen
            or row.get("app") != case["app"]
            or row.get("language") != case["language"]
            or type(row.get("success")) is not bool
            or type(row.get("valid_click")) is not bool
            or row["success"] and not row["valid_click"]
            or not isinstance(row.get("image_sha256"), str)
            or len(row["image_sha256"]) != 64
        ):
            raise ValueError("H173 duplicated or forged owner native action outcome")
        visual_identity = row["image_sha256"]
        if ix in same_case_image and same_case_image[ix] != visual_identity:
            raise ValueError("H173 baseline/candidate owner native OOD case images differ")
        if ix not in same_case_image:
            if visual_identity in image_owner:
                raise ValueError("H173 owner OOD source image reused across cases")
            image_owner[visual_identity] = ix
            same_case_image[ix] = visual_identity
        seen.add(key)
        paired[key] = row["success"]
        counts[role + "/calls"] += 1
        counts[role + "/success"] += int(row["success"])
        counts[role + "/" + case["app"] + "/success"] += int(
            row["success"]
        )
        counts[role + "/" + case["language"] + "/success"] += int(
            row["success"]
        )
    if (
        seen != {
            (case["index"], role) for case in scenes
            for role in ("before", "after")
        }
        or dict(counts) != record.get("counts")
        or len(same_case_image) != OOD_COUNT
        or len(image_owner) != OOD_COUNT
        or counts["before/calls"] != OOD_COUNT
        or counts["after/calls"] != OOD_COUNT
        or counts["before/success"] != record.get("before_success")
        or counts["after/success"] != record.get("after_success")
    ):
        raise ValueError("H173 no real complete paired success evidence")
    wins = sum(
        paired[(c["index"], "after")] and not paired[(c["index"], "before")]
        for c in scenes
    )
    regressions = sum(
        paired[(c["index"], "before")] and not paired[(c["index"], "after")]
        for c in scenes
    )
    p = exact_mcnemar(wins, regressions)
    if (
        record.get("paired_wins") != wins
        or record.get("paired_regressions") != regressions
        or type(record.get("two_sided_exact_mcnemar_p")) not in (int, float)
        or not math.isclose(record["two_sided_exact_mcnemar_p"], p,
                            rel_tol=0, abs_tol=1e-15)
    ):
        raise ValueError("H173 paired significance or regression claim miscomputed")
    return {
        "status": "H173_INDEPENDENT_32_GENUINELY_UNSEEN_ORIGINAL_OOD_NATIVE_VERIFIED",
        "before": counts["before/success"],
        "after": counts["after/success"],
        "paired_wins": wins,
        "paired_regressions": regressions,
        "exact_two_sided_mcnemar_p": p,
        "arabic_before": counts["before/ar/success"],
        "arabic_after": counts["after/ar/success"],
        "english_before": counts["before/en/success"],
        "english_after": counts["after/en/success"],
        "tingle_before": counts["before/tingle/success"],
        "tingle_after": counts["after/tingle/success"],
        "tiny_slider_before": counts["before/tiny-slider/success"],
        "tiny_slider_after": counts["after/tiny-slider/success"],
        "original_app_ood_permanently_consumed": True,
        "new_model_calls": 0,
        "new_optimizer_steps": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output, args.owner), indent=2, sort_keys=True))
