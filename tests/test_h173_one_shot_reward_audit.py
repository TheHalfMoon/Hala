"""H173 independent one-shot native OOD audit has real paired count guards."""
import hashlib
import json
from collections import Counter

import pytest

from h173 import audit_native
from h173.native_reward import (
    LOCK_NAME,
    REWARD_SCHEMA,
    exact_mcnemar,
    sha,
)
from h173.source_preflight import preregister_provisional_cases


def make_frozen_result(tmp_path, monkeypatch):
    owner = tmp_path / "owner"
    lock = owner / "private-h173" / LOCK_NAME
    lock.parent.mkdir(parents=True)
    monkeypatch.setattr(audit_native, "OOD_ZIP_SHA", "a" * 64)
    cases = preregister_provisional_cases()
    rows = []
    tally = Counter()
    for n, case in enumerate(cases):
        for role in ("before", "after"):
            success = n % 4 == 0 if role == "before" else n % 3 == 0
            rows.append({
                "index": case["index"],
                "role": role,
                "app": case["app"],
                "language": case["language"],
                "image_sha256": hashlib.sha256(
                    str(case["index"]).encode()
                ).hexdigest(),
                "success": success,
                "valid_click": True,
            })
            tally[role + "/calls"] += 1
            tally[role + "/success"] += int(success)
            tally[role + "/" + case["app"] + "/success"] += int(success)
            tally[role + "/" + case["language"] + "/success"] += int(success)
    paired = {(r["index"], r["role"]): r["success"] for r in rows}
    wins = sum(
        paired[(c["index"], "after")] and not paired[(c["index"], "before")]
        for c in cases
    )
    regressions = sum(
        paired[(c["index"], "before")] and not paired[(c["index"], "after")]
        for c in cases
    )
    doc = {
        "schema": REWARD_SCHEMA,
        "status": "H173_ONE_SHOT_GENUINELY_UNSEEN_ORIGINAL_OOD_CONSUMED",
        "cases": 32,
        "real_original_native_browser_rewards": 64,
        "source_ood_zip_sha256": "a" * 64,
        "trained_vision_sha256": audit_native.VISION_WEIGHTS_SHA,
        "gpu_proposals_sha256": "c" * 64,
        "before_success": tally["before/success"],
        "after_success": tally["after/success"],
        "paired_wins": wins,
        "paired_regressions": regressions,
        "two_sided_exact_mcnemar_p": exact_mcnemar(wins, regressions),
        "counts": dict(tally),
        "private_outcomes": rows,
        "real_native_state_not_gold_coordinate_proximity": True,
        "owner_gold_on_gpu": False,
        "old_consumed_eval_used": False,
        "optimizer_steps": 0,
        "paid_gpu": False,
        "flagship_35b_unchanged": True,
    }
    result = tmp_path / "native-reward.json"
    return owner, lock, result, doc


def rewrite(owner, lock, result, doc, lock_mutation=None):
    result.write_text(json.dumps(doc), encoding="utf8")
    receipt = {
        "schema": "hala.h173.owner-private-ood-consumption-lock.v1",
        "status": "H173_ONE_SHOT_FINAL_PERMANENTLY_CONSUMED",
        "source_ood_zip_sha256": "a" * 64,
        "gpu_proposals_sha256": "c" * 64,
        "total_browser_rewards_so_far": 64,
        "private_reward_sha256": sha(result.read_bytes()),
    }
    if lock_mutation:
        lock_mutation(receipt)
    lock.write_text(json.dumps(receipt), encoding="utf8")


def test_exact_mcnemar_from_real_h172_discordant_counts():
    assert exact_mcnemar(13, 2) == 0.00738525390625
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(2, 13) == exact_mcnemar(13, 2)


def test_independent_all_64_native_pair_counts(tmp_path, monkeypatch):
    owner, lock, result, doc = make_frozen_result(tmp_path, monkeypatch)
    rewrite(owner, lock, result, doc)
    audited = audit_native.audit(result, owner)
    assert audited["status"].endswith("NATIVE_VERIFIED")
    assert audited["original_app_ood_permanently_consumed"] is True
    assert audited["before"] == doc["before_success"]
    assert audited["after"] == doc["after_success"]


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(status="H173_NOT_QUALIFIED"),
    lambda d: d.update(real_original_native_browser_rewards=63),
    lambda d: d.update(two_sided_exact_mcnemar_p=0.0),
    lambda d: d.update(after_success=999),
    lambda d: d.update(owner_gold_on_gpu=True),
    lambda d: d.update(old_consumed_eval_used=True),
    lambda d: d["private_outcomes"].pop(),
    lambda d: d["private_outcomes"][0].update(valid_click=False),
    lambda d: d["private_outcomes"][0].update(index=9999),
    lambda d: d["private_outcomes"][0].update(language="xx"),
    lambda d: d["private_outcomes"][32].update(image_sha256="0"*64),
    lambda d: d["private_outcomes"][2].update(
        image_sha256=d["private_outcomes"][0]["image_sha256"]
    ),
])
def test_wrong_or_incomplete_native_ood_receipt_rejected(
    tmp_path, monkeypatch, mutation
):
    owner, lock, result, doc = make_frozen_result(tmp_path, monkeypatch)
    mutation(doc)
    rewrite(owner, lock, result, doc)
    with pytest.raises(ValueError):
        audit_native.audit(result, owner)


def test_partial_started_lock_rejected(tmp_path, monkeypatch):
    owner, lock, result, doc = make_frozen_result(tmp_path, monkeypatch)
    rewrite(owner, lock, result, doc,
            lambda x: x.update(status="H173_ONE_SHOT_STARTED_AND_IRREVOCABLY_CONSUMED"))
    with pytest.raises(ValueError):
        audit_native.audit(result, owner)
