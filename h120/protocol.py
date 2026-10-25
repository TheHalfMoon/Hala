"""H120 immutable next-64 original DragOn EVAL and paired strict GUI protocol.

NEVER touch model training or leak target boxes into model inputs. This
protocol is CPU-only: H119 inference is gated on independently SHA-
verified 1024-step private adapter/checkpoint/optimizer artifacts.
"""
from collections import Counter
from math import comb
from h61.official_eval import DOMAINS, score

SCHEMA="hala.h120.original-next64-three-arm-paired-GUI.v1"
START=92
PER_DOMAIN=16
END=START+PER_DOMAIN
PREVIOUS_H118_START=76
PREVIOUS_H118_END=92
ARMS=("h114_independent_points","h117_conditioned_signed_end",
      "h119_full512_robust_signed_end")
METRICS=("scenes","action_valid","start_hit","end_hit","task_success",
         "task_success_rate","degenerate_start_equals_end")


def freeze_domain(rows):
    """One frozen exact source-listing prefix; later EVAL examples never in TRAIN."""
    if (len(rows)!=END or
            any(not isinstance(x,dict) or
                not isinstance(x.get("id"),str) or not x["id"] or
                not isinstance(x.get("image_sha256"),str) or
                len(x["image_sha256"])!=64 for x in rows) or
            len({x["id"] for x in rows})!=END or
            len({x["image_sha256"] for x in rows})!=END):
        raise ValueError("H120 exact unmodified 108 original EVAL source prefix required")
    prior={x["image_sha256"] for x in rows[PREVIOUS_H118_START:PREVIOUS_H118_END]}
    chosen=rows[START:END]
    if (len(chosen)!=PER_DOMAIN or
            {x["image_sha256"] for x in chosen}&prior):
        raise ValueError("H120 overlaps frozen H118 source")
    return chosen


def freeze_all(domains):
    if not isinstance(domains,dict) or set(domains)!=set(DOMAINS):
        raise ValueError("H120 all four original DragOn EVAL domains required")
    selected=[]
    for domain in DOMAINS:
        rows=domains[domain]
        if (any(row.get("domain")!=domain for row in rows) or
                len(rows)!=END):
            raise ValueError("H120 original EVAL domain drift")
        selected.extend(freeze_domain(rows))
    if len(selected)!=64 or len({s["image_sha256"] for s in selected})!=64:
        raise ValueError("H120 exact 64 unique source images required")
    if len({s["id"] for s in selected})!=64:
        raise ValueError("H120 same original task ID repeated across domains")
    return tuple(selected)


def exact_mcnemar(left,right):
    if (len(left)!=len(right) or not left or
            any(type(x) is not bool for x in (*left,*right))):
        raise ValueError("H120 paired booleans from the same frozen scenes required")
    candidate_only=sum(a and not b for a,b in zip(left,right))
    baseline_only=sum(b and not a for a,b in zip(left,right))
    discordant=candidate_only+baseline_only
    p=min(1.,2*sum(comb(discordant,i) for i in range(min(candidate_only,baseline_only)+1))
                     /(2**discordant)) if discordant else 1.0
    return {"candidate_only_wins":candidate_only,
            "baseline_only_wins":baseline_only,
            "discordant_cases":discordant,
            "two_sided_mcnemar_exact_p":p}


def score_paired(scenes,actions):
    """Pure unchanged H61 scorer; never called during training."""
    if (len(scenes)!=64 or
            not isinstance(actions,dict) or set(actions)!=set(ARMS)
            or len({s["id"] for s in scenes})!=64):
        raise ValueError("H120 fixed 64 paired scenes and all three arms required")
    ids={s["id"] for s in scenes}
    per_arm={}
    outcomes={}
    for arm in ARMS:
        values=actions[arm]
        if (not isinstance(values,dict) or set(values)!=ids or
                any(value is not None and not isinstance(value,str)
                    for value in values.values())):
            raise ValueError("No H120 missing/extra/injected model actions allowed")
        scored=score(list(scenes),values)
        if scored["scenes"]!=64:
            raise ValueError("original H61 scorer output count drift")
        per_arm[arm]={key:scored[key] for key in METRICS}
        outcomes[arm]=[item["task_success"] for item in scored["per_scene"]]
    return {"schema":SCHEMA,"source_split":"eval","source_scenes":64,
            "selected_source_indices":"92..107 per four domains",
            "per_arm":per_arm,
            "h119_vs_h114":exact_mcnemar(outcomes[ARMS[2]],outcomes[ARMS[0]]),
            "h119_vs_h117":exact_mcnemar(outcomes[ARMS[2]],outcomes[ARMS[1]]),
            "unchanged_original_H61_strict_scorer":True,
            "no_oracle_targets_in_inference":True,
            "inference_executed_by_this_module":False,
            "training_executed_by_this_module":False,
            "model_GUI_improved_claimed":False}
