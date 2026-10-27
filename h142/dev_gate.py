"""Preregistered H142 DEV16 checkpoint go/no-go, never evaluate H140 BLIND.

Frozen original React 2 destinations (Doing+Done), bilingual, no H141
model-evaluated images in the training or DEV split. This is preliminary
one-app qualification, NEVER proof of independent app generalization.
"""
import argparse
import json
from pathlib import Path
from h140.qualification import digest
from h142.train_gpu import TRAIN_SHA,DEV_SHA,STEPS,ACCUM

def qualify(evidence):
    doc=json.loads(Path(evidence).read_text("utf8"))
    if (doc.get("schema")!="hala.h142.fresh-original-visible-column-2b-adapter.v1"
        or doc.get("status")!="COMPLETED_64_REAL_FREE_T4_TRAIN_STEPS"
        or doc.get("steps")!=STEPS or doc.get("accum")!=ACCUM
        or doc.get("train_manifest_sha256")!=TRAIN_SHA
        or doc.get("dev_manifest_sha256")!=DEV_SHA
        or doc.get("source_train_original")!=64
        or doc.get("source_dev_original")!=16
        or doc.get("start_examples_seen")!=64
        or doc.get("visible_heading_examples_seen")!=128
        or doc.get("paid_compute_used") is not False
        or doc.get("h140_blind_model_calls")!=0
        or doc.get("h141_consumed_probe_train_or_dev_used") is not False
        or doc.get("flagship_Hala_35B_updated") is not False):
        raise ValueError("H142 original source-trained LoRA not qualified")
    before=doc["v2_checkpoint_before"]
    after=doc["h142_checkpoint_after"]
    s=after["all"]
    conditions={
        "heading_hit_at_least_10_of_16":s.get("visible_column_heading_hit",0)>=10,
        "source_hit_at_least_13_of_16":s.get("start_hit",0)>=13,
        "both_visible_roles_hit_at_least_10_of_16":s.get("both_hit",0)>=10,
        "both_goals_each_heading_hit_at_least_4_of_8":all(
            after[dest].get("visible_column_heading_hit",0)>=4
            for dest in ("doing","done")),
        "both_languages_each_heading_hit_at_least_4_of_8":all(
            after[lang].get("visible_column_heading_hit",0)>=4
            for lang in ("ar","en")),
        "new_training_gain_in_visible_heading_vs_v2":(
            s.get("visible_column_heading_hit",0)>
            before["all"].get("visible_column_heading_hit",0)),
        "no_source_localization_collapse":(
            s.get("start_hit",0)>=max(13,before["all"].get("start_hit",0)-2)),
        "no_h140_blind_or_flagship_changes":True,
    }
    return {
        "schema":"hala.h142.preregistered-one-app-dev-go-no-go.v1",
        "status":("QUALIFIED_ONLY_FOR_NEW_INDEPENDENT_ACTION_COHORT"
                  if all(conditions.values())
                  else "NO_GO_DO_NOT_TOUCH_H140_BLIND"),
        "checks":conditions,
        "v2_on_new_dev":before,
        "h142_on_new_dev":after,
        "source_train_sha":TRAIN_SHA,"source_dev_sha":DEV_SHA,
        "independent_generalization_proven":False,
        "h140_blind_neural_calls":0,
        "hala_flagship_35B_updated":False,
    }
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--evidence",required=True)
    args=p.parse_args()
    print(json.dumps(qualify(args.evidence),sort_keys=True,indent=2))
