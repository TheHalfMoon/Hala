"""Owner-only independent validation of actual 64-step H136 LoRA pilot.

No sealed images are admitted here; only frozen original TRAIN288/DEV24,
real model-native raw JSON, saved PEFT tensors and pinned data manifests.
"""
import json
from pathlib import Path
from h136.source import check,sha
from h132.grounder_contract import native_point,MODEL_ID,MODEL_SHA
from h130.browser_env import WIDTH,HEIGHT
from h133.paired_stats import exact_discordant_p

EXPERIMENT="H136_REAL_64_OPTIMIZER_STEP_NATIVE_GUI_LORA_EXPERIMENT"

def score_native(source,predictions):
    if type(predictions) is not list or len(predictions)!=48:
        raise ValueError("H136 must independently verify exactly DEV24 x 2 neural responses")
    stats={"start":{"hit":0,"valid":0},"end":{"hit":0,"valid":0}}
    case_role={}
    n=0
    for case in source["cases"]:
        for role in ("start","end"):
            row=predictions[n]
            n+=1
            if row.get("case_id")!=case["id"] or row.get("role")!=role:
                raise ValueError("H136 frozen DEV neural response/role sequence mismatch")
            point=native_point(row.get("raw"),WIDTH,HEIGHT)
            if row.get("prediction")!=point:
                raise ValueError("H136 model-native point JSON does not match reported pixels")
            b=case[role+"_bbox"]
            is_hit=(point is not None and
                b["x"]<=point[0]<=b["x"]+b["width"] and
                b["y"]<=point[1]<=b["y"]+b["height"])
            if row.get("hit") is not bool(is_hit):
                raise ValueError("H136 claimed hit not matched by browser bbox")
            stats[role]["hit"]+=int(is_hit)
            stats[role]["valid"]+=int(point is not None)
            case_role[(case["id"],role)]=is_hit
    return stats,case_role

def inspect(train_root,dev_root,gpu_private_result,output_file):
    train_root,dev_root,gpu_private_result,output_file=map(
       Path,(train_root,dev_root,gpu_private_result,output_file))
    train=check(train_root,"train")
    dev=check(dev_root,"dev")
    assert len(train["cases"])==288 and len(dev["cases"])==24
    if {r["image_sha256"] for r in train["cases"]}&{
            r["image_sha256"] for r in dev["cases"]}:
        raise ValueError("H136 original TRAIN/DEV screenshot leak")
    file=gpu_private_result/"private-h136-learning-evidence.json"
    receipt=json.loads(file.read_text("utf8"))
    if (receipt.get("status")!=EXPERIMENT or
        receipt.get("actual_optimizer_steps")!=64 or
        receipt.get("base_model")!=MODEL_ID or
        receipt.get("base_revision")!=MODEL_SHA or
        receipt.get("train_original_screens")!=288 or
        receipt.get("dev_original_screens")!=24 or
        receipt.get("sealed_holdout_attached") is not False or
        receipt.get("main_Hala_flagship_weights_updated") is not False or
        receipt.get("paid_compute_used") is not False or
        receipt.get("train_source_manifest_sha")!=sha((train_root/"manifest.json").read_bytes()) or
        receipt.get("dev_source_manifest_sha")!=sha((dev_root/"manifest.json").read_bytes())):
        raise ValueError("H136 frozen GPU LoRA optimizer/source/evaluation identity failed")
    adapter=gpu_private_result/"private-hala-h136-small-handle-adapter"/"adapter_model.safetensors"
    if not adapter.is_file() or sha(adapter.read_bytes())!=receipt.get("adapter_sha256"):
        raise ValueError("H136 actual newly trained PEFT adapter tensor SHA mismatch")
    raw_before=json.loads((gpu_private_result/"private-dev-before.json").read_text("utf8"))
    raw_after=json.loads((gpu_private_result/"private-dev-after.json").read_text("utf8"))
    before,before_roles=score_native(dev,raw_before)
    after,after_roles=score_native(dev,raw_after)
    if before!=receipt.get("dev_before") or after!=receipt.get("dev_after"):
        raise ValueError("H136 reported DEV metrics do not match independent native neural replay")
    paired={}
    for role in ("start","end"):
        base_only=trained_only=0
        for case in dev["cases"]:
            key=(case["id"],role)
            base=before_roles[key]
            trained=after_roles[key]
            base_only+=int(base and not trained)
            trained_only+=int(trained and not base)
        paired[role]={"trained_only":trained_only,"base_only":base_only,
            "p_exact_two_sided":exact_discordant_p(trained_only,base_only)}
    out={
        "status":"H136_OWNER_INDEPENDENT_64_STEP_NATIVE_DEV24_REPLAY_PASS",
        "real_optimizer_steps":64,"train_screens":288,"dev_screens":24,
        "base_2b_sha":MODEL_SHA,"private_adapter_sha256":receipt["adapter_sha256"],
        "before":before,"after":after,"paired":paired,
        "new_sealed24_used":False,
        "full_browser_task_success_proven":False,
        "flagship_Hala_model_updated":False,
        "generalized_skill_proven":False}
    if output_file.exists():raise ValueError("H136 private owner evidence create-only")
    output_file.parent.mkdir(parents=True,exist_ok=True)
    output_file.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n","utf8")
    return out
