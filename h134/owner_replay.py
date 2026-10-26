"""Independent owner-only verification of genuine H134 private trained LoRA.

Reparse actual frozen model native responses, independently source-score DEV16,
verify TRAIN/DEV source SHA and adapter weights; never attach sealed test.
"""
import json
from pathlib import Path
from h134.source import validate,sha
from h132.grounder_contract import native_point
from h130.browser_env import WIDTH,HEIGHT

def recompute_dev(source,private_predictions):
    doc=source
    predictions=private_predictions
    if not isinstance(predictions,list) or len(predictions)!=32:
        raise ValueError("H134 independent owner replay needs exactly DEV16 x START/END")
    expected=[(row,role) for row in doc["cases"] for role in ("start","end")]
    totals={"start":{"hit":0,"valid":0},"end":{"hit":0,"valid":0}}
    for (row,role),sample in zip(expected,predictions):
        if sample.get("case_id")!=row["source_id"] or sample.get("role")!=role:
            raise ValueError("H134 source model prediction identity/role/ordering mismatch")
        raw=sample.get("raw")
        parsed=native_point(raw,WIDTH,HEIGHT)
        point=sample.get("prediction")
        if (parsed is None and point is not None) or (
            parsed is not None and point!=parsed):
            raise ValueError("H134 native model text/pixel coordinate mismatch")
        b=row[role+"_bbox"]
        hit=(parsed is not None and b["x"]<=parsed[0]<=b["x"]+b["width"] and
             b["y"]<=parsed[1]<=b["y"]+b["height"])
        if sample.get("hit") is not bool(hit):
            raise ValueError("H134 claimed bbox success not independently verifiable")
        totals[role]["valid"]+=int(parsed is not None)
        totals[role]["hit"]+=int(hit)
    return totals

def inspect(train_dir,dev_dir,private_training_dir,output_file):
    train_dir,dev_dir,private_training_dir=map(Path,(train_dir,dev_dir,private_training_dir))
    train=validate(train_dir,expect="train")
    dev=validate(dev_dir,expect="dev",
                 prior_shas=(r["image_sha256"] for r in train["cases"]))
    evidence=json.loads((private_training_dir/"private-h134-learning-evidence.json").read_text("utf8"))
    if (evidence.get("status")!="H134_REAL_24_OPTIMIZER_STEP_LANGUAGE_LORA_EXPERIMENT" or
        evidence.get("actual_optimizer_steps")!=24 or
        evidence.get("main_Hala_flagship_weights_updated") is not False or
        evidence.get("sealed_holdout_attached") is not False or
        evidence.get("paid_compute_used") is not False or
        evidence.get("train_source_manifest_sha")!=sha((train_dir/"manifest.json").read_bytes()) or
        evidence.get("dev_source_manifest_sha")!=sha((dev_dir/"manifest.json").read_bytes())):
        raise ValueError("H134 owner experiment/source provenance not qualified")
    adapter=private_training_dir/"private-hala-gui-grounder-adapter"/"adapter_model.safetensors"
    if not adapter.is_file() or sha(adapter.read_bytes())!=evidence.get("adapter_sha256"):
        raise ValueError("H134 actual private trained adapter tensor SHA mismatch")
    before=json.loads((private_training_dir/"private-dev-before.json").read_text("utf8"))
    after=json.loads((private_training_dir/"private-dev-after.json").read_text("utf8"))
    before_stats=recompute_dev(dev,before)
    after_stats=recompute_dev(dev,after)
    if before_stats!=evidence.get("dev_before") or after_stats!=evidence.get("dev_after"):
        raise ValueError("H134 private GPU reported dev metrics not independently reproducible")
    output={
        "status":"PASS_H134_OWNER_PRIVATE_24_STEP_ADAPTER_NEURAL_REPLAY",
        "adapter_sha256":evidence["adapter_sha256"],
        "real_optimizer_steps":24,
        "dev_original_scene_count":16,
        "dev_start_before":before_stats["start"]["hit"],
        "dev_start_after":after_stats["start"]["hit"],
        "dev_end_before":before_stats["end"]["hit"],
        "dev_end_after":after_stats["end"]["hit"],
        "private_original_training_screens":128,
        "sealed_test_evaluated":False,
        "main_Hala_flagship_model_improved_proven":False,
        "full_drag_task_success_proven":False,
    }
    output_file=Path(output_file)
    if output_file.exists():raise ValueError("H134 independent owner output create-only")
    output_file.parent.mkdir(parents=True,exist_ok=True)
    output_file.write_text(json.dumps(output,sort_keys=True,indent=2)+"\n","utf8")
    return output
