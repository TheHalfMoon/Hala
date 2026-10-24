"""H87: one genuine Qwen LoRA step, original H71 dataset, research only."""
import json, os, platform, time
from pathlib import Path
from h73.kaggle_h73_cpu import H71_CODE_SHA, ARCHIVE_SHA, locate_attached_train
from h81.staged_resume import admit_stage
from training.scripts.train_adapter import train
from training.checkpoints import load_checkpoint
from training.artifacts import sha256_file

def run(*, repo: Path, scratch: Path, output: Path):
    if (platform.system()!="Linux" or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            or os.environ.get("HALA_NONCOMMERCIAL_RESEARCH")!="1"
            or not str(scratch).startswith("/kaggle/tmp/")
            or not str(output).startswith("/kaggle/working/")):
        raise ValueError("private noncommercial Kaggle research runtime required")
    work=scratch/"run"
    evidence=admit_stage(locate_attached_train(),ARCHIVE_SHA,work,H71_CODE_SHA,rederive=True)
    if evidence["status"]!="RESTORED_TRAIN_STAGE":
        raise ValueError("H71 original TRAIN stage not qualified")
    config=json.loads((repo/"experiments/runtime/h61-colab-train-config.json").read_text("utf8"))
    config.update(platform="kaggle",max_steps=1,audit_records=32,gradient_accumulation=1)
    config_path=scratch/"research-config.json"
    config_path.write_text(json.dumps(config,sort_keys=True),encoding="utf8")
    os.environ["HALA_REMOTE_TRAINING_ALLOWED"]="1"
    start=time.monotonic()
    train(str(config_path),str(work/"export/h61_drag_curriculum.jsonl"),
          str(work/"export/h61_drag_curriculum_manifest.json"),
          str(work/"public-dev-baseline-receipt.json"),str(output))
    version,state=load_checkpoint(output,None)
    rows=state["runtime"].get("qualification_steps",[])
    if (state["step"]!=1 or state["changed_parameter_tensors"]<=0 or
            len(rows)!=1 or rows[0].get("changed_parameter_tensors_this_step",0)<=0):
        raise ValueError("no qualified real Qwen optimizer checkpoint")
    receipt={"schema":"hala.h87.noncommercial-kaggle-real-qwen-step.v1",
             "status":"REAL_QWEN_LORA_STEP_CHECKPOINT_VERIFIED",
             "model_id":config["model_id"],"model_revision":config["revision"],
             "dataset_sha256":evidence["dataset_sha256"],
             "step":state["step"],
             "changed_parameter_tensors":state["changed_parameter_tensors"],
             "gradient_norm_before_clip":rows[0]["gradient_norm_before_clip"],
             "checkpoint_sha256":sha256_file(version/"checkpoint.json"),
             "adapter_safetensors_sha256":sha256_file(version/"adapter/adapter_model.safetensors"),
             "checkpoint_version":version.name,
             "elapsed_seconds":round(time.monotonic()-start,1),
             "commercial_use":False,"model_quality_evaluated":False,
             "provider_readback_verified":False}
    (Path("/kaggle/working")/"hala-h87-training-receipt.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True),encoding="utf8")
    return receipt
