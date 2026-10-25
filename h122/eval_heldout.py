"""H122 exact original TRAIN-only 64 END heldout GPU inference.

Requires independently verified finished H122 adapter/optimizer SHA.
No EVAL/TEST data, no training, no checkpoint tuning or publishing.
Failure of preregistered heldout gate is a recorded negative result.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h96.research_probe import require_private_vision_research
from h61.curriculum import load_export as load_h71
from h61.official_eval import canonical
from h85.model_cache import HF_HOME,pinned_cache_kwargs
from h93.eval import _infer
from h122.schedule import TRAIN,HOLDOUT
from h122.research_train import owner_private_schedule
from h122.heldout_source import admit_original_train_heldout
from h122.admit_checkpoint import admit_checkpoint
from h122.heldout_gate import score_heldout
from training.artifacts import sha256_file
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

SCHEMA="hala.h122.private-original-TRAIN64-heldout-END-real-neural-v1"
SUMMARY="h122-private-original-train64-END-summary.json"
RAW="h122-private-original-train64-neural.json"
CASES="h122-private-original-train64-source-cases.json"


def run(*,repo,work,output,attached_inputs=Path("/kaggle/input")):
    repo,work,output=map(Path,(repo,work,output))
    require_private_vision_research(scratch=work,output=output)
    if output.exists() or not str(output).startswith("/kaggle/working/"):
        raise ValueError("H122 create-only private Kaggle output required")
    if work.exists() or not str(work).startswith("/kaggle/tmp/"):
        raise ValueError("H122 create-only original TRAIN scratch required")
    # Source original TRAIN is qualified BEFORE admitting adapter; neither
    # H118/H120 official original PUBLIC EVAL/TEST are ever imported.
    pointfile,schedule=owner_private_schedule(attached_inputs)
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      work/"original-H71",H71_CODE_SHA,rederive=True)
    if (stage.get("status")!="RESTORED_TRAIN_STAGE"
            or stage.get("dataset_sha256")!=TRAIN_SHA):
        raise ValueError("H122 genuine original H71 TRAIN archive missing")
    root=work/"original-H71/export"
    manifest=json.loads((root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    originals=load_h71(root/"h61_drag_curriculum.jsonl",manifest)
    cases=admit_original_train_heldout(originals,root,schedule)
    if (len(cases)!=HOLDOUT or len(schedule.train)!=TRAIN
            or len({s["image_sha256"] for s in cases})!=HOLDOUT):
        raise ValueError("H122 source-original TRAIN 448/64 disjointness failed")
    adapter,model_receipt=admit_checkpoint(
        attached_inputs,repo,originals,root,schedule)
    import torch,transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from peft import PeftModel
    if not torch.cuda.is_available() or transformers.__version__!="5.16.1":
        raise ValueError("H122 private exact pinned T4/transformers inference required")
    os.environ.update(HF_HOME=HF_HOME,HF_HUB_DISABLE_TELEMETRY="1",
                      DO_NOT_TRACK="1")
    torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    core=Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION),
        torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
    model=PeftModel.from_pretrained(
        core,adapter,is_trainable=False,trust_remote_code=False).eval()
    began=time.monotonic()
    outputs,raw=_infer(model,processor,cases,torch)
    if (len(outputs)!=HOLDOUT or len(raw)!=HOLDOUT
            or set(outputs)!={s["id"] for s in cases}):
        raise ValueError("H122 exact all 64 neural output calls required")
    # The heldout scorer verifies original image+prompt SHA and raw model
    # bytes, then reads original TRAIN end boxes only for the strict score.
    scored=score_heldout(cases,raw,schedule)
    if (scored["no_holdout_images_used_for_optimizer"] is not True
            or scored["all_model_raw_screenshot_and_prompt_SHA_verified"] is not True
            or scored["original_heldout_scenes"]!=HOLDOUT):
        raise ValueError("H122 heldout source/model scored evidence not qualified")
    output.mkdir(parents=True,exist_ok=False)
    raw_path=output/RAW
    cases_path=output/CASES
    # Private artifacts only: original source END boxes & neural outputs
    # are NEVER added to GitHub or public HF datasets.
    raw_path.write_bytes(canonical(raw))
    cases_path.write_bytes(canonical(cases))
    report={**scored,
        "schema":SCHEMA,
        "status":"H122_REAL_PRIVATE_ORIGINAL_TRAIN64_HELDOUT_SCORED",
        "heldout_gate_passed":scored["holdout_gate_passed"],
        "original_source_split":"TRAIN_ONLY_HELDOUT",
        "original_training_images":TRAIN,
        "original_heldout_images":HOLDOUT,
        "actual_model_inference_calls":len(raw),
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "owner_verified_real_model_binary_SHA":model_receipt,
        "private_raw_model_sha256":sha256_file(raw_path),
        "private_original_source_cases_sha256":sha256_file(cases_path),
        "no_fresh_PUBLIC_EVAL_or_hidden_TEST_read":True,
        "no_holdout_used_for_training":True,
        "no_training_during_eval":True,
        "external_GUI_skill_claimed":False,
        "paid_compute_used":False,
        "elapsed_seconds":round(time.monotonic()-began,2)}
    (output/SUMMARY).write_bytes(canonical(report))
    return report
