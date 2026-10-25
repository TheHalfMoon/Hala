"""H123 authentic original TRAIN128 heldout neural inference (unarmed)."""
import json,os,time
from pathlib import Path
from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h96.research_probe import require_private_vision_research
from h61.curriculum import load_export as load_h71
from h61.official_eval import canonical
from h85.model_cache import HF_HOME,pinned_cache_kwargs
from h93.eval import _infer
from h123.schedule import derive,HOLDOUT,TRAIN
from h123.heldout_source import admit_original_train128
from h123.admit_checkpoint import admit
from h123.heldout_gate import score
from training.artifacts import sha256_file
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

SCHEMA="hala.h123.real-private-original-TRAIN128-END-heldout.v1"
RAW="h123-private-original-train128-neural.json"
CASES="h123-private-original-train128-source-cases.json"
SUMMARY="h123-private-original-train128-END-summary.json"


def run(repo,work,output,attached=Path("/kaggle/input")):
    repo,work,output=map(Path,(repo,work,output))
    require_private_vision_research(scratch=work,output=output)
    if output.exists() or work.exists():
        raise ValueError("H123 original TRAIN128 evaluation must be create-only")
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      work/"original-H71",H71_CODE_SHA,rederive=True)
    if stage.get("status")!="RESTORED_TRAIN_STAGE" or stage.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("H123 source H71 original TRAIN rejected")
    root=work/"original-H71/export"
    manifest=json.loads((root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    original=load_h71(root/"h61_drag_curriculum.jsonl",manifest)
    schedule=derive(original,root)
    cases=admit_original_train128(original,root,schedule)
    if len(cases)!=HOLDOUT or len(schedule.train)!=TRAIN:
        raise ValueError("H123 exact original 128 source cases absent")
    adapter,model_evidence=admit(attached,repo,schedule)
    import torch,transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from peft import PeftModel
    if not torch.cuda.is_available() or transformers.__version__!="5.16.1":
        raise ValueError("H123 pinned genuine T4 inference required")
    os.environ.update(HF_HOME=HF_HOME,HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    base=Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION),
        torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
    model=PeftModel.from_pretrained(base,adapter,
                                   is_trainable=False,trust_remote_code=False).eval()
    began=time.monotonic()
    predictions,raw=_infer(model,processor,cases,torch)
    if len(predictions)!=HOLDOUT or len(raw)!=HOLDOUT:
        raise ValueError("H123 exactly 128 original END neural outputs required")
    measured=score(cases,raw,schedule)
    output.mkdir(parents=True,exist_ok=False)
    raw_path=output/RAW;cases_path=output/CASES
    raw_path.write_bytes(canonical(raw))
    cases_path.write_bytes(canonical(cases))
    report={**measured,"schema":SCHEMA,
      "status":"H123_REAL_PRIVATE_ORIGINAL_TRAIN128_HELDOUT_SCORED",
      "actual_model_inference_calls":HOLDOUT,
      "source_model_binary_sha_verified":model_evidence,
      "private_raw_neural_sha256":sha256_file(raw_path),
      "private_original_cases_sha256":sha256_file(cases_path),
      "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
      "elapsed_seconds":round(time.monotonic()-began,2),
      "original_EVAL_or_TEST_used":False,
      "heldout_used_for_optimizer":False,
      "training_during_eval":False,
      "external_GUI_skill_qualified":False,
      "paid_compute_used":False}
    (output/SUMMARY).write_bytes(canonical(report))
    return report
