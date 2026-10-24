"""H115 future held-out original DragOn 64-scene strict GUI evaluation.

No EVAL/TEST in training and no model publication. Hard fails until actual
H114 private checkpoint bytes and owner verification are known.
"""
from __future__ import annotations
import gc
import hashlib
import json
import os
import time
from pathlib import Path

from h61.official_eval import DOMAINS,REPO,REV,canonical,fetch_domain,load_scenes,apply_frame,score
from h85.model_cache import HF_HOME,pinned_cache_kwargs
from h93.eval import _infer
from h100.eval import require_owner_private_runtime
from h107.eval import admit_h106_checkpoint
from h115.compose import role_input,compose_strict_original_drag
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

START_INDEX=60
PER_DOMAIN=16
END_INDEX=START_INDEX+PER_DOMAIN
ARMS=("h106_original_drag","h114_two_independent_points")
SCHEMA="hala.h115.private-source-disjoint-64-original-drag-vs-trained-points.v1"

# Populated only AFTER H114 actual model completion and independently
# owner-verified SHA; never guess or accept an arbitrary saved adapter.
H114_VERIFIED_ADAPTER_SHA="33838bc150413b9f04e3a36644010ec10540693ec3095827715f08fd1a28dff6"
H114_VERIFIED_CHECKPOINT_SHA="cafae869baaef62c1b2af983e97ed67c5a7138777bf6f608ef286d08b27e39c8"

def frozen(rows):
    if len(rows)!=END_INDEX:
        raise ValueError("pinned original public DragOn EVAL 76 prefix required")
    selected=rows[START_INDEX:END_INDEX]
    if len(selected)!=PER_DOMAIN or len({x["id"] for x in selected})!=PER_DOMAIN:
        raise ValueError("16 genuinely new source cases per domain required")
    return selected

def admit_h114_checkpoint(attached:Path):
    if not (H114_VERIFIED_ADAPTER_SHA and H114_VERIFIED_CHECKPOINT_SHA):
        raise ValueError("H114 independent owner SHA checkpoint not qualified yet")
    candidates=[p.parent for p in Path(attached).rglob("latest.json")
                if "hala-h114-private-dual-point256-checkpoint" in p.as_posix()
                and "hala-h114-private-dual-point256-t4" in p.as_posix()
                and not p.is_symlink()]
    if len(candidates)!=1:raise ValueError("one exact owner-private H114 checkpoint required")
    root=candidates[0]
    version,state=load_checkpoint(root,None)
    receipt=root.parent/"hala-h114-private-point-training-receipt.json"
    if not receipt.is_file() or receipt.is_symlink():
        raise ValueError("original qualified H114 private model receipt missing")
    info=json.loads(receipt.read_text("utf8"))
    sha=sha256_file(version/"adapter/adapter_model.safetensors")
    checkpoint=sha256_file(version/"checkpoint.json")
    from h114.schedule import ORIGINAL_H113_DERIVED_SHA,STEPS
    if (info.get("status")!="H114_REAL_256_ORIGINAL_DUAL_POINT_OPTIMIZER_UPDATES_VERIFIED"
            or info.get("actual_optimizer_steps")!=STEPS
            or state.get("step")!=STEPS or state.get("max_steps")!=STEPS
            or info.get("original_h113_private_dual_point_sha256")!=ORIGINAL_H113_DERIVED_SHA
            or sha!=H114_VERIFIED_ADAPTER_SHA or sha!=info.get("adapter_sha256")
            or checkpoint!=H114_VERIFIED_CHECKPOINT_SHA
            or checkpoint!=info.get("checkpoint_sha256")
            or state.get("runtime",{}).get("h106_joint_lora_verified") is not True
            or not (version/"adapter/adapter_config.json").is_file()):
        raise ValueError("H114 owner-original true dual-point model bytes unqualified")
    return version/"adapter",{"adapter_sha256":sha,
                               "checkpoint_sha256":checkpoint,
                               "actual_point_optimizer_steps":STEPS}

def _expected_input_sha(scenes):
    return [(s["id"],s["image_sha256"],
             hashlib.sha256(s["prompt"].encode("utf8")).hexdigest()) for s in scenes]

def run(*,repo,work,output,attached_inputs=Path("/kaggle/input")):
    require_owner_private_runtime(work=work,output=output)
    repo,work,output=map(Path,(repo,work,output))
    h106=admit_h106_checkpoint(attached_inputs,repo)[0]
    h114,train_receipt=admit_h114_checkpoint(attached_inputs)
    work.mkdir(parents=True,exist_ok=False)
    originals={}
    for domain in DOMAINS:
        folder=work/domain
        fetch_domain(domain,END_INDEX,folder)
        originals[domain]=frozen(apply_frame(load_scenes(folder/"scenes.json"),folder))
    import torch,transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from peft import PeftModel
    if not torch.cuda.is_available() or transformers.__version__!="5.16.1":
        raise ValueError("owner private exact frozen free T4 Qwen processor required")
    os.environ.update(HF_HOME=HF_HOME,HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    scores={};proofs={}
    started=time.monotonic()
    fields=("scenes","action_valid","start_hit","end_hit","task_success",
            "task_success_rate","degenerate_start_equals_end")
    for arm in ARMS:
        core=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION),
            torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
        model=PeftModel.from_pretrained(
            core,h106 if arm=="h106_original_drag" else h114,
            is_trainable=False,trust_remote_code=False).eval()
        scores[arm]={};proofs[arm]={}
        for domain,source in originals.items():
            if arm=="h106_original_drag":
                actions,raw=_infer(model,processor,source,torch)
                if _expected_input_sha(source)!=[(r["id"],r["image_sha256"],r["prompt_sha256"]) for r in raw]:
                    raise ValueError("H106 original matched source/prompt SHA drift")
                proofs[arm][domain]={"original_drag":raw}
            else:
                start_inputs=[role_input(x,"start") for x in source]
                end_inputs=[role_input(x,"end") for x in source]
                starts,first=_infer(model,processor,start_inputs,torch)
                ends,second=_infer(model,processor,end_inputs,torch)
                if (_expected_input_sha(start_inputs)!=[(r["id"],r["image_sha256"],r["prompt_sha256"]) for r in first]
                        or _expected_input_sha(end_inputs)!=[(r["id"],r["image_sha256"],r["prompt_sha256"]) for r in second]):
                    raise ValueError("H114 actual independent role+source prompt SHA drift")
                actions={x["id"]:compose_strict_original_drag(
                    x,starts[x["id"]],ends[x["id"]]) for x in source}
                proofs[arm][domain]={"start":first,"end":second}
            strict=score(source,actions)
            scores[arm][domain]={k:strict[k] for k in fields}
            print("H115_FRESH_64_ORIGINAL_NATIVE_GUI",arm,domain,len(source),flush=True)
        del model,core
        gc.collect();torch.cuda.empty_cache()
    count={arm:sum(scores[arm][d]["scenes"] for d in DOMAINS) for arm in ARMS}
    if count!={a:64 for a in ARMS}:
        raise ValueError("H115 truly source-disjoint original strict 64-case incomplete")
    strict={arm:sum(scores[arm][d]["task_success"] for d in DOMAINS) for arm in ARMS}
    valid={arm:sum(scores[arm][d]["action_valid"] for d in DOMAINS) for arm in ARMS}
    output.mkdir(parents=True,exist_ok=False)
    rawpath=output/"private-raw-neural-attempts.json"
    rawpath.write_bytes(canonical(proofs))
    report={
       "schema":SCHEMA,"status":"H115_REAL_NEW_SOURCE_64_H106_VS_H114_STRICT_GUI_COMPLETE",
       "source_repo":REPO,"source_commit":REV,"source_split":"eval",
       "original_pinned_new_eval_source_indices":"60..75 each four domains",
       "model_revision":MODEL_REVISION,"source_scenes":64,
       "model_arms":list(ARMS),"sample_counts":count,
       "original_h61_strict_scorer_unchanged":True,
       "source_image_BYTES_same_original_reference":True,
       "h114_training_source_original_qualifier":train_receipt,
       "full_drag_strict_successes":strict,
       "native_valid_final_action":valid,
       "extra_model_inferences_for_h114":64,
       "per_domain_scores":scores,
       "raw_model_attempt_sha256":sha256_file(rawpath),
       "model_weights_modified":False,"training_executed_in_eval":False,
       "eval_labels_not_in_model_prompts":True,"hidden_test_used":False,
       "paid_resources_used":False,"private_raw_data_published":False,
       "general_GUI_skill_qualified":False,
       "elapsed_seconds":round(time.monotonic()-started,2)}
    (output/"h115-private-independent-point-strict-eval-summary.json").write_bytes(canonical(report))
    return report
