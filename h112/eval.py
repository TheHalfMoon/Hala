"""H112 actual-model Qwen normalized-vs-source-pixel GUI grounding A/B.

New pinned public DragOn EVAL indices 56..59/domain, 16 unseen scenes.
Four arms: original base/H106 each with existing original H61 pixel prompt
versus Qwen native [0,1000] relative prompt. Only strict normalization
converted to H61 original source pixels BEFORE unmodified H61 scoring.
"""
from __future__ import annotations
import gc,hashlib,os,time
from pathlib import Path

from h61.official_eval import DOMAINS,REPO,REV,canonical,fetch_domain,load_scenes,apply_frame,score
from h85.model_cache import HF_HOME,pinned_cache_kwargs
from h93.eval import _infer
from h100.eval import require_owner_private_runtime
from h107.eval import admit_h106_checkpoint
from h112.coordinates import normalized_prompt,convert,SCHEMA as BRIDGE_SCHEMA
from training.artifacts import sha256_file
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

START=56
COUNT=4
END=START+COUNT
ARMS=("base_pixels","base_normalized","h106_pixels","h106_normalized")
SCHEMA="hala.h112.original-source-fresh-native-qwen-relative-coordinate-ab.v1"

def frozen(rows):
    if len(rows)!=END:
        raise ValueError("original pinned public EVAL source 60-prefix required")
    chosen=rows[START:END]
    if len(chosen)!=COUNT or len({x["id"] for x in chosen})!=COUNT:
        raise ValueError("H112 four unseen original upstream source scenes")
    return chosen

def inputs_for(scene_list,relative:bool):
    out=[]
    for scene in scene_list:
        item=dict(scene)
        if relative:
            item["prompt"]=normalized_prompt(scene)
        out.append(item)
    return out

def run(*,repo,work,output,attached_inputs=Path("/kaggle/input")):
    require_owner_private_runtime(work=work,output=output)
    repo,work,output=map(Path,(repo,work,output))
    adapter,qualified=admit_h106_checkpoint(attached_inputs,repo)
    work.mkdir(parents=True,exist_ok=False)
    source={}
    for domain in DOMAINS:
        folder=work/domain
        fetch_domain(domain,END,folder)
        original=frozen(apply_frame(load_scenes(folder/"scenes.json"),folder))
        source[domain]=original
    import torch,transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from peft import PeftModel
    if not torch.cuda.is_available() or transformers.__version__!="5.16.1":
        raise ValueError("H112 free T4 and source-pinned Qwen vision processor needed")
    os.environ.update(HF_HOME=HF_HOME,HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    metrics={};proofs={}
    started=time.monotonic()
    fields=("scenes","action_valid","start_hit","end_hit","task_success",
            "task_success_rate","degenerate_start_equals_end")
    for arm in ARMS:
        core=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION),
            torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
        model=(core if arm.startswith("base_") else PeftModel.from_pretrained(
            core,adapter,is_trainable=False,trust_remote_code=False).eval())
        relative=arm.endswith("_normalized")
        metrics[arm]={};proofs[arm]={}
        for domain,original in source.items():
            presented=inputs_for(original,relative)
            raw_outputs,attempts=_infer(model,processor,presented,torch)
            # Relative parser is exact. No repaired JSON or inferred tools.
            interpreted={
                s["id"]:(convert(raw_outputs[s["id"]],s["image_width"],s["image_height"])
                         if relative else raw_outputs[s["id"]])
                for s in original}
            scored=score(original,interpreted)
            expected=[(s["id"],s["image_sha256"],
                       hashlib.sha256(s["prompt"].encode("utf8")).hexdigest())
                      for s in presented]
            if expected!=[(a["id"],a["image_sha256"],a["prompt_sha256"]) for a in attempts]:
                raise ValueError("actual real model original screenshot/prompt pair drift")
            metrics[arm][domain]={k:scored[k] for k in fields}
            proofs[arm][domain]=attempts
            print("H112_REAL_NORMALIZED_OR_PIXEL",arm,domain,len(original),flush=True)
        del model,core
        gc.collect();torch.cuda.empty_cache()
    count={arm:sum(metrics[arm][d]["scenes"] for d in DOMAINS) for arm in ARMS}
    if count!={arm:16 for arm in ARMS}:
        raise ValueError("H112 four original paired arms not 16 scenes")
    successes={arm:sum(metrics[arm][d]["task_success"] for d in DOMAINS) for arm in ARMS}
    valid={arm:sum(metrics[arm][d]["action_valid"] for d in DOMAINS) for arm in ARMS}
    output.mkdir(parents=True,exist_ok=False)
    raw=output/"private-raw-neural-attempts.json"
    raw.write_bytes(canonical(proofs))
    report={
        "schema":SCHEMA,"status":"H112_REAL_SOURCE_DISJOINT_QWEN_RELATIVE_AB_COMPLETE",
        "source_repo":REPO,"source_commit":REV,"source_split":"eval",
        "source_selection":"original upstream eval 56..59 in each of four domains",
        "original_paired_source_count":16,"sample_counts":count,
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "relative_coordinate_bridge_schema":BRIDGE_SCHEMA,
        "same_source_media_all_arms":True,
        "native_relative_outputs_mapped_before_original_strict_scorer":True,
        "original_H61_strict_scorer_unchanged":True,
        "native_strict_action_successes":successes,
        "native_strict_action_json_valid":valid,
        "per_domain_scores":metrics,
        "h106_original_qualified_checkpoint":qualified,
        "raw_model_attempt_sha256":sha256_file(raw),
        "no_training_or_weight_change":True,
        "eval_labels_not_in_prompts":True,
        "hidden_test_used":False,"paid_resource_used":False,
        "model_improvement_claimed":False,
        "private_raw_data_published":False,
        "commercial_use":False,"elapsed_seconds":round(time.monotonic()-started,2)}
    (output/"h112-private-native-relative-vs-pixels-summary.json").write_bytes(canonical(report))
    return report
