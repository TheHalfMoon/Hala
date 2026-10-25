"""H118 frozen new 64 original GUI scenes: H106 vs H114 vs H117.

H117 trained model is deliberately rejected until exact owner Windows SHA
byte attestation is available. The original H61 strict scorer is never edited.
"""
import gc,hashlib,json,os,time
from pathlib import Path
from h61.official_eval import (DOMAINS,REPO,REV,canonical,fetch_domain,
                               load_scenes,apply_frame,score,strict_json)
from h85.model_cache import HF_HOME,pinned_cache_kwargs
from h93.eval import _infer
from h100.eval import require_owner_private_runtime
from h107.eval import admit_h106_checkpoint
from h115.eval import admit_h114_checkpoint
from h115.compose import role_input,compose_strict_original_drag,parse_point
from h116.delta import prompt_end_delta,compose_model_drag_from_delta
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

START=76
PER_DOMAIN=16
END=START+PER_DOMAIN
ARMS=("h106_original_drag","h114_independent_points","h117_conditioned_signed_end")
SCHEMA="hala.h118.private-new64-absolute-vs-conditioned-signed-end-GUI.v1"
H117_ADAPTER_SHA="32222c4c75f2046cf5914caf1c9554c6bd15bb02c33ac4c4075ea9f1b8c577cf"
H117_CHECKPOINT_SHA="cd7c84936b623937fc2fa32554368fa28a9f0f1878c74b4019e8929ebb9fbcb0"

def frozen(rows):
    if len(rows)!=END:
        raise ValueError("H118 original pinned public EVAL 92 prefix required")
    chosen=rows[START:END]
    if len(chosen)!=PER_DOMAIN or len({x["id"] for x in chosen})!=PER_DOMAIN:
        raise ValueError("H118 must have sixteen fresh source images per domain")
    return chosen

def admit_h117_checkpoint(attached=Path("/kaggle/input")):
    if not (H117_ADAPTER_SHA and H117_CHECKPOINT_SHA):
        raise ValueError("H117 owner byte-verified new optimizer/adapter not yet admitted")
    candidates=[p.parent for p in Path(attached).rglob("latest.json")
                if "hala-h117-private-conditioned-delta256-checkpoint" in p.as_posix()
                and "hala-h117-private-conditioned-delta256-t4" in p.as_posix()
                and p.is_file() and not p.is_symlink()]
    if len(candidates)!=1:
        raise ValueError("one qualified H117 private checkpoint required")
    root=candidates[0]
    version,state=load_checkpoint(root,None)
    receipt=root.parent/"hala-h117-private-conditioned-delta-training-receipt.json"
    if not receipt.is_file() or receipt.is_symlink():
        raise ValueError("H117 signed END training receipt missing")
    info=json.loads(receipt.read_text("utf8"))
    a=sha256_file(version/"adapter/adapter_model.safetensors")
    b=sha256_file(version/"checkpoint.json")
    if (info.get("status")!="H117_REAL_256_ORIGINAL_START_CONDITIONED_DELTA_UPDATES_VERIFIED"
            or info.get("actual_optimizer_steps")!=256 or state.get("step")!=256
            or state.get("max_steps")!=256
            or info.get("original_h116_conditioned_signed_delta_sha256")!=
                "174e16634b317ee23a4b6661568f51721a0ca72b6cdabd83d9f107bf361b3d7c"
            or a!=H117_ADAPTER_SHA or a!=info.get("adapter_sha256")
            or b!=H117_CHECKPOINT_SHA or b!=info.get("checkpoint_sha256")
            or state.get("runtime",{}).get("h106_joint_lora_verified") is not True):
        raise ValueError("H117 actual original signed-delta model binary identity mismatch")
    return version/"adapter",{"adapter_sha256":a,"checkpoint_sha256":b}

def strict_start_anchor(raw,width,height):
    if parse_point(raw,"start",width,height) is None:
        return None
    try:
        point=strict_json(raw)["point"]
    except (TypeError,ValueError):
        return None
    if type(point) is not list or len(point)!=2 or any(type(v) is not int for v in point):
        return None
    return point

def input_proof(scenes,evidence):
    expected=[(s["id"],s["image_sha256"],
               hashlib.sha256(s["prompt"].encode("utf8")).hexdigest()) for s in scenes]
    returned=[(x["id"],x["image_sha256"],x["prompt_sha256"]) for x in evidence]
    if expected!=returned:
        raise ValueError("H118 actual original screenshot/prompt SHA differs from real inference")

def run(*,repo,work,output,attached_inputs=Path("/kaggle/input")):
    require_owner_private_runtime(work=work,output=output)
    repo,work,output=map(Path,(repo,work,output))
    original_h106=admit_h106_checkpoint(attached_inputs,repo)[0]
    h114=admit_h114_checkpoint(attached_inputs)[0]
    h117,new_receipt=admit_h117_checkpoint(attached_inputs)
    work.mkdir(parents=True,exist_ok=False)
    sources={}
    for domain in DOMAINS:
        dest=work/domain
        fetch_domain(domain,END,dest)
        sources[domain]=frozen(apply_frame(load_scenes(dest/"scenes.json"),dest))
    import torch,transformers
    from peft import PeftModel
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available() or transformers.__version__!="5.16.1":
        raise ValueError("owner free T4 original Qwen processor not available")
    os.environ.update(HF_HOME=HF_HOME,HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    scores={};raw={};calls={arm:0 for arm in ARMS}
    metrics=("scenes","action_valid","start_hit","end_hit","task_success",
             "task_success_rate","degenerate_start_equals_end")
    started=time.monotonic()
    for arm in ARMS:
        core=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION),
            torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
        adapter={"h106_original_drag":original_h106,
                 "h114_independent_points":h114,
                 "h117_conditioned_signed_end":h117}[arm]
        model=PeftModel.from_pretrained(
            core,adapter,is_trainable=False,trust_remote_code=False).eval()
        scores[arm]={};raw[arm]={}
        for domain,scenes in sources.items():
            if arm=="h106_original_drag":
                actions,attempts=_infer(model,processor,scenes,torch)
                input_proof(scenes,attempts)
                evidence={"full_drag":attempts}
                calls[arm]+=len(scenes)
            else:
                starts=[role_input(s,"start") for s in scenes]
                output_start,first=_infer(model,processor,starts,torch)
                input_proof(starts,first)
                calls[arm]+=len(starts)
                if arm=="h114_independent_points":
                    ends=[role_input(s,"end") for s in scenes]
                    output_end,last=_infer(model,processor,ends,torch)
                    input_proof(ends,last)
                    calls[arm]+=len(ends)
                    actions={s["id"]:compose_strict_original_drag(
                        s,output_start[s["id"]],output_end[s["id"]])
                        for s in scenes}
                    evidence={"start":first,"end":last}
                else:
                    conditioned=[]
                    for s in scenes:
                        point=strict_start_anchor(
                            output_start[s["id"]],s["image_width"],s["image_height"])
                        if point is not None:
                            row=dict(s)
                            row["prompt"]=prompt_end_delta(s["task"]["intent"],point)
                            conditioned.append(row)
                    if conditioned:
                        output_delta,last=_infer(model,processor,conditioned,torch)
                        input_proof(conditioned,last)
                    else:
                        output_delta,last={},[]
                    calls[arm]+=len(last)
                    actions={s["id"]:compose_model_drag_from_delta(
                        s,output_start[s["id"]],output_delta[s["id"]])
                        if s["id"] in output_delta else None for s in scenes}
                    evidence={"start":first,"delta":last}
            strict=score(scenes,actions)
            scores[arm][domain]={k:strict[k] for k in metrics}
            raw[arm][domain]=evidence
            print("H118_REAL_UNSEEN_SOURCE_STRICT_GUI",arm,domain,len(scenes),flush=True)
        del model,core
        gc.collect()
        torch.cuda.empty_cache()
    counts={arm:sum(scores[arm][d]["scenes"] for d in DOMAINS) for arm in ARMS}
    if counts!={a:64 for a in ARMS}:
        raise ValueError("H118 3-arm original 64-source evaluation incomplete")
    output.mkdir(parents=True,exist_ok=False)
    evidence=output/"private-real-neural-attempts.json"
    evidence.write_bytes(canonical(raw))
    result={
        "schema":SCHEMA,"status":"H118_REAL_SOURCE_DISJOINT_64_THREE_ARM_STRICT_GUI_COMPLETE",
        "source_repo":REPO,"source_commit":REV,"source_split":"eval",
        "original_public_eval_indices":"76..91 each four domains",
        "source_scenes":64,"sample_counts":counts,
        "model_arms":list(ARMS),
        "strict_full_drag_successes":{a:sum(scores[a][d]["task_success"] for d in DOMAINS)
                                      for a in ARMS},
        "native_valid_final_actions":{a:sum(scores[a][d]["action_valid"] for d in DOMAINS)
                                      for a in ARMS},
        "per_domain_scores":scores,"model_inference_calls":calls,
        "frozen_h117_adapter":new_receipt,
        "unchanged_original_H61_strict_scorer":True,
        "eval_target_boxes_not_used_by_models":True,
        "hidden_TEST_used":False,"paid_compute_used":False,
        "training_performed_in_evaluation":False,
        "general_GUI_skill_qualified":False,
        "private_neural_attempts_sha256":sha256_file(evidence),
        "elapsed_seconds":round(time.monotonic()-started,2)}
    (output/"h118-private-three-arm-original-strict-summary.json").write_bytes(canonical(result))
    return result
