"""H110 source-disjoint H106 one-pass vs two-crop local pixel refinement.

Pilot is NOT proof of model gain or a faithful GUI-Spotlight implementation.
Original public EVAL indices 44..47/family (16 total), untouched native H61
scorer. Model coarse predictions ONLY choose crops. True boxes only evaluated
after model response.
"""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path

from h61.official_eval import (DOMAINS,REPO,REV,canonical,fetch_domain,
    load_scenes,apply_frame,parse_action,score)
from h93.eval import _infer
from h100.eval import require_owner_private_runtime
from h107.eval import admit_h106_checkpoint
from h85.model_cache import HF_HOME,pinned_cache_kwargs
from training.artifacts import sha256_file
from training.scripts.baseline import MODEL_ID,MODEL_REVISION
from h110.refine import make_focus,merge_refinements,ZOOM_SIZE

START=44
COUNT=4
END=START+COUNT
ARMS=("h106_original","h106_two_crops_512")
SCHEMA="hala.h110.new-source-16-original-versus-model-guided-focus.v1"

def frozen(scenes):
    if len(scenes)!=END:
        raise ValueError("source original EVAL 48-prefix required")
    selected=scenes[START:END]
    if len(selected)!=COUNT or len({x["id"] for x in selected})!=COUNT:
        raise ValueError("exactly four untouched real source screenshots")
    return selected

def center_in_roi(box,frame):
    point=[(box[0]+box[2])//2,(box[1]+box[3])//2]
    return (frame.left<=point[0]<frame.left+frame.width
            and frame.top<=point[1]<frame.top+frame.height)

def run(*,repo,work,output,attached_inputs=Path("/kaggle/input")):
    require_owner_private_runtime(work=work,output=output)
    repo,work,output=map(Path,(repo,work,output))
    adapter,private_training_receipt=admit_h106_checkpoint(attached_inputs,repo)
    work.mkdir(parents=True,exist_ok=False)
    original={}
    for domain in DOMAINS:
        folder=work/domain
        fetch_domain(domain,END,folder)
        rows=frozen(apply_frame(load_scenes(folder/"scenes.json"),folder))
        original[domain]=rows
    import torch,transformers
    from peft import PeftModel
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available() or transformers.__version__!="5.16.1":
        raise ValueError("pinned model and free Kaggle T4 required")
    os.environ.update(HF_HOME=HF_HOME,HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    base=Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION),
        torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
    model=PeftModel.from_pretrained(
        base,adapter,is_trainable=False,trust_remote_code=False).eval()
    results={arm:{} for arm in ARMS}
    raw_proofs={}
    diagnostics={"source_cases":0,"coarse_native_json_valid":0,
                 "refinement_calls":0,"both_model_guided_rois_contained_targets":0,
                 "both_refiners_native_json_valid":0}
    started=time.monotonic()
    fields=("scenes","action_valid","start_hit","end_hit",
            "task_success","task_success_rate","degenerate_start_equals_end")
    for domain,scenes in original.items():
        coarse_outputs,initial_raw=_infer(model,processor,scenes,torch)
        final_outputs=dict(coarse_outputs)
        evidence=[]
        for idx,scene in enumerate(scenes):
            raw=coarse_outputs[scene["id"]]
            if hashlib.sha256(raw.encode("utf8")).hexdigest()!=initial_raw[idx]["output_sha256"]:
                raise ValueError("initial model raw-output SHA drift")
            proof={"source":initial_raw[idx],"refinement_attempted":False}
            diagnostics["source_cases"]+=1
            candidate=parse_action(raw,scene["image_width"],scene["image_height"])
            if candidate is not None:
                diagnostics["coarse_native_json_valid"]+=1
                folder=work/(domain+"-focused")
                focus_a=make_focus(scene,candidate,"start",folder/f"{idx:02d}-start.png")
                focus_b=make_focus(scene,candidate,"end",folder/f"{idx:02d}-end.png")
                raw_a,p_a=_infer(model,processor,[focus_a.scene],torch)
                raw_b,p_b=_infer(model,processor,[focus_b.scene],torch)
                diagnostics["refinement_calls"]+=2
                text_a=raw_a[scene["id"]];text_b=raw_b[scene["id"]]
                full_action,used=merge_refinements(
                    scene,raw,focus_a,text_a,focus_b,text_b)
                final_outputs[scene["id"]]=full_action
                diagnostics["both_refiners_native_json_valid"]+=int(all(used.values()))
                diagnostics["both_model_guided_rois_contained_targets"]+=int(
                    center_in_roi(scene["task"]["start_bbox"],focus_a.frame)
                    and center_in_roi(scene["task"]["end_bbox"],focus_b.frame))
                proof.update(refinement_attempted=True,
                             focused_start=p_a[0],focused_end=p_b[0],
                             final_raw=full_action,
                             final_output_sha256=hashlib.sha256(full_action.encode("utf8")).hexdigest(),
                             changed_endpoint_flags=used)
            evidence.append(proof)
        raw_proofs[domain]=evidence
        reference=score(scenes,coarse_outputs)
        refined=score(scenes,final_outputs)
        results["h106_original"][domain]={k:reference[k] for k in fields}
        results["h106_two_crops_512"][domain]={k:refined[k] for k in fields}
        print("H110_ORIGINAL_NEW_FOCUS_SOURCE",domain,len(scenes),flush=True)
    if diagnostics["source_cases"]!=16:
        raise ValueError("H110 must cover exact 16 new EVAL scenes")
    output.mkdir(parents=True,exist_ok=False)
    outfile=output/"private-raw-three-pass-attempts.json"
    outfile.write_bytes(canonical(raw_proofs))
    strict={arm:sum(results[arm][d]["task_success"] for d in DOMAINS) for arm in ARMS}
    native={arm:sum(results[arm][d]["action_valid"] for d in DOMAINS) for arm in ARMS}
    report={"schema":SCHEMA,"status":"H110_REAL_PAIRED_FRESH_16_MODEL_CROPS_COMPLETE",
        "source_commit":REV,"source_repo":REPO,"source_split":"eval",
        "source_selection":"original pinned public EVAL indices 44..47 per 4 domains",
        "source_cases":16,"arms":list(ARMS),"refinement_crop_size_original_pixels":ZOOM_SIZE,
        "refinement_image_zoom":2,
        "model_generated_crop_centers_only":True,
        "original_source_boxes_not_used_during_inference":True,
        "original_H61_native_strict_scorer":True,
        "strict_full_drag_successes":strict,
        "native_valid_action_counts":native,
        "other_diagnostics":diagnostics,
        "per_domain_scores":results,
        "original_H106_checkpoint":private_training_receipt,
        "raw_neural_sha256":sha256_file(outfile),
        "model_weights_changed":False,"new_training_executed":False,
        "model_generalization_established":False,
        "hidden_test_used":False,"paid_cloud_used":False,
        "private_model_weights_published":False,
        "commercial_use":False,
        "elapsed_seconds":round(time.monotonic()-started,2)}
    (output/"h110-private-fresh-crop512-ab-summary.json").write_bytes(canonical(report))
    return report
