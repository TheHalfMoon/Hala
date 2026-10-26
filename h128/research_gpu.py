"""H128 private free-T4 paired ShowUI direct-vs-model-anchored-zoom pilot.

Exactly 16 NEW source-disjoint original TRAIN cases; 64 actual pretrained
model calls, no fine tuning and no oracle bounding boxes in inference.
"""
from pathlib import Path
import hashlib
import json
import os
import time

from h61.curriculum import load_export
from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h96.research_probe import require_private_vision_research
from h125.model_contracts import MODELS,SHOWUI_SYSTEM,parse_point,prompt_for
from h128.source import propose,TOTAL
from h128.zoom import anchored_crop,native_to_original,score
from training.artifacts import artifact_path,sha256_file

ARM="showui_2b"

def _infer(model,processor,torch,image,intent,role):
    if role not in ("start","end"):
        raise ValueError("H128 only START/END input")
    prompt=prompt_for(ARM,intent,role.upper())
    if not prompt.startswith(SHOWUI_SYSTEM+"\n"):
        raise ValueError("H128 cannot drift from official ShowUI role prompt contract")
    msg=[{"role":"user","content":[
        {"type":"text","text":SHOWUI_SYSTEM},
        {"type":"image"},
        {"type":"text","text":prompt[len(SHOWUI_SYSTEM)+1:]}]}]
    rendered=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    batch=processor(text=[rendered],images=[image],return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        generated=model.generate(**batch,max_new_tokens=64,
                                 do_sample=False,max_time=45)
    torch.cuda.synchronize(0)
    raw=processor.batch_decode(generated[:,batch.input_ids.shape[1]:],
                                skip_special_tokens=True)[0]
    if type(raw) is not str or len(raw)>128:
        raise ValueError("H128 invalid real model response")
    del batch,generated
    return raw

def run(repo,scratch,output):
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration,AutoProcessor
    import torch,transformers

    repo,scratch,output=map(Path,(repo,scratch,output))
    require_private_vision_research(scratch=scratch,output=output)
    if (scratch.exists() or output.exists() or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) or
        transformers.__version__!="5.16.1"):
        raise ValueError("H128 fresh owner-private free T4 with frozen package runtime required")
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      scratch/"H71",H71_CODE_SHA,rederive=True)
    if stage.get("status")!="RESTORED_TRAIN_STAGE" or stage.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("H128 owner private H71 true original TRAIN archive mismatch")
    root=scratch/"H71/export"
    manifest=json.loads((root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    rows=load_export(root/"h61_drag_curriculum.jsonl",manifest)
    schedule=propose(rows,root)
    if len(schedule.cases)!=TOTAL or schedule.summary["all_previous_ID_and_SHAs_excluded"]!=2688:
        raise ValueError("H128 preregistered entirely fresh original pilot source missing")
    # Independently validate actual new screenshot byte SHA before loading model.
    for c in schedule.cases:
        media=artifact_path(root,c["image_path"])
        if not media.is_file() or media.is_symlink() or sha256_file(media)!=c["image_sha256"]:
            raise ValueError("H128 private actual screenshot changed / missing")
        with Image.open(media) as im:
            if im.size!=(c["image_width"],c["image_height"]) or im.format not in ("PNG","JPEG"):
                raise ValueError("H128 actual original image format/dimension mismatch")
    cfg=MODELS[ARM]
    torch.manual_seed(42)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    processor=AutoProcessor.from_pretrained(
        cfg.model_id,revision=cfg.revision_sha,
        min_pixels=256*28*28,max_pixels=1280*28*28)
    model=Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,revision=cfg.revision_sha,
        torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
    if model.config.model_type!="qwen2_vl":
        raise ValueError("H128 expected pinned pretrained GUI 2B only")
    calls=0; rows_out=[]; score_out=[]
    started=time.monotonic()
    for case in schedule.cases:
        with Image.open(artifact_path(root,case["image_path"])) as im:
            image=im.convert("RGB")
        result={"id":case["id"],"image_sha256":case["image_sha256"]}
        private={"id":case["id"],"image_sha256":case["image_sha256"]}
        for role in ("start","end"):
            raw=_infer(model,processor,torch,image,case["intent"],role);calls+=1
            direct=parse_point(ARM,raw,image.width,image.height)
            if direct is None:
                raise ValueError("H128 native-valid baseline required for paired crop anchor")
            crop=anchored_crop(image.width,image.height,direct,ratio=0.5)
            cropped=image.crop((crop.left,crop.top,crop.right,crop.bottom))
            raw_refine=_infer(model,processor,torch,cropped,case["intent"],role)
            calls+=1
            local=parse_point(ARM,raw_refine,crop.width,crop.height)
            zoom=native_to_original(crop,local,image.width,image.height) if local else None
            result["direct_"+role]=direct
            result["zoom_"+role]=zoom
            result["crop_"+role]=crop
            private["direct_raw_"+role]=raw
            private["zoom_raw_"+role]=raw_refine
            private["crop_"+role]=[crop.left,crop.top,crop.right,crop.bottom]
            private["direct_"+role]=direct
            private["zoom_"+role]=zoom
        rows_out.append(private);score_out.append(result)
    metrics=score(schedule.cases,score_out)
    if len(rows_out)!=16 or calls!=64:
        raise ValueError("H128 real paired pretrained model 64 calls incomplete")
    output.mkdir(parents=True,exist_ok=False)
    raw_file=output/"h128-owner-private-16x4-neural.json"
    raw_file.write_text(json.dumps(rows_out,sort_keys=True,indent=2),encoding="utf8")
    report={**metrics,
        "status":"H128_REAL_PRIVATE_EXPLORATORY_PAIRED_ZOOM16_SCORED",
        "schedule_sha256":schedule.summary["schedule_sha256"],
        "source_cases_sha256":schedule.summary["cases_sha256"],
        "original_new_images_sha_verified":16,
        "original_prior_excluded":2688,
        "model_id":cfg.model_id,"model_revision":cfg.revision_sha,
        "model_license":cfg.license,
        "real_neural_calls":calls,
        "private_raw_sha256":sha256_file(raw_file),
        "elapsed_seconds":round(time.monotonic()-started,2),
        "model_skill_qualified":False,"optimizer_steps":0,
        "original_EVAL_TEST_or_prior_heldout_used":False,
        "public_original_images_or_model_weights":False,
        "paid_compute_used":False}
    (output/"h128-owner-private-metrics.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return report
