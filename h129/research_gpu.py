"""H129 private actual ShowUI 2B no-training: first guess then marker feedback.

Strict source-disjoint original TRAIN16 paired 64 real pretrained model calls.
No feedback image, prompt, or crop receives correct bbox/source target.
"""
from pathlib import Path
import json
import os
import time
from h61.curriculum import load_export
from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h96.research_probe import require_private_vision_research
from h125.model_contracts import MODELS,SHOWUI_SYSTEM,parse_point,prompt_for
from h129.source import propose,TOTAL,PREVIOUS
from h129.feedback import draw_marker,feedback_prompt,score
from training.artifacts import artifact_path,sha256_file

ARM="showui_2b"

def _infer(model,processor,torch,image,intent,role,feedback=False):
    if role not in ("start","end"):
        raise ValueError("H129 role must be exact START/END")
    text=(feedback_prompt(intent,role) if feedback else
          prompt_for(ARM,intent,role.upper()))
    if not text.startswith(SHOWUI_SYSTEM+"\n"):
        raise ValueError("H129 ShowUI native system prompt must remain pinned")
    content=[{"type":"text","text":SHOWUI_SYSTEM},{"type":"image"},
             {"type":"text","text":text[len(SHOWUI_SYSTEM)+1:]}]
    prompt=processor.apply_chat_template(
        [{"role":"user","content":content}],
        tokenize=False,add_generation_prompt=True)
    tensors=processor(text=[prompt],images=[image],return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        prediction=model.generate(**tensors,max_new_tokens=64,
                                  do_sample=False,max_time=45)
    torch.cuda.synchronize(0)
    output=processor.batch_decode(
        prediction[:,tensors.input_ids.shape[1]:],
        skip_special_tokens=True)[0]
    if type(output) is not str or len(output)>128:
        raise ValueError("H129 actual native ShowUI response invalid/oversized")
    del tensors,prediction
    return output

def run(repo,scratch,output):
    from PIL import Image
    import torch,transformers
    from transformers import AutoProcessor,Qwen2VLForConditionalGeneration

    repo,scratch,output=map(Path,(repo,scratch,output))
    require_private_vision_research(scratch=scratch,output=output)
    if (scratch.exists() or output.exists() or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())) or
        transformers.__version__!="5.16.1"):
        raise ValueError("H129 fresh owner-private noncommercial FREE Kaggle T4 required")
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      scratch/"H71",H71_CODE_SHA,rederive=True)
    if stage.get("status")!="RESTORED_TRAIN_STAGE" or stage.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("H129 original H71 TRAIN archive content changed")
    root=scratch/"H71/export"
    manifest=json.loads((root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    source_rows=load_export(root/"h61_drag_curriculum.jsonl",manifest)
    schedule=propose(source_rows,root)
    if (len(schedule.cases)!=TOTAL or
        schedule.summary["all_previous_ID_and_SHAs_excluded"]!=PREVIOUS):
        raise ValueError("H129 source original prior ID/media-SHA exclusions incomplete")
    for case in schedule.cases:
        media=artifact_path(root,case["image_path"])
        if not media.is_file() or media.is_symlink() or sha256_file(media)!=case["image_sha256"]:
            raise ValueError("H129 original TRAIN screenshot genuine byte SHA mismatch")
        with Image.open(media) as image:
            if (image.size!=(case["image_width"],case["image_height"])
                or image.format not in ("JPEG","PNG")):
                raise ValueError("H129 original image dimensions/media invalid")
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
        raise ValueError("H129 pinned frozen pretrained ShowUI model architecture drift")
    neural=[];scoring=[];calls=0
    start=time.monotonic()
    for case in schedule.cases:
        with Image.open(artifact_path(root,case["image_path"])) as im:
            image=im.convert("RGB")
        result={"id":case["id"],"image_sha256":case["image_sha256"]}
        public_score_record={"id":case["id"],"image_sha256":case["image_sha256"]}
        for role in ("start","end"):
            first=_infer(model,processor,torch,image,case["intent"],role)
            calls+=1
            direct=parse_point(ARM,first,image.width,image.height)
            if direct is None:
                raise ValueError("H129 native-valid first model point required")
            marked=draw_marker(image,direct)
            corrected=_infer(model,processor,torch,marked,case["intent"],role,
                             feedback=True)
            calls+=1
            feedback_point=parse_point(ARM,corrected,image.width,image.height)
            if feedback_point is None:
                raise ValueError("H129 native-valid model-feedback point required")
            result["direct_raw_"+role]=first
            result["feedback_raw_"+role]=corrected
            result["direct_"+role]=direct
            result["feedback_"+role]=feedback_point
            public_score_record["direct_"+role]=direct
            public_score_record["feedback_"+role]=feedback_point
        neural.append(result);scoring.append(public_score_record)
    if calls!=64 or len(scoring)!=16:
        raise ValueError("H129 full paired 64 real neural calls not complete")
    report=score(schedule.cases,scoring)
    output.mkdir(parents=True,exist_ok=False)
    raw_file=output/"h129-private-original16x4-neural.json"
    raw_file.write_text(json.dumps(neural,indent=2,sort_keys=True)+"\n","utf8")
    report.update(status="H129_ACTUAL_PRIVATE_PRETRAINED_CURSOR_MARKER16_SCORED",
                  model_id=cfg.model_id,model_revision=cfg.revision_sha,
                  model_license=cfg.license,
                  real_neural_calls=64,
                  private_raw_sha256=sha256_file(raw_file),
                  source_cases_sha256=schedule.summary["cases_sha256"],
                  source_schedule_sha256=schedule.summary["schedule_sha256"],
                  original_train_images_sha_verified=16,
                  old_original_source_ID_and_sha_excluded=PREVIOUS,
                  elapsed_seconds=round(time.monotonic()-start,2),
                  paid_compute_used=False,
                  public_original_images_or_model_weights=False)
    (output/"h129-private-paired-metrics.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n","utf8")
    return report
