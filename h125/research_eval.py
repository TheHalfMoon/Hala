"""H125 real frozen pretrained GUI 2B *inference only* on new source64.

No fine-tuning, optimizer, H123/H124 previous heldout, public EVAL/TEST,
paid cloud, deployment or weight publication. One arm per private T4.
"""
import hashlib,json,os,time
from pathlib import Path
from h61.curriculum import load_export as load_h71
from h61.official_eval import canonical
from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h96.research_probe import require_private_vision_research
from h125.source import derive,TOTAL
from h125.eval_plan import get_plan,score,SOURCE_SHA,SCHEDULE_SHA
from h125.heldout_source import admit_original_train64
from h125.model_contracts import MODELS,SHOWUI_SYSTEM
from training.artifacts import sha256_file

SCHEMA="hala.h125.real-original-TRAIN64-zero-shot-specialized2b-v1"
RAW="h125-owner-private-new64-neural.json"
CASES="h125-owner-private-original64-case.json"
SUMMARY="h125-owner-private-native-2b-summary.json"

def _inference(model,processor,cases,torch,arm):
    from PIL import Image
    results=[]
    for item in cases:
        with Image.open(item["image_path"]) as inp:
            image=inp.convert("RGB")
        result={"id":item["id"],"arm":arm,
                "image_sha256":item["image_sha256"]}
        for role in ("start","end"):
            prompt=item["prompt_"+role]
            if arm=="showui_2b":
                if not prompt.startswith(SHOWUI_SYSTEM+"\n"):
                    raise ValueError("H125 official ShowUI system/task prompt identity drift")
                task=prompt[len(SHOWUI_SYSTEM)+1:]
                content=[{"type":"text","text":SHOWUI_SYSTEM},
                         {"type":"image"},
                         {"type":"text","text":task}]
            else:
                content=[{"type":"image"},{"type":"text","text":prompt}]
            msg=[{"role":"user","content":content}]
            rendered=processor.apply_chat_template(
                msg,tokenize=False,add_generation_prompt=True)
            batch=processor(text=[rendered],images=[image],
                            return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                generated=model.generate(**batch,max_new_tokens=64,
                                         do_sample=False,max_time=45)
            torch.cuda.synchronize(0)
            raw=processor.batch_decode(
                generated[:,batch.input_ids.shape[1]:],
                skip_special_tokens=True)[0]
            if type(raw) is not str or len(raw)>128:
                raise ValueError("H125 genuine GUI model output unsafe/oversize")
            result["prompt_"+role+"_sha256"]=hashlib.sha256(
                prompt.encode("utf8")).hexdigest()
            result["raw_"+role]=raw
            result["raw_"+role+"_sha256"]=hashlib.sha256(
                raw.encode("utf8")).hexdigest()
            del batch,generated
        results.append(result)
    return results

def run(repo,work,output,arm):
    repo,work,output=map(Path,(repo,work,output))
    if arm not in MODELS:
        raise ValueError("H125 only preregistered pretrained GUI 2B arms")
    require_private_vision_research(scratch=work,output=output)
    if output.exists() or work.exists():
        raise ValueError("H125 private source/response experiment create-only")
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      work/"H71",H71_CODE_SHA,rederive=True)
    if stage.get("status")!="RESTORED_TRAIN_STAGE" or stage.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("H125 true original source TRAIN SHA not qualified")
    root=work/"H71/export"
    manifest=json.loads((root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    rows=load_h71(root/"h61_drag_curriculum.jsonl",manifest)
    plan=derive(rows,root)
    get_plan(plan)
    if (plan.summary.get("private_selected_source_sha256")!=SOURCE_SHA
            or plan.summary.get("schedule_sha256")!=SCHEDULE_SHA
            or plan.summary.get("prior_original_screenshots_excluded")!=2560):
        raise ValueError("H125 original unseen image selection differs from frozen source")
    cases=admit_original_train64(rows,root,plan,arm)
    if len(cases)!=TOTAL:
        raise ValueError("H125 exactly 64 new GUI scene images expected")
    import torch,transformers
    from transformers import Qwen2VLForConditionalGeneration,AutoProcessor
    if (not torch.cuda.is_available()
            or transformers.__version__!="5.16.1"
            or not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H125 only actual free Kaggle T4 Qwen2VL")
    cfg=MODELS[arm]
    torch.manual_seed(42)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    processor=AutoProcessor.from_pretrained(
        cfg.model_id,revision=cfg.revision_sha,
        min_pixels=256*28*28,max_pixels=1280*28*28)
    model=Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,revision=cfg.revision_sha,
        torch_dtype=torch.float16,
        attn_implementation="sdpa").to("cuda:0").eval()
    if (model.config.model_type!="qwen2_vl"
            or cfg.arch!="qwen2_vl" or cfg.pretrained_gui_grounding is not True):
        raise ValueError("H125 GUI-specialized pinned public architecture mismatch")
    began=time.monotonic()
    raw=_inference(model,processor,cases,torch,arm)
    if len(raw)!=TOTAL:
        raise ValueError("H125 exactly 128 real native model point calls required")
    result=score(cases,raw,plan,arm)
    output.mkdir(parents=True,exist_ok=False)
    rawfile=output/RAW;casesfile=output/CASES
    rawfile.write_bytes(canonical(raw))
    casesfile.write_bytes(canonical(cases))
    summary={**result,"schema":SCHEMA,
       "status":"H125_REAL_PRIVATE_NEW64_PRETRAINED_GUI2B_SCORED",
       "model_id":cfg.model_id,"model_revision":cfg.revision_sha,
       "model_license":cfg.license,
       "real_neural_endpoint_calls":TOTAL*2,
       "original_new_image_byte_SHA_verified":TOTAL,
       "private_raw_sha256":sha256_file(rawfile),
       "private_original_cases_sha256":sha256_file(casesfile),
       "elapsed_seconds":round(time.monotonic()-began,2),
       "no_finetuning":True,"model_train_steps":0,
       "original_EVAL_or_TEST_used":False,
       "prior_H123_H124_heldout_reused":False,
       "public_weights_or_original_images":False,
       "paid_compute_used":False,
       "externally_qualified_GUI_skill":False}
    (output/SUMMARY).write_bytes(canonical(summary))
    return summary
