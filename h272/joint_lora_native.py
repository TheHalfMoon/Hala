"""H272 — private FREE T4 joint visual-language LoRA with native target-hit evaluation.

Trains 8 actual optimizer steps on 8 genuinely rendered Stockroom frames.
Pretrained/LoRA paired true browser state is checked on 2 distinct Arabic
scenes, strictly source-family research ONLY (not official TRAIN/DEV/OOD).
Never save private PNG/gold/model text/weights to Kaggle outputs or GitHub.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import math
import os
import shutil
import subprocess
import sys

SOURCE_REPO = "https://github.com/TheHalfMoon/Hala.git"
SOURCE_BRANCH = "ps/h271-vision-lora-private-native-diagnostic"
SOURCE_SHA = "3e53eca0991b1032ceafdb20fd28e277e8032a98"
APP_SHA = "3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
OUTPUT = "hala-h272-private-joint-native-aggregate.json"
N_STEPS = 8
COORDINATE_WEIGHT = 8.0
TASK = "Click the +1 button of the FIRST visible inventory item to add one unit."
# Predeclared source-owned GUI arrangements. Nothing references a private
# third-party source family, consumed benchmark, or previous scored OOD.
# The two Arabic heldout modes do not appear in training, even by PNG.
TRAIN_SCENES = (
    ("en_default", "en", False, ""),
    ("en_sorted", "en", True, ""),
    ("en_gauze", "en", False, "Gauze"),
    ("en_tape", "en", False, "Paper tape"),
    ("en_gloves", "en", False, "Exam gloves"),
    ("en_labels", "en", False, "Shelf labels"),
    ("ar_default", "ar", False, ""),
    ("ar_sorted", "ar", True, ""),
)
EVAL_SCENES = (
    ("ar_sorted_tape", "ar", True, "Paper tape"),
    ("ar_sorted_gloves", "ar", True, "Exam gloves"),
)


def require(ok, reason):
    if not ok:
        raise ValueError("H272 private source-bound research blocked: " + reason)


def check_free_t4(torch):
    require(sys.platform == "linux" and Path("/kaggle/working").is_dir()
            and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            and os.environ.get("HALA_H272_PRIVATE_NONCOMMERCIAL_FREE_T4")=="1",
            "authorized private Kaggle Linux only")
    require(torch.cuda.is_available() and torch.cuda.device_count() in (1,2)
            and all("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())),
            "real free T4 GPU required")
    require(shutil.disk_usage("/tmp").free >= 8*1024**3,
            "remote ephemeral model space insufficient")
    require(not (Path("/kaggle/working")/OUTPUT).exists(),
            "create-only remote report path")
    return {"free_t4_verified":True,"t4_count":torch.cuda.device_count()}


def clone_original_source(root):
    require(not root.exists(), "fresh private Git source scratch required")
    subprocess.run(["git","clone","--depth","1","--single-branch",
                    "--branch",SOURCE_BRANCH,SOURCE_REPO,str(root)],
                   check=True,capture_output=True,text=True)
    rev=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],
                   check=True,capture_output=True,text=True).stdout.strip()
    require(rev==SOURCE_SHA,"pinned source revision drift")
    dirty=subprocess.run(["git","-C",str(root),"status","--porcelain"],
                   check=True,capture_output=True,text=True).stdout.strip()
    require(not dirty,"first-party source checkout dirty")
    html=root/"h262"/"stockroom.html"
    require(html.is_file() and not html.is_symlink()
            and sha256(html.read_bytes()).hexdigest()==APP_SHA,
            "original self-authored app changed")
    return html


def original_case(browser,html,scene,*,training):
    """Render real native offline original GUI; DOM target only for TRAIN."""
    from h263.native_browser_executor import new_offline_context
    name,language,sorted_order,query=scene
    require(scene in TRAIN_SCENES+EVAL_SCENES
            and training == (scene in TRAIN_SCENES),
            "registered scene/split required")
    context=new_offline_context(browser)
    def firewall(route):
        if route.request.url.startswith(("http:","https:","ws:","wss:")):
            route.abort()
        else:
            route.continue_()
    context.route("**/*",firewall)
    page=context.new_page()
    page.goto(html.as_uri(),wait_until="load")
    if sorted_order:
        page.locator("#sort").click()
    if language=="ar":
        page.locator("#lang").click()
    if query:
        page.locator("#search").fill(query)
    require(page.locator(".item").count()>=1,
            "first-party requested scene has no visible stock")
    point=None
    if training:
        # Only TRAIN uses this original DOM source-authored oracle.
        button=page.locator("button[data-restock]").first
        box=button.bounding_box()
        require(box is not None and box["width"]>=10 and box["height"]>=10,
                "original supervised button not visible")
        point=[int(box["x"]+box["width"]/2),
               int(box["y"]+box["height"]/2)]
        require(0<=point[0]<960 and 0<=point[1]<720,
                "train oracle point exceeds screenshot")
    png=page.screenshot(animations="disabled",full_page=False)
    require(png[:8]==b"\x89PNG\r\n\x1a\n"
            and len(png)<4*1024*1024,"bad decoded original screenshot")
    before=int(page.locator(".item .count").first.inner_text())
    return context,page,png,point,before


def build_train(browser,html,root):
    from h199.native_action_sft import parse_actions,source_task_prompt
    from PIL import Image
    media=root/"source_owned_train_png"
    media.mkdir(mode=0o700)
    rows=[]
    hashes=set()
    for index,scene in enumerate(TRAIN_SCENES):
        context=None
        try:
            context,page,png,point,_=original_case(
                browser,html,scene,training=True)
            with Image.open(BytesIO(png)) as image:
                require(image.format=="PNG" and image.size==(960,720),
                        "training screenshot must be 960x720 PNG")
                image.load()
            digest=sha256(png).hexdigest()
            require(digest not in hashes,
                    "train scenes contain duplicate PNG bytes")
            hashes.add(digest)
            completion=json.dumps(
                {"actions":[{"kind":"click","point":point}]},
                separators=(",",":"))
            parse_actions(completion,training_target=True)
            path=media/(f"firstparty_train_{index}.png")
            require(not path.exists(),"private original training PNG exists")
            path.write_bytes(png)
            rows.append({
                "id":f"h272-original-joint-train-{index}",
                "image_path":path.name,
                "image_sha256":digest,
                "image_format":"PNG",
                "prompt":source_task_prompt(TASK),
                "completion":completion,
            })
        finally:
            if context is not None:
                context.close()
    require(len(rows)==N_STEPS and len(hashes)==N_STEPS,
            "8 distinct first-party training screenshots required")
    return media,rows,hashes


def eval_pair(browser,html,processor,model,grammar_filter,train_hashes):
    """Real native action on two research scenes; reference never to model."""
    from h208.preloaded_vlm_policy import PreloadedVisionPolicy
    from h264.screenshot_locked_research import screenshot_locked_turn
    from h267.free_kaggle_format_probe import ClassifyingProcessor
    from h268.free_t4_constrained_native import NativeGrammarModel
    rows=[]
    stamps=[]
    for scene in EVAL_SCENES:
        context=None
        wrapped=NativeGrammarModel(model,grammar_filter)
        classified=ClassifyingProcessor(processor)
        predictor=PreloadedVisionPolicy(
            classified,wrapped,max_new_tokens=160)
        kind=None
        dispatched=False
        clicked_first_plus=False
        model_point=None
        try:
            context,page,png,_,before=original_case(
                browser,html,scene,training=False)
            digest=sha256(png).hexdigest()
            require(digest not in train_hashes,
                    "heldout original screenshot collides with TRAIN")
            stamps.append(digest)
            from h199.native_action_sft import parse_actions
            # Capture model-chosen point in RAM solely for AFTER-the-action
            # coarse source-owned scorer; never log or persist coordinates.
            def private_policy_only(pixels,task):
                nonlocal model_point
                raw=predictor(pixels,task)
                try:
                    action=parse_actions(raw)["actions"][0]
                    if action["kind"]=="click":
                        model_point=tuple(action["point"])
                except (RuntimeError,ValueError,KeyError,TypeError):
                    model_point=None
                return raw
            try:
                output=screenshot_locked_turn(
                    page,html,html.as_uri(),TASK,private_policy_only)
                kind=output["kind"]
                dispatched=bool(output["executed"])
            except (RuntimeError,ValueError):
                # No oracle click fallback, no post-hoc JSON repair.
                pass
            after=int(page.locator(".item .count").first.inner_text())
            if dispatched and kind=="click":
                require(model_point is not None,
                        "missing model-only point after native click")
                # Independent original DOM grader only AFTER inference
                # and action; raw point and target bbox never leave RAM.
                rect=page.locator("button[data-restock]").first.bounding_box()
                if rect is not None:
                    clicked_first_plus=(
                        rect["x"]<=model_point[0]<rect["x"]+rect["width"]
                        and rect["y"]<=model_point[1]<rect["y"]+rect["height"])
        finally:
            if context is not None:
                context.close()
        rows.append({
            "scene":scene[0],
            "real_generation_complete":wrapped.completed_generations==1,
            "model_time_grammar_called":wrapped.grammar_installed_calls==1,
            "strict_native_json":classified.format_class=="STRICT_NATIVE_JSON",
            "kind":kind if kind in ("click","abstain") else None,
            "browser_action_dispatched":dispatched,
            "model_chosen_click_hits_first_plus":bool(clicked_first_plus),
            "firstparty_task_success":bool(dispatched and after==before+1),
        })
    require(len(set(stamps))==len(EVAL_SCENES),
            "evaluation scenes must be byte-distinct")
    return rows,stamps


def measured_aggregate(step_evidence,base,adapted,platform,*,language_modules):
    """Never turns a first-party score into independent OOD proof."""
    require(type(language_modules) is int and 16<=language_modules<=256,
            "verified number of language-model attention layers required")
    require(type(step_evidence) is list and len(step_evidence)==N_STEPS,
            "exact genuine 8-step optimizer evidence required")
    require(all(type(x.get("changed_tensors")) is int and x["changed_tensors"]>0
                and type(x.get("coordinate_tokens")) is int
                and x["coordinate_tokens"]>0
                and math.isfinite(x["gradient_norm"]) and x["gradient_norm"]>0
                and math.isfinite(x["loss"]) and x["loss"]>0
                for x in step_evidence),
            "nonzero finite coordinate-weighted updates required")
    require(platform.get("free_t4_verified") is True,
            "no real free GPU attestation")
    for rows in (base,adapted):
        require(type(rows) is list and len(rows)==len(EVAL_SCENES),
                "both native paired GUI evaluations required")
        require([x.get("scene") for x in rows]==
                [s[0] for s in EVAL_SCENES],
                "paired scenario order mismatch")
        for row in rows:
            require(row.get("real_generation_complete") is True
                    and row.get("model_time_grammar_called") is True
                    and type(row.get("browser_action_dispatched")) is bool
                    and type(row.get("firstparty_task_success")) is bool
                    and type(row.get("model_chosen_click_hits_first_plus")) is bool
                    and (not row["firstparty_task_success"]
                         or (row["browser_action_dispatched"]
                             and row.get("kind")=="click"
                             and row.get("strict_native_json") is True
                             and row["model_chosen_click_hits_first_plus"])),
                    "must have real native actions, not mock reward")
    b=sum(x["firstparty_task_success"] for x in base)
    a=sum(x["firstparty_task_success"] for x in adapted)
    return {
        "status":"H272_REAL_JOINT_FIRSTPARTY_RESEARCH_NOT_ADMITTED",
        "public_source_git_sha":SOURCE_SHA,
        "model_id":MODEL_ID,"model_revision":MODEL_REV,
        "hardware":platform,
        "real_optimizer_steps":N_STEPS,
        "train_distinct_source_owned_screenshots":N_STEPS,
        "source_owned_train_language_scenes":{"en":6,"ar":2},
        "coordinate_weight":COORDINATE_WEIGHT,
        "all_steps_nonzero_changed_adapters":True,
        "all_steps_finite_nonzero_gradient":True,
        "all_steps_supervised_coordinate_tokens":True,
        "adapter_scope":"joint_last_four_vision_and_language_qv",
        "exact_vision_linear_targets_verified":True,
        "exact_language_qv_targets_verified":True,
        "trainable_vision_adapter_tensors":16,
        "trainable_language_adapter_tensors":2*language_modules,
        "total_modified_adapter_tensors_across_steps":
            sum(x["changed_tensors"] for x in step_evidence),
        "paired_untrained":base,
        "paired_adapted":adapted,
        "paired_firstparty_base_successes":b,
        "paired_firstparty_adapted_successes":a,
        "paired_firstparty_delta":a-b,
        "paired_base_first_plus_hits":sum(
            x["model_chosen_click_hits_first_plus"] for x in base),
        "paired_adapted_first_plus_hits":sum(
            x["model_chosen_click_hits_first_plus"] for x in adapted),
        "eval_exact_pngs_identical_before_after":True,
        "no_train_eval_png_byte_duplicates":True,
        "firstparty_previously_exposed_source_only":True,
        "independent_blind_ood_proven":False,
        "official_train_dev_ood_admitted":False,
        "old_h173_consumed_ood_used":False,
        "model_generalized_improvement_proven":False,
        "model_weights_published":False,
        "raw_predictions_images_coords_labels_saved_to_output":False,
    }


def run():
    import torch
    hardware=check_free_t4(torch)
    root=Path("/tmp/hala-h272-firstparty")
    require(not root.exists(),"fresh private training session only")
    root.mkdir(mode=0o700)
    try:
        app=clone_original_source(root/"source")
        sys.path.insert(0,str(app.parents[1]))
        from training.scripts.train_adapter import (
            prepare_inputs,JOINT_LORA_TARGET_PATTERN,
            qualify_joint_target_names)
        from h61.supervision import weighted_lm_loss
        from h268.free_t4_constrained_native import (
            build_core_token_filter,verify_exact_lmfe_installed_source)
        import transformers,peft,importlib.metadata
        require(transformers.__version__=="5.16.1"
                and peft.__version__=="0.20.0",
                "pinned Transformers/PEFT required")
        verify_exact_lmfe_installed_source(
            importlib.metadata.distribution("lm-format-enforcer"))
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        from peft import LoraConfig,get_peft_model
        from playwright.sync_api import sync_playwright
        os.environ.update(
            HF_HOME=str(root/"hf"),HF_HUB_CACHE=str(root/"hf"/"hub"),
            HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        torch.manual_seed(270)
        cache=str(root/"hf"/"hub")
        processor=AutoProcessor.from_pretrained(
            MODEL_ID,revision=MODEL_REV,cache_dir=cache,trust_remote_code=False)
        model=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,revision=MODEL_REV,cache_dir=cache,
            trust_remote_code=False,torch_dtype=torch.float16,
            attn_implementation="sdpa").to("cuda:0").eval()
        require(getattr(model.config,"model_type",None)=="qwen3_vl",
                "wrong pinned pretrained model")
        grammar=build_core_token_filter(processor.tokenizer)
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
            try:
                media,train_rows,train_hashes=build_train(browser,app,root)
                before,base_stamps=eval_pair(
                    browser,app,processor,model,grammar,train_hashes)
                # H106 joint target selection: eight late ViT layers
                # and actual language q/v, no unrelated trainable params.
                modules=dict(model.named_modules())
                joint_targets,vision_targets,language_targets=(
                    qualify_joint_target_names(modules))
                require(len(vision_targets)==8
                        and 16<=len(language_targets)<=256
                        and all(isinstance(modules[n],torch.nn.Linear)
                                for n in joint_targets),
                        "original joint scope must be real Linear layers")
                lora=LoraConfig(
                    r=4,lora_alpha=8,
                    target_modules=JOINT_LORA_TARGET_PATTERN,
                    lora_dropout=0,task_type="CAUSAL_LM",bias="none")
                model=get_peft_model(model,lora)
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                model.config.use_cache=False
                named_params=[(n,p) for n,p in model.named_parameters()
                              if p.requires_grad]
                params=[p for n,p in named_params]
                require(len(named_params)==2*len(joint_targets)
                        and all("lora_" in n
                            and any(v in n for v in joint_targets)
                            for n,p in named_params)
                        and 0<sum(p.numel() for p in params)<=8_000_000,
                        "only exact original H106 joint adapter tensors")
                require(sum(".visual.blocks." in n for n,p in named_params)==16
                        and sum(".language_model." in n for n,p in named_params)
                            ==2*len(language_targets),
                        "joint vision and language adapter partition mismatch")
                # Match existing Hala trainer's fp32 trainable adapter
                # practice; base vision + text weights remain frozen.
                for p in params:
                    p.data=p.data.float()
                optimizer=torch.optim.AdamW(params,lr=5e-5)
                # Match the actual Hala trainer conservative FP16 scaling.
                # Default 65536 overflowed H270 weighted coordinate grads.
                scaler=torch.amp.GradScaler(
                    "cuda",init_scale=1.0,enabled=True)
                model.train()
                steps=[]
                for i,row in enumerate(train_rows):
                    batch,labels,weights=prepare_inputs(
                        row,processor,media,8192,
                        supervision="coordinate_weighted",
                        coordinate_weight=COORDINATE_WEIGHT,
                        return_weights=True)
                    coordinate_tokens=int((weights>1.0).sum().item())
                    require(coordinate_tokens>0 and
                            int((labels!=-100).sum().item())>=coordinate_tokens,
                            "bad completion-only coordinate mask")
                    batch=batch.to("cuda:0")
                    labels=labels.to("cuda:0")
                    weights=weights.to("cuda:0")
                    before_params=[p.detach().cpu().clone() for p in params]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda",dtype=torch.float16):
                        logits=model(**batch).logits
                        loss=weighted_lm_loss(logits,labels,weights)
                    require(bool(torch.isfinite(loss).item()),
                            "nonfinite real coordinate-weighted loss")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    norm=torch.nn.utils.clip_grad_norm_(params,1.0).item()
                    require(math.isfinite(norm) and norm>0,
                            "nonfinite/zero coordinate gradient")
                    scaler.step(optimizer)
                    scaler.update()
                    changed=sum(not torch.equal(original,p.detach().cpu())
                                for original,p in zip(before_params,params))
                    require(changed>0,"no actual new LoRA parameter update")
                    steps.append({
                        "changed_tensors":changed,
                        "coordinate_tokens":coordinate_tokens,
                        "gradient_norm":float(norm),"loss":float(loss.item())})
                    del batch,labels,weights,before_params,logits,loss
                    torch.cuda.synchronize(0)
                model.eval()
                model.config.use_cache=True
                after,adapted_stamps=eval_pair(
                    browser,app,processor,model,grammar,train_hashes)
                require(base_stamps==adapted_stamps,
                        "paired original research eval screenshots changed")
                report=measured_aggregate(
                    steps,before,after,hardware,
                    language_modules=len(language_targets))
            finally:
                browser.close()
        (Path("/kaggle/working")/OUTPUT).write_text(
            json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        return report
    finally:
        # Only ephemeral model/screenshot/label/optimizer data, never export.
        shutil.rmtree(root,ignore_errors=True)


if __name__=="__main__":
    print(json.dumps(run(),sort_keys=True),flush=True)
