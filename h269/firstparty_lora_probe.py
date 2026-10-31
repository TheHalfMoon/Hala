"""H269: owner-private FREE T4, tiny first-party LoRA paired native GUI pilot.

Genuine pretrained/adapted comparison ON THE SAME first-party H262 app.
This is NOT officially admitted TRAIN/DEV/OOD, independent unseen evaluation,
published adapter, or claim of generalized Hala improvement. No old H173/H212
data; no private screenshots, labels, predictions or weights leave /tmp.
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
SOURCE_BRANCH = "ps/h268-free-t4-token-grammar"
SOURCE_SHA = "d2de8e264fb2f90fa84a2b50c447bb035e04feef"
APP_SHA = "3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
N_STEPS = 2
RESULT = "hala-h269-private-firstparty-lora-aggregate.json"
TASK = "Click the +1 button of the FIRST visible inventory item to add one unit."
MODES = ("default_en", "sorted_en", "heldout_ar")


def require(ok, why):
    if not ok:
        raise ValueError("H269 fail closed: " + why)


def check_free_environment(torch):
    root = Path("/kaggle/working")
    require(sys.platform=="linux" and root.is_dir()
            and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            and os.environ.get("HALA_H269_PRIVATE_FREE_NONCOMMERCIAL")=="1"
            and torch.cuda.is_available()
            and torch.cuda.device_count() in (1,2)
            and all("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())),
            "owner-private free Kaggle NVIDIA T4 only")
    require(shutil.disk_usage("/tmp").free > 8*1024**3,
            "not enough ephemeral private model cache space")
    require(not (root/RESULT).exists(), "refuse overwriting earlier result")
    return {"free_t4":True,"devices":torch.cuda.device_count()}


def frozen_first_party_source(destination):
    require(not destination.exists(),"fresh isolated first-party source")
    subprocess.run(["git","clone","--depth","1","--single-branch",
                    "--branch",SOURCE_BRANCH,SOURCE_REPO,str(destination)],
                   check=True,capture_output=True,text=True)
    head = subprocess.run(["git","-C",str(destination),"rev-parse","HEAD"],
             check=True,capture_output=True,text=True).stdout.strip()
    require(head==SOURCE_SHA,"pinned Git source identity drift")
    status = subprocess.run(["git","-C",str(destination),"status","--porcelain"],
             check=True,capture_output=True,text=True).stdout.strip()
    require(not status,"frozen first-party source modified")
    html=destination/"h262"/"stockroom.html"
    require(html.is_file() and not html.is_symlink()
            and sha256(html.read_bytes()).hexdigest()==APP_SHA,
            "first-party app identity changed")
    return html


def stockroom_case(browser, html, mode):
    """Fresh original app. DOM oracle accessible only OUTSIDE model callback."""
    require(mode in MODES,"unregistered first-party scene")
    from h263.native_browser_executor import new_offline_context
    context=new_offline_context(browser)
    def route_only_local(route):
        if route.request.url.startswith(("http:","https:","ws:","wss:")):
            route.abort()
        else:
            route.continue_()
    context.route("**/*",route_only_local)
    page=context.new_page()
    page.goto(html.as_uri(),wait_until="load")
    if mode=="sorted_en":
        page.locator("#sort").click()
    elif mode=="heldout_ar":
        page.locator("#lang").click()
    target=page.locator('button[data-restock]').first
    rect=target.bounding_box()
    require(rect is not None and rect["width"]>=10 and rect["height"]>=10,
            "true original restock control missing")
    xy=[int(rect["x"]+rect["width"]/2),
        int(rect["y"]+rect["height"]/2)]
    require(0<=xy[0]<960 and 0<=xy[1]<720,
            "training oracle target outside screenshot viewport")
    # Screenshot PNG remains in RAM on remote Kaggle, never printed.
    png=page.screenshot(animations="disabled",full_page=False)
    require(png[:8]==b"\x89PNG\r\n\x1a\n"
            and len(png)<4*1024*1024,"bad first-party image bytes")
    before=int(page.locator(".item .count").first.inner_text())
    return context,page,png,xy,before


def make_training_rows(browser,html,private_root):
    """Source-authored pixel target from train-scene DOM only; no test oracle."""
    from h199.native_action_sft import parse_actions,source_task_prompt
    from PIL import Image
    media_dir=private_root/"ephemeral_train_images"
    media_dir.mkdir(mode=0o700)
    examples=[]
    for index,mode in enumerate(MODES[:2]):
        context=None
        try:
            context,page,png,point,_=stockroom_case(browser,html,mode)
            with Image.open(BytesIO(png)) as image:
                require(image.size==(960,720) and image.format=="PNG",
                        "invalid training image pixels")
                image.load()
            completion=json.dumps({"actions":[{"kind":"click","point":point}]},
                separators=(",",":"))
            parse_actions(completion,training_target=True)
            path=media_dir/(f"source_train_{index}.png")
            path.write_bytes(png)
            examples.append({
                "id":f"h269-firstparty-train-{index}",
                "image_path":path.name,"image_sha256":sha256(png).hexdigest(),
                "image_format":"PNG",
                "prompt":source_task_prompt(TASK),"completion":completion,
            })
        finally:
            if context is not None:
                context.close()
    require(len(examples)==N_STEPS, "two independent first-party train scenes")
    return media_dir,examples


def run_native_eval(browser,html,processor,model,grammar_filter,train_hashes):
    """Same first-party heldout_AR scene, replayed independently pre and post."""
    from h208.preloaded_vlm_policy import PreloadedVisionPolicy
    from h264.screenshot_locked_research import screenshot_locked_turn
    from h267.free_kaggle_format_probe import ClassifyingProcessor
    from h268.free_t4_constrained_native import NativeGrammarModel
    wrapped=NativeGrammarModel(model,grammar_filter)
    classified=ClassifyingProcessor(processor)
    policy=PreloadedVisionPolicy(classified,wrapped,max_new_tokens=160)
    context=None
    kind=None
    dispatched=False
    before=after=0
    try:
        context,page,png,_,before=stockroom_case(browser,html,"heldout_ar")
        heldout_sha=sha256(png).hexdigest()
        require(heldout_sha not in train_hashes,
                "heldout image exactly duplicates a training screenshot")
        try:
            action=screenshot_locked_turn(page,html,html.as_uri(),TASK,policy)
            kind=action["kind"]
            dispatched=bool(action["executed"])
        except (ValueError,RuntimeError):
            # Never repair JSON, log model text, or override bad click with gold.
            pass
        after=int(page.locator(".item .count").first.inner_text())
    finally:
        if context is not None:
            context.close()
    return {
        "model_generation_complete": wrapped.completed_generations==1,
        "grammar_applied_during_generation":wrapped.grammar_installed_calls==1,
        "strict_format":classified.format_class=="STRICT_NATIVE_JSON",
        "action_kind":kind if kind in ("click","abstain") else None,
        "native_action_dispatched":dispatched,
        "first_party_task_success":bool(dispatched and after==before+1),
    },heldout_sha


def training_summary(step_stats,pre,post,platform):
    """Only learned-weight evidence and first-party coarse outcome flags."""
    require(type(step_stats) is list and len(step_stats)==N_STEPS,
            "two genuine optimizer steps required")
    require(all(type(s["changed_tensors"]) is int
                and s["changed_tensors"]>0
                and math.isfinite(s["grad_norm"]) and s["grad_norm"]>0
                and math.isfinite(s["loss"]) and s["loss"]>0
                for s in step_stats), "genuine finite changed adapter required")
    for scenario in (pre,post):
        require(type(scenario) is dict
                and scenario.get("model_generation_complete") is True
                and scenario.get("grammar_applied_during_generation") is True
                and type(scenario.get("native_action_dispatched")) is bool
                and type(scenario.get("first_party_task_success")) is bool
                and (not scenario["first_party_task_success"]
                     or (scenario["native_action_dispatched"]
                         and scenario.get("action_kind")=="click"
                         and scenario.get("strict_format") is True)),
                "paired first-party task metrics require genuine generated actions")
    require(platform["free_t4"], "real free GPU evidence required")
    return {
      "status":"H269_REAL_FIRSTPARTY_TWO_STEP_LORA_PILOT_NOT_ADMITTED",
      "source_git_sha":SOURCE_SHA,
      "source_first_party_only":True,
      "model_id":MODEL_ID,"model_revision":MODEL_REV,
      "real_free_gpu":platform,
      "real_optimizer_steps":N_STEPS,
      "finite_nonzero_gradients":True,
      "all_steps_change_adapter_tensors":True,
      "steps_aggregate_only":[{"step":i+1,
        "changed_tensors":s["changed_tensors"],
        "grad_norm_finite_nonzero":True,
        "loss_finite_positive":True} for i,s in enumerate(step_stats)],
      "paired_firstparty_untrained":pre,
      "paired_firstparty_adapted":post,
      "heldout_ar_not_in_train_images":True,
      "independent_source_disjoint_ood":False,
      "official_train_dev_ood_admitted":False,
      "consumed_h173_ood_used":False,
      "learned_generalizable_gui_gain_proven":False,
      "trained_model_or_weights_published":False,
      "raw_screenshots_labels_predictions_saved_in_output":False,
    }


def run():
    import torch
    gpu=check_free_environment(torch)
    root=Path("/tmp/hala-h269-private")
    require(not root.exists(), "no reused training experiment")
    root.mkdir(mode=0o700)
    try:
        source=frozen_first_party_source(root/"source")
        sys.path.insert(0,str(source.parents[1]))
        from training.scripts.train_adapter import (
            prepare_inputs,LORA_TARGET_PATTERN)
        from h268.free_t4_constrained_native import (
            build_core_token_filter,verify_exact_lmfe_installed_source,
            LMFE_SOURCE_SHA)
        import transformers, peft, importlib.metadata
        require(transformers.__version__=="5.16.1"
                and peft.__version__=="0.20.0", "pinned HF/PEFT required")
        dist=importlib.metadata.distribution("lm-format-enforcer")
        verify_exact_lmfe_installed_source(dist)
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        from peft import LoraConfig,get_peft_model
        from playwright.sync_api import sync_playwright

        os.environ.update(HF_HOME=str(root/"hf"),
            HF_HUB_CACHE=str(root/"hf"/"hub"),
            HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        torch.manual_seed(269)
        cache=str(root/"hf"/"hub")
        processor=AutoProcessor.from_pretrained(
            MODEL_ID,revision=MODEL_REV,cache_dir=cache,trust_remote_code=False)
        model=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,revision=MODEL_REV,cache_dir=cache,trust_remote_code=False,
            torch_dtype=torch.float16,
            attn_implementation="sdpa").to("cuda:0").eval()
        require(model.config.model_type=="qwen3_vl", "wrong model")
        token_filter=build_core_token_filter(processor.tokenizer)
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
            try:
                media,examples=make_training_rows(browser,source,root)
                train_hashes={x["image_sha256"] for x in examples}
                require(len(train_hashes)==N_STEPS,
                        "training scenes repeated identical image bytes")
                before,baseline_sha=run_native_eval(
                    browser,source,processor,model,token_filter,train_hashes)
                lora=LoraConfig(r=4,lora_alpha=8,
                    target_modules=LORA_TARGET_PATTERN,
                    lora_dropout=0,task_type="CAUSAL_LM",bias="none")
                model=get_peft_model(model,lora)
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                model.config.use_cache=False
                trainable=[p for p in model.parameters() if p.requires_grad]
                require(len(trainable)>0 and sum(p.numel() for p in trainable)<=2_000_000,
                        "only bounded language-only rank4 LoRA")
                opt=torch.optim.AdamW(trainable,lr=5e-5)
                scaler=torch.amp.GradScaler("cuda",enabled=True)
                model.train()
                stats=[]
                for step in range(N_STEPS):
                    sample=examples[step]
                    inputs,labels=prepare_inputs(
                        sample,processor,media,8192)
                    require(int((labels!=-100).sum().item())>0,
                            "completion-only loss mask required")
                    inputs=inputs.to("cuda:0")
                    labels=labels.to("cuda:0")
                    previous=[p.detach().cpu().clone() for p in trainable]
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda",dtype=torch.float16):
                        loss=model(**inputs,labels=labels).loss
                    require(bool(torch.isfinite(loss).item()),
                            "nonfinite optimizer loss")
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    norm=torch.nn.utils.clip_grad_norm_(trainable,1.0).item()
                    require(math.isfinite(norm) and norm>0,
                            "nonfinite or zero real adapter gradient")
                    scaler.step(opt)
                    scaler.update()
                    changed=sum(not torch.equal(old,new.detach().cpu())
                                for old,new in zip(previous,trainable))
                    require(changed>0, "optimizer failed to update adapter")
                    stats.append({"changed_tensors":changed,
                                  "grad_norm":float(norm),
                                  "loss":float(loss.item())})
                    del inputs,labels,loss,previous
                    torch.cuda.synchronize(0)
                model.eval()
                model.config.use_cache=True
                after,adapter_sha=run_native_eval(
                    browser,source,processor,model,token_filter,train_hashes)
                require(baseline_sha==adapter_sha,
                        "paired before/after evaluation screenshots are not identical")
                report=training_summary(stats,before,after,gpu)
            finally:
                browser.close()
        (Path("/kaggle/working")/RESULT).write_text(
            json.dumps(report,sort_keys=True,indent=2)+"\n",
            encoding="utf-8")
        return report
    finally:
        # All model cache, source and first-party screenshot labels are
        # ephemeral/private. No adapter or training media is persisted.
        shutil.rmtree(root,ignore_errors=True)


if __name__=="__main__":
    print(json.dumps(run(),sort_keys=True),flush=True)
