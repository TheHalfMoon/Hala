"""H273 owner-private free T4 pretrained Qwen native-pixel vs relative grid.

Predeclared A/B on two identical FIRST-PARTY H262 RTL screenshots.
No fine tuning or independent OOD. Raw model responses and coordinates
are volatile; only aggregate bools live in Kaggle output.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

SOURCE_REPO="https://github.com/TheHalfMoon/Hala.git"
SOURCE_BRANCH="ps/h272-joint-vision-language-native-pilot"
SOURCE_SHA="6091a13497881774b41fc6cc1244d2b0e882c0fe"
HTML_SHA="3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a"
MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
MODEL_SHA="89644892e4d85e24eaac8bacfd4f463576704203"
RESULT_NAME="hala-h273-private-normalized-vs-native-aggregate.json"
ARMS=("native_pixel_original","qwen_relative_0_999")
ALLOWED_SCENES=("ar_sorted_tape","ar_sorted_gloves")


def check(cond,reason):
    if not cond:
        raise ValueError("H273 blocked: "+reason)


def verify_environment(torch):
    check(sys.platform=="linux"
          and Path("/kaggle/working").is_dir()
          and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
          and os.environ.get("HALA_H273_FREE_PRIVATE_NONCOMMERCIAL")=="1"
          and torch.cuda.is_available()
          and torch.cuda.device_count() in (1,2)
          and all("T4" in torch.cuda.get_device_name(i)
                  for i in range(torch.cuda.device_count())),
          "authorized owner-private free Kaggle NVIDIA T4 required")
    check(shutil.disk_usage("/tmp").free>=8*1024**3,
          "private ephemeral pretrained weights need 8GiB")
    check(not (Path("/kaggle/working")/RESULT_NAME).exists(),
          "create-only private aggregate required")
    return {"free_t4":True,"t4_count":torch.cuda.device_count()}


def clone_firstparty(destination):
    check(not destination.exists(),"fresh private source checkout required")
    subprocess.run(["git","clone","--depth","1","--single-branch",
                    "--branch",SOURCE_BRANCH,SOURCE_REPO,str(destination)],
                   check=True,capture_output=True,text=True)
    rev=subprocess.run(["git","-C",str(destination),"rev-parse","HEAD"],
        check=True,capture_output=True,text=True).stdout.strip()
    status=subprocess.run(["git","-C",str(destination),"status","--porcelain"],
        check=True,capture_output=True,text=True).stdout.strip()
    check(rev==SOURCE_SHA and not status,"original source drift")
    html=destination/"h262"/"stockroom.html"
    check(html.is_file() and not html.is_symlink()
          and sha256(html.read_bytes()).hexdigest()==HTML_SHA,
          "not the author-owned pinned original GUI")
    return html


def run_case(browser,html,scene,arm,processor,model,token_filter):
    from h271.vision_lora_native import original_case,EVAL_SCENES
    from h268.free_t4_constrained_native import NativeGrammarModel
    # The trusted H273 coordinate bridge is source-bundled into this
    # private code-only kernel; no external/private module is loaded.
    from h208.preloaded_vlm_policy import PreloadedVisionPolicy
    from h264.screenshot_locked_research import screenshot_locked_turn
    from h199.native_action_sft import parse_actions
    check(scene in EVAL_SCENES and arm in ARMS and scene[0] in ALLOWED_SCENES,
          "only preregistered first-party research case")
    ctx=None
    model_point=None
    wrapped=NativeGrammarModel(model,token_filter)
    policy=PreloadedVisionPolicy(processor,wrapped,max_new_tokens=160)
    predict=policy if arm==ARMS[0] else bind_relative_policy(policy)
    def inspect_model_only(png,prompt):
        nonlocal model_point
        # Pure observation of model output after explicit normalization for
        # RELATIVE arm; never retarget to a DOM oracle.
        text=predict(png,prompt)
        action=parse_actions(text)["actions"][0]
        model_point=tuple(action["point"]) if action["kind"]=="click" else None
        return text
    kind=None
    dispatched=False
    correct_hit=False
    before=after=0
    png_sha=None
    try:
        ctx,page,png,_,before=original_case(
            browser,html,scene,training=False)
        png_sha=sha256(png).hexdigest()
        try:
            outcome=screenshot_locked_turn(
                page,html,html.as_uri(),FIRST_STOCK_TASK,inspect_model_only)
            kind=outcome["kind"]
            dispatched=bool(outcome["executed"])
        except (ValueError,RuntimeError):
            # Never synthesize valid actions from malformed model output.
            pass
        after=int(page.locator(".item .count").first.inner_text())
        if kind=="click" and dispatched:
            check(model_point is not None,"native click without model point")
            rect=page.locator("button[data-restock]").first.bounding_box()
            if rect is not None:
                correct_hit=(
                    rect["x"]<=model_point[0]<rect["x"]+rect["width"]
                    and rect["y"]<=model_point[1]<rect["y"]+rect["height"])
    finally:
        if ctx is not None:
            ctx.close()
    success=bool(dispatched and after==before+1)
    check(not success or (kind=="click" and correct_hit),
          "stock reward without actual correct FIRST +1 hit")
    result={
        "scene":scene[0],"arm":arm,
        "completed_real_pretrained_generations":wrapped.completed_generations,
        "token_level_filter_generate_calls":wrapped.grammar_installed_calls,
        "output_valid_native_after_preregistered_frame_conversion":
            kind in ("abstain","click"),
        "kind":kind,
        "native_browser_input_dispatched":dispatched,
        "model_chosen_point_hits_correct_first_button":correct_hit,
        "firstparty_inventory_task_success":success,
    }
    return result,png_sha


def aggregate(rows,checks,hardware):
    check(hardware.get("free_t4") is True,"real free T4 required")
    check(type(rows) is list and len(rows)==4
          and len(checks)==4,"two paired comparisons required")
    for row in rows:
        check(row.get("arm") in ARMS
              and row.get("scene") in ALLOWED_SCENES
              and row.get("completed_real_pretrained_generations")==1
              and row.get("token_level_filter_generate_calls")==1
              and type(row.get("native_browser_input_dispatched")) is bool
              and type(row.get("model_chosen_point_hits_correct_first_button")) is bool
              and type(row.get("firstparty_inventory_task_success")) is bool
              and (not row["firstparty_inventory_task_success"]
                   or (row["native_browser_input_dispatched"]
                       and row["model_chosen_point_hits_correct_first_button"])),
              "only real model-generated native actions qualify")
    for i in range(2):
        a,b=rows[2*i:2*i+2]
        check(a["arm"]==ARMS[0] and b["arm"]==ARMS[1]
              and a["scene"]==b["scene"]
              and checks[2*i]==checks[2*i+1],
              "prospective paired screenshots not identical")
    native=[r for r in rows if r["arm"]==ARMS[0]]
    relative=[r for r in rows if r["arm"]==ARMS[1]]
    return {
      "status":"H273_PRETRAINED_RELATIVE_FRAME_FIRSTPARTY_RESEARCH_NOT_OOD",
      "public_source_sha":SOURCE_SHA,
      "pinned_pretrained_model":MODEL_ID,"pinned_model_revision":MODEL_SHA,
      "real_gpu":hardware,
      "real_model_generations":4,
      "pixel_frame_original":native,
      "relative_frame_preregistered":relative,
      "pixel_first_plus_hits":sum(r["model_chosen_point_hits_correct_first_button"] for r in native),
      "relative_first_plus_hits":sum(r["model_chosen_point_hits_correct_first_button"] for r in relative),
      "pixel_native_task_success":sum(r["firstparty_inventory_task_success"] for r in native),
      "relative_native_task_success":sum(r["firstparty_inventory_task_success"] for r in relative),
      "pairwise_identical_screenshot_pixels":True,
      "raw_normalized_text_or_points_persisted":False,
      "original_screen_or_gold_persisted":False,
      "public_weights_or_adapters_persisted":False,
      "new_hala_optimizer_steps":0,
      "independent_blind_ood":False,
      "official_train_dev_ood_admitted":False,
      "model_learned_generalizable_skill_proven":False,
    }


def run():
    import torch
    hardware=verify_environment(torch)
    root=Path("/tmp/hala-h273-relative-point")
    check(not root.exists(),"fresh ephemeral source and weights required")
    root.mkdir(mode=0o700)
    try:
        original=clone_firstparty(root/"source")
        sys.path.insert(0,str(original.parents[1]))
        import importlib.metadata,transformers
        check(transformers.__version__=="5.16.1",
              "pinned Hugging Face Transformers required")
        from h268.free_t4_constrained_native import (
            build_core_token_filter,verify_exact_lmfe_installed_source)
        # Source-bundled H273 normalized model-time grammar is available.
        verify_exact_lmfe_installed_source(
            importlib.metadata.distribution("lm-format-enforcer"))
        from h271.vision_lora_native import EVAL_SCENES
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        from playwright.sync_api import sync_playwright
        os.environ.update(HF_HOME=str(root/"hf"),
            HF_HUB_CACHE=str(root/"hf"/"hub"),
            HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        torch.manual_seed(273)
        cache=str(root/"hf"/"hub")
        processor=AutoProcessor.from_pretrained(
            MODEL_ID,revision=MODEL_SHA,cache_dir=cache,
            trust_remote_code=False)
        model=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,revision=MODEL_SHA,cache_dir=cache,
            trust_remote_code=False,torch_dtype=torch.float16,
            attn_implementation="sdpa").to("cuda:0").eval()
        check(getattr(model.config,"model_type",None)=="qwen3_vl",
              "wrong pretrained model type")
        native_filter=build_core_token_filter(processor.tokenizer)
        relative_filter=build_normalized_token_filter(processor.tokenizer)
        rows=[]
        stamps=[]
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(
                headless=True,args=["--no-sandbox"])
            try:
                for scene in EVAL_SCENES:
                    for arm,token_filter in (
                        (ARMS[0],native_filter),
                        (ARMS[1],relative_filter)):
                        result,digest=run_case(
                            browser,original,scene,arm,
                            processor,model,token_filter)
                        rows.append(result)
                        stamps.append(digest)
            finally:
                browser.close()
        result=aggregate(rows,stamps,hardware)
        (Path("/kaggle/working")/RESULT_NAME).write_text(
            json.dumps(result,sort_keys=True,indent=2)+"\n",
            encoding="utf-8")
        return result
    finally:
        shutil.rmtree(root,ignore_errors=True)


if __name__=="__main__":
    print(json.dumps(run(),sort_keys=True),flush=True)
