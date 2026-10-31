"""H274: classify private Qwen relative output failures without retaining text.

Owner-private FREE Kaggle T4 diagnostic on already-exposed first-party
Stockroom scenes. Never save raw completions, point coordinates, screenshots,
adapter weights, or gold. This is NOT a performance benchmark.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys

SOURCE_SHA="53d7fb25e113f8f2cd337bb4aebeafac026d1a22"
SOURCE_BRANCH="ps/h273-qwen-native-normalized-point-guard"
SOURCE_REPO="https://github.com/TheHalfMoon/Hala.git"
MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
MODEL_SHA="89644892e4d85e24eaac8bacfd4f463576704203"
HTML_SHA="3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a"
REPORT="hala-h274-private-relative-format-reason-aggregate.json"
SCENES=("ar_sorted_tape","ar_sorted_gloves")

def demand(ok,reason):
    if not ok:
        raise ValueError("H274 fail-closed: "+reason)

def classify_strict_model_text(raw,regex):
    """Whitelisted low-entropy labels. Never echo or return text or points."""
    if type(raw) is not str:
        return "NON_STRING"
    if not raw:
        return "EMPTY"
    if len(raw)>128:
        return "OVER_128_CHARS"
    if re.fullmatch(regex,raw):
        return "STRICT_EXACT_VALID"
    if raw.lstrip().startswith("```"):
        return "MARKDOWN_FENCE"
    try:
        parsed=json.loads(raw)
    except (json.JSONDecodeError,ValueError):
        if raw.startswith('{"actions":[') and not raw.endswith("]}"):
            return "INCOMPLETE_OR_UNCLOSED_JSON"
        return "NOT_JSON_OR_NONCANONICAL_PREFIX"
    if type(parsed) is not dict or set(parsed)!={"actions"}:
        return "JSON_DIFFERENT_TOP_LEVEL_KEYS"
    actions=parsed.get("actions")
    if type(actions) is not list or len(actions)!=1 or type(actions[0]) is not dict:
        return "JSON_DIFFERENT_ACTION_COUNT"
    action=actions[0]
    if action.get("kind")=="click":
        point=action.get("point")
        if type(point) is not list or len(point)!=2 or any(
                type(v) is not int or not 0<=v<=999 for v in point):
            return "JSON_CLICK_INVALID_RELATIVE_POINT"
        return "JSON_VALID_CLICK_NONCANONICAL_SYNTAX"
    if action.get("kind")=="abstain":
        return "JSON_VALID_ABSTAIN_NONCANONICAL_SYNTAX"
    return "JSON_OTHER_ACTION"

def verify_free(torch):
    root=Path("/kaggle/working")
    demand(sys.platform=="linux" and root.is_dir()
           and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
           and os.environ.get("HALA_H274_PRIVATE_FREE_NONCOMMERCIAL")=="1"
           and torch.cuda.is_available()
           and torch.cuda.device_count() in (1,2)
           and all("T4" in torch.cuda.get_device_name(i)
                   for i in range(torch.cuda.device_count())),
           "authorized owner-private free Kaggle T4 only")
    demand(shutil.disk_usage("/tmp").free>=8*1024**3,
           "remote ephemeral model space too low")
    demand(not (root/REPORT).exists(),"do not overwrite private aggregate")
    return {"free_t4":True,"devices":torch.cuda.device_count()}

def clone_source(root):
    demand(not root.exists(),"fresh pinned first-party checkout")
    subprocess.run(["git","clone","--depth","1","--single-branch",
                    "--branch",SOURCE_BRANCH,SOURCE_REPO,str(root)],
                   check=True,capture_output=True,text=True)
    head=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],
                        check=True,capture_output=True,text=True).stdout.strip()
    dirty=subprocess.run(["git","-C",str(root),"status","--porcelain"],
                         check=True,capture_output=True,text=True).stdout.strip()
    demand(head==SOURCE_SHA and not dirty,"exact clean source SHA required")
    html=root/"h262"/"stockroom.html"
    demand(html.is_file() and not html.is_symlink()
           and sha256(html.read_bytes()).hexdigest()==HTML_SHA,
           "original app hash changed")
    return html

def summarize(rows,hardware):
    demand(hardware.get("free_t4") is True
           and type(rows) is list and len(rows)==2,
           "real paired relative diagnostic required")
    for i,r in enumerate(rows):
        demand(r.get("scene")==SCENES[i]
               and r.get("real_model_generation_count")==1
               and r.get("real_token_filter_generate_calls")==1
               and r.get("original_screenshot_unchanged") is True
               and r.get("format_class") in {
                   "STRICT_EXACT_VALID","NON_STRING","EMPTY","OVER_128_CHARS",
                   "MARKDOWN_FENCE","INCOMPLETE_OR_UNCLOSED_JSON",
                   "NOT_JSON_OR_NONCANONICAL_PREFIX",
                   "JSON_DIFFERENT_TOP_LEVEL_KEYS","JSON_DIFFERENT_ACTION_COUNT",
                   "JSON_CLICK_INVALID_RELATIVE_POINT",
                   "JSON_VALID_CLICK_NONCANONICAL_SYNTAX",
                   "JSON_VALID_ABSTAIN_NONCANONICAL_SYNTAX","JSON_OTHER_ACTION"},
               "unverified model/format diagnosis")
    return {
      "status":"H274_PRIVATE_QWEN_RELATIVE_FORMAT_DIAGNOSTIC_NOT_OOD",
      "pinned_original_source_sha":SOURCE_SHA,
      "pinned_model_id":MODEL_ID,"pinned_model_revision":MODEL_SHA,
      "hardware":hardware,
      "real_pretrained_image_generations":2,
      "rows":rows,
      "native_actions_dispatched":0,
      "real_optimizer_updates":0,
      "raw_model_text_points_or_images_saved":False,
      "weights_or_adapters_published":False,
      "independent_blind_ood":False,
      "official_train_dev_ood_admitted":False,
      "generalizable_hala_skill_proven":False,
    }

def run():
    import torch
    hardware=verify_free(torch)
    root=Path("/tmp/hala-h274-private-format-reasons")
    demand(not root.exists(),"fresh ephemeral experiment only")
    root.mkdir(mode=0o700)
    try:
        html=clone_source(root/"source")
        sys.path.insert(0,str(html.parents[1]))
        from h271.vision_lora_native import EVAL_SCENES,original_case
        from h273.relative_native_bridge import (
            RELATIVE_PROMPT,NORMALIZED_REGEX,build_normalized_token_filter)
        from h268.free_t4_constrained_native import (
            NativeGrammarModel,verify_exact_lmfe_installed_source)
        from h208.preloaded_vlm_policy import PreloadedVisionPolicy
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        import transformers,importlib.metadata
        from playwright.sync_api import sync_playwright
        demand(transformers.__version__=="5.16.1","pinned HF5 required")
        verify_exact_lmfe_installed_source(
            importlib.metadata.distribution("lm-format-enforcer"))
        os.environ.update(HF_HOME=str(root/"hf"),
            HF_HUB_CACHE=str(root/"hf"/"hub"),
            HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        torch.manual_seed(274)
        cache=str(root/"hf"/"hub")
        processor=AutoProcessor.from_pretrained(
            MODEL_ID,revision=MODEL_SHA,cache_dir=cache,
            trust_remote_code=False)
        model=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,revision=MODEL_SHA,cache_dir=cache,
            trust_remote_code=False,torch_dtype=torch.float16,
            attn_implementation="sdpa").to("cuda:0").eval()
        demand(getattr(model.config,"model_type",None)=="qwen3_vl",
               "wrong pretrained model")
        filter_fn=build_normalized_token_filter(processor.tokenizer)
        result_rows=[]
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
            try:
                for scene in EVAL_SCENES:
                    demand(scene[0] in SCENES,"only original preregistered scenes")
                    ctx=None
                    wrapped=NativeGrammarModel(model,filter_fn)
                    policy=PreloadedVisionPolicy(
                        processor,wrapped,max_new_tokens=160)
                    try:
                        ctx,page,png,_,_=original_case(
                            browser,html,scene,training=False)
                        # ONLY volatile raw model string; never log or save it.
                        raw=policy(png,RELATIVE_PROMPT)
                        category=classify_strict_model_text(
                            raw,NORMALIZED_REGEX)
                        del raw
                        unchanged=(sha256(png).digest()==sha256(
                            page.screenshot(
                                animations="disabled",full_page=False)).digest())
                        demand(unchanged,"original screenshot changed during generate")
                        result_rows.append({
                            "scene":scene[0],
                            "real_model_generation_count":wrapped.completed_generations,
                            "real_token_filter_generate_calls":wrapped.grammar_installed_calls,
                            "original_screenshot_unchanged":unchanged,
                            "format_class":category,
                        })
                    finally:
                        if ctx is not None:
                            ctx.close()
            finally:
                browser.close()
        output=summarize(result_rows,hardware)
        (Path("/kaggle/working")/REPORT).write_text(
            json.dumps(output,sort_keys=True,indent=2)+"\n",
            encoding="utf-8")
        return output
    finally:
        shutil.rmtree(root,ignore_errors=True)

if __name__=="__main__":
    print(json.dumps(run(),sort_keys=True),flush=True)
