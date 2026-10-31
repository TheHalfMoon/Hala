"""H279: genuine private Qwen native clicks from strict relative visual output.

Owner-private FREE Kaggle T4 diagnostic on already-exposed first-party
Stockroom scenes. Use exactly two previously exposed first-party scenes and frozen prompt/grammar.
Actuate only strict native conversions after a screenshot-integrity check;
score via source-owned state only after prediction. No blinded OOD claim.
Never persist raw completions, coordinates, screenshots, weights, or gold.
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
REPORT="hala-h279-private-relative-native-aggregate.json"
SCENES=("ar_sorted_tape","ar_sorted_gloves")

def demand(ok,reason):
    if not ok:
        raise ValueError("H279 fail-closed: "+reason)

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
        if (raw.startswith('{"actions":[') and (raw.count("{")>raw.count("}") or raw.count("[")>raw.count("]"))):
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
           and os.environ.get("HALA_H279_PRIVATE_FREE_NONCOMMERCIAL")=="1"
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


def summarize(rows, hardware):
    """Publish bounded aggregate booleans only; never coordinates or model text."""
    demand(hardware.get("free_t4") is True
           and type(rows) is list and len(rows) == 2,
           "two real image-conditioned source-owned cases required")
    for i, row in enumerate(rows):
        demand(type(row) is dict and set(row) == {
            "scene", "real_model_generation_count", "real_token_filter_generate_calls",
            "original_screenshot_unchanged", "format_class",
            "strict_relative_action_valid", "native_action_kind",
            "native_action_dispatched", "first_plus_target_hit",
            "firstparty_inventory_task_success", "other_rows_unchanged",
        } and row["scene"] == SCENES[i]
           and row["real_model_generation_count"] == 1
           and row["real_token_filter_generate_calls"] == 1
           and row["original_screenshot_unchanged"] is True
           and row["format_class"] in {
               "STRICT_EXACT_VALID", "NON_STRING", "EMPTY",
               "OVER_128_CHARS", "MARKDOWN_FENCE",
               "INCOMPLETE_OR_UNCLOSED_JSON",
               "NOT_JSON_OR_NONCANONICAL_PREFIX",
               "JSON_DIFFERENT_TOP_LEVEL_KEYS",
               "JSON_DIFFERENT_ACTION_COUNT",
               "JSON_CLICK_INVALID_RELATIVE_POINT",
               "JSON_VALID_CLICK_NONCANONICAL_SYNTAX",
               "JSON_VALID_ABSTAIN_NONCANONICAL_SYNTAX",
               "JSON_OTHER_ACTION",
           }
           and type(row["strict_relative_action_valid"]) is bool
           and row["strict_relative_action_valid"] == (
               row["format_class"] == "STRICT_EXACT_VALID")
           and row["native_action_kind"] in ("click", "abstain", None)
           and type(row["native_action_dispatched"]) is bool
           and type(row["first_plus_target_hit"]) is bool
           and type(row["firstparty_inventory_task_success"]) is bool
           and type(row["other_rows_unchanged"]) is bool
           and (not row["native_action_dispatched"]
                or row["strict_relative_action_valid"])
           and (not row["first_plus_target_hit"]
                or (row["native_action_kind"] == "click"
                    and row["native_action_dispatched"]))
           and (not row["firstparty_inventory_task_success"]
                or (row["first_plus_target_hit"]
                    and row["other_rows_unchanged"])),
           "invalid native visual action evidence")
    return {
        "status": "H279_PRIVATE_PRETRAINED_NATIVE_RELATIVE_FIRSTPARTY_NOT_OOD",
        "pinned_original_source_sha": SOURCE_SHA,
        "pinned_model_id": MODEL_ID,
        "pinned_model_revision": MODEL_SHA,
        "hardware": hardware,
        "real_pretrained_image_generations": 2,
        "rows": rows,
        "strict_relative_actions": sum(r["strict_relative_action_valid"] for r in rows),
        "native_actions_dispatched": sum(r["native_action_dispatched"] for r in rows),
        "first_plus_target_hits": sum(r["first_plus_target_hit"] for r in rows),
        "actual_firstparty_task_success": sum(
            r["firstparty_inventory_task_success"] for r in rows),
        "real_optimizer_updates": 0,
        "no_raw_predictions_points_or_images_saved": True,
        "weights_or_adapters_published": False,
        "independent_blind_ood": False,
        "official_train_dev_ood_admitted": False,
        "generalizable_hala_skill_proven": False,
    }

def run():
    import torch
    hardware=verify_free(torch)
    root=Path("/tmp/hala-h279-private-relative-native")
    demand(not root.exists(),"fresh ephemeral experiment only")
    root.mkdir(mode=0o700)
    try:
        html=clone_source(root/"source")
        sys.path.insert(0,str(html.parents[1]))
        from h271.vision_lora_native import EVAL_SCENES,original_case
        from h273.relative_native_bridge import (
            RELATIVE_PROMPT,NORMALIZED_REGEX,build_normalized_token_filter,
            normalize_strict_action)
        from h268.free_t4_constrained_native import (
            NativeGrammarModel,verify_exact_lmfe_installed_source)
        from h208.preloaded_vlm_policy import PreloadedVisionPolicy
        from h263.native_browser_executor import execute_one
        from h199.native_action_sft import parse_actions
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
                        ctx,page,_,_,_=original_case(
                            browser,html,scene,training=False)
                        # Preserve H274 original scene setup, prompt, model,
                        # relative grammar, seed, and 160-token generation.
                        page.locator("#search").evaluate("(el) => el.blur()")
                        from h264.screenshot_locked_research import _observed_png
                        png=_observed_png(page)
                        before_counts=[
                            int(x) for x in
                            page.locator(".item .count").all_inner_texts()
                        ]
                        demand(len(before_counts)>=1,
                               "missing original first-party visible items")
                        # ONLY the PNG and text prompt enter the actual
                        # pretrained model; NEVER a DOM/target/gold/bbox.
                        raw=policy(png,RELATIVE_PROMPT)
                        category=classify_strict_model_text(
                            raw,NORMALIZED_REGEX)
                        unchanged=(sha256(png).digest()==sha256(
                            _observed_png(page)).digest())
                        demand(unchanged,"original screenshot changed during generate")
                        # Frame conversion is the exact prospective H273
                        # relative 0..999 -> 960x720 bridge. Invalid JSON
                        # causes NO action; no silent cleanup or gold click.
                        strict_native=None
                        try:
                            strict_native=normalize_strict_action(raw)
                        except ValueError:
                            pass
                        del raw
                        valid=(strict_native is not None)
                        kind=None
                        dispatched=False
                        first_hit=False
                        success=False
                        neighbors_unchanged=True
                        if valid:
                            action=parse_actions(strict_native)["actions"][0]
                            kind=action["kind"]
                            outcome=execute_one(page,strict_native)
                            dispatched=bool(outcome["executed"])
                            del strict_native
                            after_counts=[
                                int(x) for x in
                                page.locator(".item .count").all_inner_texts()
                            ]
                            demand(len(after_counts)==len(before_counts),
                                   "original firstparty row count changed")
                            neighbors_unchanged=(
                                after_counts[1:]==before_counts[1:])
                            if kind=="click" and dispatched:
                                # Only AFTER model prediction AND native
                                # actuation does evaluator look at the
                                # pinned first source-owned DOM button.
                                point=action["point"]
                                rect=page.locator(
                                    "button[data-restock]").first.bounding_box()
                                if rect is not None:
                                    first_hit=bool(
                                        rect["x"]<=point[0]<rect["x"]+rect["width"]
                                        and rect["y"]<=point[1]<rect["y"]+rect["height"])
                                del point
                            success=bool(
                                kind=="click" and dispatched and first_hit
                                and neighbors_unchanged
                                and after_counts[0]==before_counts[0]+1)
                        result_rows.append({
                            "scene":scene[0],
                            "real_model_generation_count":wrapped.completed_generations,
                            "real_token_filter_generate_calls":wrapped.grammar_installed_calls,
                            "original_screenshot_unchanged":unchanged,
                            "format_class":category,
                            "strict_relative_action_valid":valid,
                            "native_action_kind":kind,
                            "native_action_dispatched":dispatched,
                            "first_plus_target_hit":first_hit,
                            "firstparty_inventory_task_success":success,
                            "other_rows_unchanged":neighbors_unchanged,
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
