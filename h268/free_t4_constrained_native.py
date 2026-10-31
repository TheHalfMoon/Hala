"""H268: genuine pretrained multimodal Qwen *token-level* native JSON grammar probe.

Owner-private FREE Kaggle T4 FIRST-PARTY research only. This does NOT train a
model, admit TRAIN/DEV/OOD data, reuse H173 sealed OOD or prove model gain.
Raw model text, images, reference geometry and weights never enter output.
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

KERNEL_RESULT = "hala-h268-private-constrained-baseline.json"
RESULT_STATUS = "H268_GENUINE_PRETRAINED_CONSTRAINED_FIRSTPARTY_NOT_OOD"
SOURCE_REPO = "https://github.com/TheHalfMoon/Hala.git"
SOURCE_BRANCH = "ps/h267-real-vlm-format-diagnostics"
SOURCE_SHA = "8fdd0012f996524ce8c684b5a5ba698f8a2a9ef8"
HTML_SHA256 = "3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
LMFE_VERSION = "0.11.3"
MODEL_TASK = ("Click the +1 button belonging to the first inventory item "
              "to add one unit. Return ONLY one JSON actions object with "
              "a single click or abstain action. Use screenshot pixels.")
# The two alternatives are the FULL, exact H199 native action JSON.
# Coordinates are freely model-chosen in the legal viewport. No gold or
# oracle coordinate is present in grammar, prompt or inference adapter.
X = "(0|[1-9][0-9]?|[1-8][0-9][0-9]|9[0-5][0-9])"
Y = "(0|[1-9][0-9]?|[1-6][0-9][0-9]|7[0-1][0-9])"
ABSTAIN = r'\{"actions":\[\{"kind":"abstain"\}\]\}'
CLICK = r'\{"actions":\[\{"kind":"click","point":\[' + X + "," + Y + r'\]\}\]\}'
NATIVE_REGEX = "(" + ABSTAIN + "|" + CLICK + ")"
MAX_NEW_TOKENS = 160


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("H268 private constrained inference blocked: " + reason)


def strict_grammar_proof() -> None:
    from h199.native_action_sft import parse_actions
    examples = (
        '{"actions":[{"kind":"abstain"}]}',
        '{"actions":[{"kind":"click","point":[0,0]}]}',
        '{"actions":[{"kind":"click","point":[959,719]}]}',
        '{"actions":[{"kind":"click","point":[12,650]}]}',
    )
    for sample in examples:
        require(re.fullmatch(NATIVE_REGEX, sample) is not None,
                "native grammar rejects legal action")
        parse_actions(sample)
    invalid = (
        "```json\n" + examples[0] + "\n```",
        '{"actions":[{"kind":"click","point":[960,719]}]}',
        '{"actions":[{"kind":"click","point":[959,720]}]}',
        '{"actions":[{"kind":"click","point":[01,1]}]}',
        '{"actions":[{"kind":"click","point":[1,-1]}]}',
        '{"actions":[{"kind":"drag","start":[10,20],"end":[30,40]}]}',
        '{"actions":[{"kind":"abstain"}],"extra":1}',
        '{"actions":[{"kind":"click","point":[1,2]}]} trailing',
    )
    for sample in invalid:
        require(re.fullmatch(NATIVE_REGEX, sample) is None,
                "native grammar accepts disallowed action or markdown")


class NativeGrammarModel:
    """Wraps the REAL loaded model, not a simulator or oracle.

    The parser token filter is installed DURING generate; not a
    post-generation fence-stripping or JSON repair. It preserves optional
    abstention and allows ANY valid pixel coordinate chosen by the model.
    """
    def __init__(self, model, prefix_allowed_tokens_fn):
        require(callable(getattr(model, "generate", None))
                and callable(prefix_allowed_tokens_fn),
                "real preloaded model and grammar filter needed")
        self._model = model
        self.device = model.device
        self._filter = prefix_allowed_tokens_fn
        self.completed_generations = 0
        self.grammar_installed_calls = 0

    def eval(self):
        self._model.eval()
        return self

    def generate(self, **kwargs):
        require("prefix_allowed_tokens_fn" not in kwargs,
                "caller must not replace pinned grammar")
        self.grammar_installed_calls += 1
        output = self._model.generate(
            **kwargs, prefix_allowed_tokens_fn=self._filter)
        self.completed_generations += 1
        return output


def measure(*, completed: int, filtered: int, format_class: str,
            kind: str | None, dispatched: bool, before: int, after: int,
            gpu: dict) -> dict:
    """Aggregate-only. Never include raw model response, PNG or gold pixels."""
    from h267.free_kaggle_format_probe import ALLOWED_FORMATS
    require(type(completed) is int and completed in (0, 1)
            and type(filtered) is int and filtered in (0, 1)
            and completed <= filtered
            and format_class in ALLOWED_FORMATS
            and kind in (None, "click", "abstain")
            and type(dispatched) is bool
            and type(before) is int and type(after) is int
            and gpu.get("real_free_t4") is True,
            "malformed private aggregate")
    require(not dispatched or (completed == filtered == 1 and
                               format_class == "STRICT_NATIVE_JSON" and
                               kind == "click"),
            "never report dispatch without strict real model output")
    require(not (kind == "abstain" and dispatched),
            "abstention cannot cause browser input")
    success = bool(dispatched and after == before + 1)
    return {
        "status":RESULT_STATUS,
        "pinned_public_source_sha":SOURCE_SHA,
        "pinned_pretrained_model":MODEL_ID,
        "pinned_pretrained_revision":MODEL_REVISION,
        "decoding_library_version":LMFE_VERSION,
        "real_model_generations_completed":completed,
        "token_filter_generate_calls":filtered,
        "raw_output_format_class":format_class,
        "strict_action_kind":kind,
        "native_browser_action_dispatched":dispatched,
        "first_party_stock_plus_one":after == before+1,
        "first_party_task_success":success,
        "free_gpu":gpu,
        "fresh_source_disjoint_blind_ood":False,
        "official_train_dev_ood_admitted":False,
        "consumed_h173_scored_ood_used":False,
        "new_hala_optimizer_steps":0,
        "trained_hala_model_gain":False,
        "raw_model_text_screenshots_gold_saved":False,
        "kaggle_working_model_cache":False,
    }


def verify_remote_free_gpu(torch) -> dict:
    require(sys.platform == "linux" and Path("/kaggle/working").is_dir()
            and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            and os.environ.get("HALA_H268_NONCOMMERCIAL_FREE_GPU")=="1",
            "owner-private free Kaggle only")
    require(torch.cuda.is_available() and torch.cuda.device_count() in (1,2)
            and all("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())),
            "only actual free NVIDIA T4 allowed")
    require(shutil.disk_usage("/tmp").free >= 8*1024**3,
            "ephemeral model cache needs 8GiB free")
    return {"real_free_t4":True, "cuda_t4_count":torch.cuda.device_count()}


def original_source_into_tmp(root: Path) -> Path:
    require(not root.exists(),"fresh /tmp source required")
    subprocess.run(
        ["git","clone","--depth","1","--single-branch",
         "--branch",SOURCE_BRANCH,SOURCE_REPO,str(root)],
        check=True,capture_output=True,text=True)
    head=subprocess.run(
        ["git","-C",str(root),"rev-parse","HEAD"],
        check=True,capture_output=True,text=True).stdout.strip()
    require(head == SOURCE_SHA,"wrong original source commit")
    clean=subprocess.run(
        ["git","-C",str(root),"status","--porcelain"],
        check=True,capture_output=True,text=True).stdout.strip()
    require(not clean, "source tree changed")
    html=root/"h262"/"stockroom.html"
    require(html.is_file() and not html.is_symlink() and
            sha256(html.read_bytes()).hexdigest()==HTML_SHA256,
            "original first-party source drift")
    return html


def run() -> dict:
    # Explicitly disallow owner Windows as a model-weight runtime.
    if sys.platform != "linux":
        raise ValueError("H268 only runs on ephemeral FREE Kaggle T4")
    import torch
    gpu=verify_remote_free_gpu(torch)
    out=Path("/kaggle/working")/KERNEL_RESULT
    require(not out.exists(), "create-only private aggregate output")
    root=Path("/tmp/hala-h268-research")
    require(not root.exists(),"fresh and empty /tmp session required")
    root.mkdir(mode=0o700)
    source=original_source_into_tmp(root/"source")
    # The original H199 native-action parser must be importable BEFORE
    # the CPU grammar proof; never bypass this check on remote Kaggle.
    sys.path.insert(0,str(source.parents[1]))
    os.environ.update(
        HF_HOME=str(root/"hf"),
        HF_HUB_CACHE=str(root/"hf"/"hub"),
        HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1",
    )
    strict_grammar_proof()
    import importlib.metadata
    import transformers
    require(transformers.__version__ == "5.16.1",
            "verified H265 transformers compatibility required")
    require(importlib.metadata.version("lm-format-enforcer")==LMFE_VERSION,
            "exact token grammar library version required")
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from lmformatenforcer import RegexParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn)
    from playwright.sync_api import sync_playwright
    sys.path.insert(0,str(source.parents[1]))
    from h208.preloaded_vlm_policy import PreloadedVisionPolicy
    from h263.native_browser_executor import new_offline_context
    from h264.screenshot_locked_research import screenshot_locked_turn
    from h267.free_kaggle_format_probe import ClassifyingProcessor

    torch.manual_seed(268)
    cache=str(root/"hf"/"hub")
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_REVISION,trust_remote_code=False,
        cache_dir=cache)
    model=Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,revision=MODEL_REVISION,trust_remote_code=False,
        torch_dtype=torch.float16,attn_implementation="sdpa",
        cache_dir=cache).to("cuda:0").eval()
    require(getattr(model.config,"model_type",None)=="qwen3_vl",
            "wrong pretrained model architecture")
    token_filter=build_transformers_prefix_allowed_tokens_fn(
        processor.tokenizer, RegexParser(NATIVE_REGEX))
    require(callable(token_filter), "failed real grammar filter")
    counted=NativeGrammarModel(model,token_filter)
    classified=ClassifyingProcessor(processor)
    policy=PreloadedVisionPolicy(classified,counted,
                                 max_new_tokens=MAX_NEW_TOKENS)
    completed_before=counted.completed_generations
    classified.format_class="NO_DECODE"
    kind=None
    dispatched=False
    first=after=0
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
        try:
            context=new_offline_context(browser)
            try:
                def block_external(route):
                    if route.request.url.startswith(
                        ("http:","https:","ws:","wss:")):
                        route.abort()
                    else:
                        route.continue_()
                context.route("**/*",block_external)
                page=context.new_page()
                url=source.as_uri()
                page.goto(url,wait_until="load")
                first=int(page.locator(".item .count").first.inner_text())
                try:
                    decision=screenshot_locked_turn(
                        page,source,url,MODEL_TASK,policy)
                    kind=decision["kind"]
                    dispatched=decision["executed"]
                except (ValueError,RuntimeError):
                    # No HTML/DOM reference click, no retrospective model
                    # output fence stripping, no lenient JSON parsing.
                    pass
                after=int(page.locator(".item .count").first.inner_text())
            finally:
                context.close()
        finally:
            browser.close()
    result=measure(
        completed=counted.completed_generations-completed_before,
        filtered=counted.grammar_installed_calls,
        format_class=classified.format_class,
        kind=kind,dispatched=dispatched,before=first,after=after,gpu=gpu)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                   encoding="utf-8")
    return result


if __name__=="__main__":
    print(json.dumps(run(),sort_keys=True),flush=True)
