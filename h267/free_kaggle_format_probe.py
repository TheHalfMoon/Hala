"""H267: genuine pre-trained vision inference, private FORMAT diagnostics only.

No raw model output, screenshots, task gold, credentials, weights or private
media persist. First-party H262 research, NOT source-disjoint blind OOD.
No fine-tuning or TRAIN admission.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_SHA = "89644892e4d85e24eaac8bacfd4f463576704203"
SOURCE_REPO = "https://github.com/TheHalfMoon/Hala.git"
SOURCE_BRANCH = "ps/h266-kaggle-chromium-deps"
SOURCE_SHA = "35134ae194644191f5647a0058f5d5f0aecc56a2"
HTML_SHA = "3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a"
REPORT = "hala-h267-private-real-vlm-formats.json"
H262_TASK = ("Click the +1 button belonging to the first inventory item "
             "to add one unit. Return ONLY one JSON actions object with "
             "a single click or abstain action. Use screenshot pixels.")
ARMS = ("original_h199", "strict_json_reminder")
ALLOWED_FORMATS = frozenset({
    "NO_DECODE", "STRICT_NATIVE_JSON", "JSON_WRONG_SCHEMA",
    "MARKDOWN_FENCE", "EMPTY_TEXT", "NOT_JSON",
})
STRICT_SUFFIX = (
    " FINAL OUTPUT CONTRACT: exactly one machine-readable JSON object "
    'of shape {"actions":[{"kind":"click","point":[x,y]}]}. '
    "Use integer screenshot pixel coordinates, x 0..959, y 0..719. "
    "If you cannot find the target, output exactly "
    '{"actions":[{"kind":"abstain"}]}. '
    "Never use markdown, code fences, prose, explanations or extra keys."
)


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError("H267 private research blocked: "+msg)


def classify_raw_text(value: object) -> str:
    """Classify in memory; never return/log the output itself or coordinates."""
    from h199.native_action_sft import parse_actions
    if type(value) is not str or not value.strip():
        return "EMPTY_TEXT"
    try:
        parse_actions(value)
        return "STRICT_NATIVE_JSON"
    except (TypeError, ValueError):
        pass
    if value.strip().startswith("```"):
        return "MARKDOWN_FENCE"
    try:
        json.loads(value)
    except (TypeError, ValueError):
        return "NOT_JSON"
    return "JSON_WRONG_SCHEMA"


class ClassifyingProcessor:
    """Only aggregates category from continuation decoded by REAL processor."""
    def __init__(self, processor):
        self.processor=processor
        self.format_class = "NO_DECODE"

    def apply_chat_template(self, *args, **kwargs):
        return self.processor.apply_chat_template(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        outputs=self.processor.batch_decode(*args, **kwargs)
        self.format_class = (classify_raw_text(outputs[0])
                             if type(outputs) is list and len(outputs)==1
                             else "EMPTY_TEXT")
        return outputs


def prompt_variant(source_prompt: str, arm: str) -> str:
    require(type(source_prompt) is str and "Task:" in source_prompt,
            "H199 task prompt required")
    require(arm in ARMS, "unregistered research prompt arm")
    value = (source_prompt if arm=="original_h199" else
             source_prompt+STRICT_SUFFIX)
    require(len(value)<=2000 and "hidden HTML" in value,
            "prompt must preserve original screenshot-only constraints")
    return value


def check_private_env(root: Path, torch) -> dict:
    require(sys.platform=="linux" and isinstance(root,Path)
            and str(root)=="/kaggle/working" and root.is_dir(),
            "real Kaggle Linux working root only")
    require(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            and os.environ.get("HALA_H267_NONCOMMERCIAL_FREE_GPU")=="1",
            "explicit private noncommercial free kernel flag missing")
    require(torch.cuda.is_available() and torch.cuda.device_count() in (1,2),
            "CUDA T4 required")
    names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    require(all("T4" in n for n in names), "only free Nvidia T4 GPUs")
    require(shutil.disk_usage("/tmp").free>=8*1024**3,
            "at least 8 GiB ephemeral /tmp required to keep weights off output")
    return {"free_t4_gpu_verified":True, "device_count":len(names)}


def source_preflight(repo: Path) -> Path:
    require(not repo.exists(), "source scratch must be fresh")
    subprocess.run(["git","clone","--depth","1","--single-branch",
                    "--branch",SOURCE_BRANCH,SOURCE_REPO,str(repo)],
                   check=True, capture_output=True, text=True)
    rev=subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],
                check=True,capture_output=True,text=True).stdout.strip()
    require(rev==SOURCE_SHA, "upstream revision drift")
    require(not subprocess.run(["git","-C",str(repo),"status","--porcelain"],
                check=True,capture_output=True,text=True).stdout.strip(),
            "upstream source dirty")
    file=repo/"h262"/"stockroom.html"
    require(file.is_file() and not file.is_symlink() and
            sha256(file.read_bytes()).hexdigest()==HTML_SHA,
            "H262 first-party app drift")
    return file


def make_private_report(rows: list[dict], hardware:dict) -> dict:
    """Output only aggregate booleans/enums, never raw predictions or hashes."""
    require(type(rows) is list and len(rows)==len(ARMS), "two actual prompt arms")
    for i,row in enumerate(rows):
        require(type(row) is dict and row.get("arm")==ARMS[i],
                "ordered private experimental arms required")
        require(type(row.get("generate_calls")) is int
                and row["generate_calls"] in (0,1)
                and row.get("format") in ALLOWED_FORMATS
                and type(row.get("native_input_dispatched")) is bool
                and type(row.get("task_succeeded")) is bool
                and row["task_succeeded"] <= row["native_input_dispatched"]
                and not (row["native_input_dispatched"] and
                         (row["format"] != "STRICT_NATIVE_JSON"
                          or row["generate_calls"] != 1)),
                
                "unsafe aggregate outcome")
    require(hardware.get("free_t4_gpu_verified") is True, "unverified hardware")
    return {
      "status":"H267_REAL_PRETRAINED_FIRST_PARTY_FORMAT_RESEARCH_NOT_ADMITTED",
      "source_revision":SOURCE_SHA,
      "pinned_pretrained_model_id":MODEL_ID,
      "pinned_pretrained_revision":MODEL_SHA,
      "free_hardware":hardware,
      "arms":rows,
      "real_generate_calls":sum(r["generate_calls"] for r in rows),
      "first_party_task_successes":sum(r["task_succeeded"] for r in rows),
      "private_model_text_or_image_saved":False,
      "model_weights_in_kaggle_working_outputs":False,
      "official_train_dev_ood_admitted":False,
      "source_independent_blind_ood":False,
      "old_consumed_h173_ood_used":False,
      "new_hala_optimizer_steps":0,
      "trained_hala_model_gain_proven":False,
    }


def run() -> dict:
    import torch
    root=Path("/kaggle/working")
    hardware=check_private_env(root,torch)
    require(not (root/REPORT).exists(), "create-only aggregate result")
    temporary_root=Path("/tmp/hala-h267-research")
    require(not temporary_root.exists(),"refuse reused private temp root")
    temporary_root.mkdir(mode=0o700)
    source=source_preflight(temporary_root/"source")
    os.environ.update(
        HF_HOME=str(temporary_root/"hf"),
        HF_HUB_CACHE=str(temporary_root/"hf"/"hub"),
        HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1"
    )
    # The model is loaded only after free-T4 and original source checks.
    import transformers
    require(transformers.__version__=="5.16.1",
            "exactly pinned HF runtime required")
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from playwright.sync_api import sync_playwright
    sys.path.insert(0,str(source.parents[1]))
    from h208.preloaded_vlm_policy import PreloadedVisionPolicy
    from h263.native_browser_executor import new_offline_context
    from h264.screenshot_locked_research import screenshot_locked_turn
    from h265.free_kaggle_pretrained_probe import CountedPreloadedModel

    torch.manual_seed(267)
    processor=AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_SHA,
                                           trust_remote_code=False,
                                           cache_dir=str(temporary_root/"hf"/"hub"))
    model=Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa",
        cache_dir=str(temporary_root/"hf"/"hub")).to("cuda:0").eval()
    require(model.config.model_type=="qwen3_vl","not pinned model architecture")
    counted=CountedPreloadedModel(model)
    classified=ClassifyingProcessor(processor)
    base_policy=PreloadedVisionPolicy(classified,counted,max_new_tokens=160)
    rows=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
        try:
            for arm in ARMS:
                context=new_offline_context(browser)
                try:
                    def abort_http(route):
                        if route.request.url.startswith(("http:","https:","ws:","wss:")):
                            route.abort()
                        else:
                            route.continue_()
                    context.route("**/*",abort_http)
                    page=context.new_page()
                    page.goto(source.as_uri(),wait_until="load")
                    baseline=int(page.locator(".item .count").first.inner_text())
                    prior=counted.completed_generations
                    classified.format_class="NO_DECODE"
                    dispatched=False
                    try:
                        def prediction(png,prompt):
                            return base_policy(png,prompt_variant(prompt,arm))
                        decision=screenshot_locked_turn(
                            page,source,source.as_uri(),H262_TASK,prediction)
                        dispatched=decision["executed"]
                    except (ValueError,RuntimeError):
                        # No raw text, oracle click or permissive repair.
                        pass
                    after=int(page.locator(".item .count").first.inner_text())
                    rows.append({
                        "arm":arm,
                        "generate_calls":counted.completed_generations-prior,
                        "format":classified.format_class,
                        "native_input_dispatched":dispatched,
                        "task_succeeded":bool(dispatched and after==baseline+1),
                    })
                finally:
                    context.close()
        finally:
            browser.close()
    result=make_private_report(rows,hardware)
    (root/REPORT).write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                             encoding="utf-8")
    # No model cache, original HTML screenshots or model responses in output.
    return result


if __name__=="__main__":
    print(json.dumps(run(),sort_keys=True),flush=True)
