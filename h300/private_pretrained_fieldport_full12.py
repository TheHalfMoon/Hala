"""H300: owner-private FREE noncommercial T4 pretrained two-step form pilot.

A NEW distinct firstparty EXPOSED original FieldPort study, not H293
training and not blind OOD. Never write screenshots, source gold,
native point coordinates, model text, weights or optimizer states.
Only fixed tiny low-entropy success / stage-only categorical receipts.
"""
from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys

SOURCE_REPO = "https://github.com/TheHalfMoon/Hala.git"
SOURCE_BRANCH = "ps/h298-preloaded-qwen-two-stage-form-policy"
SOURCE_SHA = "2babf8c91ffb63dd1123070449d1cbc240ab0859"
SOURCE_BLOB = "14bc9041fabf59d76adb0fc78b6a16d6d53c6e80"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
KERNEL_FLAG = "HALA_H300_PRIVATE_NONCOMMERCIAL_FREE_T4"
REPORT = "hala-h300-private-fieldport-two-step-aggregate.json"
DIAGNOSTIC_REPORT = "hala-h300-private-fieldport-stage-diagnostic.json"
STATUS = "H300_REAL_PRETRAINED_FIELDPORT_PUBLIC_EXPOSED_NOT_OOD"
DIAG_STATUS = "H300_PRIVATE_STAGE_ONLY_NOT_A_MODEL_SCORE"
STAGES = frozenset({"ENV","SOURCE","DEPENDENCIES","MODEL_LOAD",
                    "REAL_BROWSER","PRIVATE_AGGREGATE"})
STAGE = "ENV"
SCENE_INDICES = tuple(range(12))


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H300 private pretrained two-action pilot refused: " + reason)


def verify_free(torch) -> dict:
    w = Path("/kaggle/working")
    require(sys.platform == "linux" and w.is_dir()
            and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            and os.environ.get(KERNEL_FLAG) == "1"
            and torch.cuda.is_available()
            and torch.cuda.device_count() in (1,2)
            and all("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())),
            "owner-private noncommercial FREE T4 ONLY")
    require(shutil.disk_usage("/tmp").free >= 8 * 1024**3,
            "no ephemeral 8GiB model cache capacity")
    require(not (w / REPORT).exists()
            and not (w / DIAGNOSTIC_REPORT).exists(),
            "only fresh create-only private outputs allowed")
    return {"free_t4": True, "devices": torch.cuda.device_count()}


def clone_original_source(dest: Path) -> Path:
    require(not dest.exists(), "new ephemeral private Git checkout only")
    subprocess.run(
        ["git","-c","core.autocrlf=false","clone","--depth","1",
         "--single-branch","--branch",SOURCE_BRANCH,SOURCE_REPO,str(dest)],
        capture_output=True, check=True, text=True)
    head = subprocess.run(
        ["git","-C",str(dest),"rev-parse","HEAD"],
        capture_output=True,check=True,text=True).stdout.strip()
    dirty = subprocess.run(
        ["git","-C",str(dest),"status","--porcelain"],
        capture_output=True,check=True,text=True).stdout.strip()
    require(head == SOURCE_SHA and not dirty,
            "immutable original two-step model/source head changed")
    html=dest/"h296"/"fieldport.html"
    require(html.is_file() and not html.is_symlink(),
            "original author-owned form source absent")
    raw=html.read_bytes()
    blob=sha1(
        b"blob "+str(len(raw)).encode("ascii")+b"\x00"+raw
    ).hexdigest()
    require(blob == SOURCE_BLOB, "original source HTML Git blob drift")
    return html


def verified_report(rows: list[dict], hardware: dict) -> dict:
    require(type(hardware) is dict and set(hardware) == {"free_t4","devices"}
            and hardware["free_t4"] is True
            and type(hardware["devices"]) is int
            and hardware["devices"] in (1,2),
            "not the authorized real FREE T4")
    require(type(rows) is list and len(rows) == 12,
            "all 12 frozen original 6EN/6AR source scenes")
    for i,row in enumerate(rows):
        require(type(row) is dict and set(row)=={
            "language","generations","grammar_calls","screenshot_steps",
            "focus_clicks","type_calls","abstained","task_success",
            "all_four_source_fields_checked",
        } and row["language"] == ("en" if i<6 else "ar"),
                "unrecognized source row or PRIVATE raw field")
        n=row["generations"]
        require(type(n) is int and n in (1,2)
                and type(row["grammar_calls"]) is int
                and row["grammar_calls"] == n
                and type(row["screenshot_steps"]) is int
                and row["screenshot_steps"] == n
                and type(row["focus_clicks"]) is int
                and row["focus_clicks"] in (0,1)
                and type(row["type_calls"]) is int
                and row["type_calls"] in (0,1)
                and type(row["abstained"]) is bool
                and type(row["task_success"]) is bool
                and row["all_four_source_fields_checked"] is True,
                "invalid genuine two-step native task/count proof")
        require((n==1) == (row["focus_clicks"] == 0)
                and (row["type_calls"] <= row["focus_clicks"])
                and (not row["task_success"]
                     or (n == 2 and row["focus_clicks"] ==
                         row["type_calls"] == 1
                         and not row["abstained"]))
                and (not row["abstained"] or not row["task_success"]),
                "impossible two-step source task-state outcome")
    return {
        "status": STATUS,
        "source_commit": SOURCE_SHA,
        "original_html_git_blob": SOURCE_BLOB,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REV,
        "hardware": hardware,
        "source_research_scenes": 12,
        "real_model_visual_generations": sum(x["generations"] for x in rows),
        "model_time_grammar_generate_calls": sum(x["grammar_calls"] for x in rows),
        "rows": rows,
        "original_firstparty_task_successes": sum(int(x["task_success"]) for x in rows),
        "pretrained_no_optimizer_steps": 0,
        "source_public_previously_exposed": True,
        "official_train_dev_ood_admitted": False,
        "independent_blind_ood": False,
        "trained_adapter_gain_proven": False,
        "raw_model_output_media_points_gold_or_weights_saved": False,
    }


def safe_failure_receipt(stage: str, kind: str) -> dict:
    require(type(stage) is str and stage in STAGES,
            "bounded error stage only")
    allowed=frozenset({
        "ValueError","RuntimeError","OSError","ImportError",
        "ModuleNotFoundError","CalledProcessError","FileNotFoundError",
        "OutOfMemoryError",
    })
    require(type(kind) is str, "categorical exception class")
    return {
        "status": DIAG_STATUS,
        "failure_stage": stage,
        "exception_class": kind if kind in allowed else "OTHER_ERROR_CLASS",
        "source_commit": SOURCE_SHA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REV,
        "verified_model_skill_scores": False,
        "verified_optimizer_steps": 0,
        "official_train_dev_admitted": False,
        "blind_ood_proven": False,
        "raw_exception_text_model_outputs_media_points_saved": False,
    }


def write_private_failure(stage: str, error: BaseException) -> None:
    # Only owner PRIVATE Kaggle scratch. Never write to GitHub/user console.
    w=Path("/kaggle/working")
    if (sys.platform!="linux" or not w.is_dir()
            or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            or os.environ.get(KERNEL_FLAG)!="1"):
        return
    dest=w/DIAGNOSTIC_REPORT
    if not dest.exists():
        dest.write_text(
            json.dumps(safe_failure_receipt(stage,type(error).__name__),
                       separators=(",",":"),sort_keys=True)+"\n",
            encoding="utf-8")


def run() -> dict:
    global STAGE
    import torch
    hardware=verify_free(torch)
    private_root=Path("/tmp/hala-h300-original-fieldport-pretrained")
    require(not private_root.exists(), "one fresh private experiment only")
    private_root.mkdir(mode=0o700)
    try:
        STAGE="SOURCE"
        html=clone_original_source(private_root/"source")
        sys.path.insert(0,str(html.parents[1]))
        STAGE="DEPENDENCIES"
        import transformers
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        from playwright.sync_api import sync_playwright
        from h268.free_t4_constrained_native import verify_exact_lmfe_installed_source
        from h296.fieldport_sequence_research import SCENES, checked_source
        from h298.preloaded_form_two_stage_policy import run_fixed_original_source_research
        require(transformers.__version__=="5.16.1",
                "Transformers pinned production Qwen processor changed")
        verify_exact_lmfe_installed_source(
            importlib.metadata.distribution("lm-format-enforcer"))
        checked_source(html)
        os.environ.update(
            HF_HOME=str(private_root/"hf"),
            HF_HUB_CACHE=str(private_root/"hf"/"hub"),
            HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        STAGE="MODEL_LOAD"
        torch.manual_seed(299)
        cache=str(private_root/"hf"/"hub")
        processor=AutoProcessor.from_pretrained(
            MODEL_ID,revision=MODEL_REV,cache_dir=cache,
            trust_remote_code=False)
        model=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,revision=MODEL_REV,cache_dir=cache,
            trust_remote_code=False,torch_dtype=torch.float16,
            attn_implementation="sdpa").to("cuda:0").eval()
        require(getattr(model.config,"model_type",None)=="qwen3_vl",
                "wrong frozen pretrained visual model type")
        STAGE="REAL_BROWSER"
        rows=[]
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
            try:
                for index in SCENE_INDICES:
                    scene=SCENES[index]
                    observed=run_fixed_original_source_research(
                        browser,html,processor,model,scene)
                    ctr=observed["preloaded_model_contract"]
                    require(observed["native_actions"] in (1,2)
                            and ctr["model_generations_via_wrapped_generate"]
                                == observed["native_actions"]
                            and ctr["model_time_token_filters_installed"]
                                == observed["native_actions"]
                            and ctr["distinct_original_pngs_observed"]
                                == observed["native_actions"]
                            and not ctr["pretrained_model_identity_attested"]
                            and not ctr["real_owner_private_gpu_run_attested"]
                            and not ctr["native_source_task_success_attested"],
                            "local source-policy contract count drift")
                    rows.append({
                        "language": scene[1],
                        "generations": observed["native_actions"],
                        "grammar_calls": ctr["model_time_token_filters_installed"],
                        "screenshot_steps": ctr["distinct_original_pngs_observed"],
                        "focus_clicks": ctr["focus_clicks"],
                        "type_calls": ctr["type_calls"],
                        "abstained": ctr["abstained"],
                        "task_success": observed["source_original_gui_state_success"],
                        "all_four_source_fields_checked": True,
                    })
            finally:
                browser.close()
        STAGE="PRIVATE_AGGREGATE"
        receipt=verified_report(rows,hardware)
        (Path("/kaggle/working")/REPORT).write_text(
            json.dumps(receipt,sort_keys=True,separators=(",",":"))+"\n",
            encoding="utf-8")
        return receipt
    finally:
        shutil.rmtree(private_root,ignore_errors=True)


if __name__=="__main__":
    try:
        run()
    except Exception as error:
        write_private_failure(STAGE,error)
        raise RuntimeError(
            "H300 private pretrained FieldPort trial stopped; "
            "inspect ONLY whitelisted private stage report"
        ) from None
