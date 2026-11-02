"""H352 owner-private free-only paired pretrained Qwen pixel snap study.

One real first-click model generation per exposed BeaconBoard scene, two
fresh original-browser control/candidate contexts with identical screenshots.
ONLY low-entropy bilingual aggregate may leave private memory. Pixel policy
uses image bytes + native model action, never DOM, reference ID, oracle or gold.
No training, no independent blind OOD, no adapted model gain claim.
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

REPO = "https://github.com/TheHalfMoon/Hala.git"
BRANCH = "ps/h351-firstparty-qwen-miss-diagnosis"
PIN = "06596fdd70214be8f6729eb94615f569d9a7ec85"
BLOB = "6e8467e123c7f0fbac77ea553622ab8e659f9c16"
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REV = "89644892e4d85e24eaac8bacfd4f463576704203"
FLAG = "HALA_H352_PRIVATE_NONCOMMERCIAL_FREE_T4"
REPORT = "hala-h352-private-paired-snap-aggregate.json"
DIAG = "hala-h352-private-stage-only.json"
STATUS = "H352_PRIVATE_REAL_QWEN_EXPOSED_PIXEL_SNAP_PAIRED"
STAGES = frozenset(("ENV", "SOURCE", "DEPENDENCIES", "MODEL_LOAD",
                    "REAL_PAIRED_BROWSER", "REPORT"))
STAGE = "ENV"
SNAP_CODE = "H352_SNAP_SOURCE_PENDING"

def need(value):
    if not value:
        raise ValueError("H352 PRIVATE PAIRED MODEL REFUSED")

def verify_free(torch):
    working = Path("/kaggle/working")
    need(sys.platform == "linux" and working.is_dir()
         and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
         and os.environ.get(FLAG) == "1"
         and torch.cuda.is_available()
         and torch.cuda.device_count() in (1, 2)
         and all("T4" in torch.cuda.get_device_name(i)
                 for i in range(torch.cuda.device_count()))
         and shutil.disk_usage("/tmp").free >= 8 * 1024**3
         and not (working / REPORT).exists()
         and not (working / DIAG).exists())
    return {"free_owner_private_t4": True, "t4_count": torch.cuda.device_count()}

def pinned_source(dest):
    need(not dest.exists())
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "clone", "--depth", "1",
         "--single-branch", "--branch", BRANCH, REPO, str(dest)],
        check=True, capture_output=True, text=True, timeout=180)
    actual = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=15).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(dest), "status", "--porcelain"],
        check=True, capture_output=True, text=True, timeout=15).stdout.strip()
    need(actual == PIN and not dirty)
    html = dest / "h319" / "beaconboard.html"
    need(html.is_file() and not html.is_symlink()
         and not html.parent.is_symlink())
    raw = html.read_bytes()
    blob = sha1(b"blob " + str(len(raw)).encode("ascii")
                + b"\x00" + raw).hexdigest()
    need(blob == BLOB)
    return html

def load_screenshot_only_policy():
    need(type(SNAP_CODE) is str
         and SNAP_CODE.startswith('"""H352: screenshot-PX-ONLY'))
    namespace = {"__name__": "h352_snapshot_pixel_only"}
    exec(compile(SNAP_CODE, "h352/pixel_button_snap.py", "exec"),
         namespace)
    snap = namespace.get("snap_native_first_click")
    need(callable(snap))
    return snap

def score_dispatched_native(page, native, *, target):
    from h199.native_action_sft import parse_actions
    from h263.native_browser_executor import execute_one
    from h319.beaconboard_dynamic_research import IDS, inspect_all_stations
    action = parse_actions(native)["actions"][0]
    need(action["kind"] in ("click", "abstain"))
    if action["kind"] == "abstain":
        return False
    outcome = execute_one(page, native)
    need(outcome == {"kind": "click", "executed": True})
    state = inspect_all_stations(page)
    focused = page.evaluate("""() => {
        const e=document.activeElement;
        return e instanceof HTMLInputElement
            ? e.dataset.station : null;
    }""")
    expected = {id_: (id_ == target, "") for id_ in IDS}
    return state == expected and focused == target

def native_action_changed(original: str, revised: str) -> bool:
    """Compare parsed native actions, not JSON whitespace or key ordering."""
    from h199.native_action_sft import parse_actions
    before = parse_actions(original)["actions"]
    after = parse_actions(revised)["actions"]
    need(len(before) == len(after) == 1)
    return before != after


def measure_one_paired(browser, source, scene, processor, model, snap):
    from h319.beaconboard_dynamic_research import (
        IDS, _observed_png, _model_visible_rgb_digest,
        checked_source, exact_task, original_scene, inspect_all_stations,
    )
    from h323.preloaded_dynamic_visual_policy import PreloadedDynamicVisualPolicy
    from h199.native_action_sft import source_task_prompt
    from h319.beaconboard_dynamic_research import SCENES
    need(scene in SCENES and callable(snap))
    source, url = checked_source(source)
    task = exact_task(scene)
    target = IDS[scene[4]]
    policy = PreloadedDynamicVisualPolicy(processor, model, task)
    with original_scene(browser, source, url, scene) as page:
        need(inspect_all_stations(page) ==
             {s: (False, "") for s in IDS})
        png = _observed_png(page)
        rgb_digest = _model_visible_rgb_digest(png)
        native = policy(png, source_task_prompt(task))
        # Screenshot-ONLY transform receives NO task, ID, DOM, state, gold.
        revised = snap(png, native)
        need(rgb_digest == _model_visible_rgb_digest(_observed_png(page)))
        baseline = score_dispatched_native(page, native, target=target)
    with original_scene(browser, source, url, scene) as other_page:
        need(inspect_all_stations(other_page) ==
             {s: (False, "") for s in IDS})
        other_png = _observed_png(other_page)
        need(rgb_digest == _model_visible_rgb_digest(other_png))
        candidate = score_dispatched_native(other_page, revised,
                                            target=target)
    summary = policy.safe_contract_summary()
    need(type(summary) is dict
         and type(summary.get("model_generations_via_wrapped_generate")) is int
         and summary["model_generations_via_wrapped_generate"] == 1
         and type(summary.get("model_time_token_filters_installed")) is int
         and summary["model_time_token_filters_installed"] == 1
         and type(summary.get("distinct_original_pngs_observed")) is int
         and summary["distinct_original_pngs_observed"] == 1
         and summary["type_calls"] == 0
         and summary["pretrained_model_identity_attested"] is False
         and summary["real_owner_private_gpu_run_attested"] is False
         and summary["generalizable_hala_gain_proven"] is False)
    return {
        "language": scene[1], "baseline": baseline,
        "candidate": candidate,
        "changed": native_action_changed(native, revised),
    }

def aggregate_pairs(rows):
    need(type(rows) is list and len(rows) == 12)
    groups = {lang: {
        "scenes": 0, "baseline_success": 0, "pixel_snap_success": 0,
        "pixel_snap_changes": 0, "paired_wins": 0,
        "paired_losses": 0, "ties_both_success": 0, "ties_both_fail": 0,
    } for lang in ("en", "ar")}
    for row in rows:
        need(type(row) is dict and set(row) == {
            "language", "baseline", "candidate", "changed"}
             and row["language"] in groups
             and all(type(row[k]) is bool for k in (
                 "baseline", "candidate", "changed")))
        # Identical native actions in identical source contexts cannot
        # legitimately be counted as a calibration win or loss.
        # Reject browser nondeterminism instead of inflating snap gains.
        need(row["changed"] or row["baseline"] == row["candidate"])
        group = groups[row["language"]]
        group["scenes"] += 1
        group["baseline_success"] += int(row["baseline"])
        group["pixel_snap_success"] += int(row["candidate"])
        group["pixel_snap_changes"] += int(row["changed"])
        if row["baseline"] and row["candidate"]:
            group["ties_both_success"] += 1
        elif not row["baseline"] and not row["candidate"]:
            group["ties_both_fail"] += 1
        elif row["candidate"]:
            group["paired_wins"] += 1
        else:
            group["paired_losses"] += 1
    need(all(g["scenes"] == 6
             and g["paired_wins"] + g["paired_losses"]
                 + g["ties_both_success"] + g["ties_both_fail"] == 6
             and g["pixel_snap_success"]
                 == g["baseline_success"] + g["paired_wins"]
                    - g["paired_losses"]
             for g in groups.values()))
    return groups

def safe_report(rows, hardware):
    groups = aggregate_pairs(rows)
    need(type(hardware) is dict
         and set(hardware) == {"free_owner_private_t4", "t4_count"}
         and hardware["free_owner_private_t4"] is True
         and type(hardware["t4_count"]) is int
         and hardware["t4_count"] in (1, 2))
    return {
        "status": STATUS, "source_commit": PIN,
        "source_html_blob": BLOB,
        "model_id": MODEL, "model_revision": REV,
        "hardware": hardware, "by_language": groups,
        "claimed_wrapped_model_generations": 12,
        "paired_source_exposed_not_blind": True,
        "private_job_attestation_required": True,
        "official_train_admitted": False, "fresh_blind_ood": False,
        "adapted_model_gain_proven": False, "new_optimizer_steps": 0,
        "can_release": False, "raw_predictions_or_media_exported": False,
    }

def run():
    global STAGE
    import torch
    hw = verify_free(torch)
    scratch = Path("/tmp/hala-h352-private-real-paired")
    need(not scratch.exists())
    scratch.mkdir(mode=0o700)
    try:
        STAGE = "SOURCE"
        html = pinned_source(scratch / "source")
        sys.path.insert(0, str(html.parents[1]))
        STAGE = "DEPENDENCIES"
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from playwright.sync_api import sync_playwright
        from h268.free_t4_constrained_native import verify_exact_lmfe_installed_source
        from h319.beaconboard_dynamic_research import SCENES, checked_source
        need(transformers.__version__ == "5.16.1")
        verify_exact_lmfe_installed_source(
            importlib.metadata.distribution("lm-format-enforcer"))
        checked_source(html)
        snap = load_screenshot_only_policy()
        os.environ.update(
            HF_HOME=str(scratch / "hf"),
            HF_HUB_CACHE=str(scratch / "hf" / "hub"),
            HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        STAGE = "MODEL_LOAD"
        torch.manual_seed(352)
        processor = AutoProcessor.from_pretrained(
            MODEL, revision=REV, cache_dir=str(scratch / "hf" / "hub"),
            trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL, revision=REV, cache_dir=str(scratch / "hf" / "hub"),
            trust_remote_code=False, torch_dtype=torch.float16,
            attn_implementation="sdpa").to("cuda:0").eval()
        need(getattr(model.config, "model_type", None) == "qwen3_vl")
        STAGE = "REAL_PAIRED_BROWSER"
        order = (0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11)
        rows = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True, args=["--no-sandbox"])
            try:
                for index in order:
                    rows.append(measure_one_paired(
                        browser, html, SCENES[index],
                        processor, model, snap))
            finally:
                browser.close()
        STAGE = "REPORT"
        report = safe_report(rows, hw)
        raw = json.dumps(report, sort_keys=True, separators=(",", ":"))
        need(len(raw.encode("utf-8")) <= 4096)
        (Path("/kaggle/working") / REPORT).write_text(
            raw + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

def stage_only_report():
    return {
        "status": "H352_PRIVATE_STAGE_ONLY_NOT_MODEL_SCORE",
        "stage": STAGE if STAGE in STAGES else "ENV",
        "can_release": False, "fresh_blind_ood": False,
        "adapted_model_gain_proven": False,
        "official_train_admitted": False,
        "raw_private_data_saved": False,
    }

if __name__ == "__main__":
    try:
        run()
    except Exception:
        if (sys.platform == "linux" and Path("/kaggle/working").is_dir()
                and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
                and os.environ.get(FLAG) == "1"
                and not (Path("/kaggle/working") / DIAG).exists()):
            (Path("/kaggle/working") / DIAG).write_text(
                json.dumps(stage_only_report(), sort_keys=True) + "\n",
                encoding="utf-8")
        raise RuntimeError("H352 private model study stopped; only stage is available") from None
