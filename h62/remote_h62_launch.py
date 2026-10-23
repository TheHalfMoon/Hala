"""H62 the pinned-source remote entry point for the H61 diagnostic run.

This script is executed INSIDE an authorised free Colab Linux CUDA session.
It is deliberately self-contained: it cannot import anything from the Hala
repository before the repository has been fetched and verified, because the
repository is the thing being verified.

The execution contract is fail closed:

* the runtime must prove it is a genuine Colab session with a `/content`
  workspace, and an operator must confirm that the session is on the free
  tier and not a paid or Pro runtime;
* the source is fetched by the exact 40-character commit SHA, never by a
  branch name, so a branch that moved during a long wait cannot change what
  is trained;
* the fetched HEAD is compared with the requested SHA and a mismatch aborts
  before any model weight is downloaded;
* the package versions are pinned and measured;
* a nonzero exit from the H61 pipeline is a failure of this entry point, and
  a failure receipt is written before the error propagates.

A success receipt from this script therefore means: the pipeline ran to
completion on the pinned source, and the pipeline's own receipts carry the
training evidence. It does not by itself mean the model improved.
"""
from __future__ import annotations

import json
from importlib import metadata
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/TheHalfMoon/Hala.git"
REQUIREMENTS = ("torch==2.11.0", "transformers==5.16.1", "peft==0.20.0", "Pillow==11.3.0")
MODULE_FOR_REQUIREMENT = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "pillow": "PIL",
}
RECEIPT_SCHEMA = "hala.h62.remote-launch.v1"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def fail(message: str) -> "RuntimeError":
    return RuntimeError(message)


def require_free_colab_runtime() -> dict:
    """Refuse to run anywhere that is not a confirmed free Colab session.

    The code cannot read the account's billing plan. It can only require the
    provider marker and an explicit operator confirmation, and it records
    both in the receipt so the claim is auditable rather than assumed.
    """
    if not Path("/content").exists():
        raise fail("the remote session has no interactive Colab workspace")
    release = os.environ.get("COLAB_RELEASE_TAG")
    if not release:
        raise fail("the remote session does not identify itself as Colab")
    if os.environ.get("HALA_FREE_COLAB_RESEARCH") != "1":
        raise fail("an operator must confirm the free, non-paid Colab runtime")
    if os.environ.get("HALA_NO_PAID_RUNTIME_CONFIRMED") != "1":
        raise fail("an operator must confirm that no paid runtime is enabled")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise fail("this entry point is for a Colab session, not a Kaggle notebook")
    return {"colab_release_tag": release, "free_runtime_confirmed": True}


def fetch_pinned_source(repo_dir: Path, pinned_sha: str) -> str:
    """Fetch the exact commit and check it out detached.

    The fetch is by SHA and not by branch. The H61 run waited hours for a
    free GPU; a branch that advanced during that wait must not silently
    change the source of the training run.
    """
    if not SHA_PATTERN.match(pinned_sha or ""):
        raise fail("an explicit 40 character pinned source SHA is required")
    repo_dir.mkdir(parents=True, exist_ok=True)
    if not (repo_dir / ".git").exists():
        subprocess.run(["git", "init", "-q", str(repo_dir)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo_dir), "remote", "add", "origin", REPO_URL],
                       check=True, capture_output=True)
    fetched = subprocess.run(
        ["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", pinned_sha],
        capture_output=True, text=True,
    )
    if fetched.returncode != 0:
        raise fail(f"the pinned source could not be fetched: {fetched.stderr[-300:]}")
    checkout = subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "--detach", "FETCH_HEAD"],
        capture_output=True, text=True,
    )
    if checkout.returncode != 0:
        raise fail(f"the pinned source could not be checked out: {checkout.stderr[-300:]}")
    head = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != pinned_sha:
        raise fail(f"the remote checkout is {head} and not the pinned {pinned_sha}")
    return head


def package_versions() -> dict:
    """Inspect installed distributions without importing/caching modules.

    Importing transformers/peft *before* pip pinning leaves their older
    __version__ values in sys.modules even after a successful install.
    Metadata reads the actual installed distribution, then a fresh process
    verifies imports. This was an observed false failure on free T4.
    """
    versions = {}
    for module, dist in (("torch", "torch"), ("transformers", "transformers"),
                         ("peft", "peft"), ("PIL", "Pillow")):
        try:
            versions[module] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[module] = "unavailable: PackageNotFoundError"
    return versions


def ensure_packages() -> dict:
    """Install the pinned versions only when they are not already present."""
    versions = package_versions()
    missing = []
    for requirement in REQUIREMENTS:
        name, _, wanted = requirement.partition("==")
        module = MODULE_FOR_REQUIREMENT.get(name.lower())
        actual = versions.get(module, "") if module else ""
        if actual.split("+")[0] != wanted:
            missing.append(requirement)
    if missing:
        # Colab has Torch and the model's dependency graph already installed.
        # Installing two pure Python package pins WITH dependencies can try to
        # re-resolve massive GPU libraries during a short-lived T4 allocation.
        # We measured --no-deps on a free Colab CPU session on 2026-10-23:
        # 17.327 s to pin transformers 5.16.1 + peft 0.20.0, followed by
        # successful imports. Never infer compatibility from pip exit alone.
        print(f"H62_INSTALL_PIN_ONLY {missing}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "--quiet", "--no-deps", *missing],
            check=True, capture_output=True, text=True,
            timeout=180,
        )
        versions = package_versions()
    for requirement in REQUIREMENTS:
        name, _, wanted = requirement.partition("==")
        module = MODULE_FOR_REQUIREMENT[name.lower()]
        if versions.get(module, "").split("+")[0] != wanted:
            raise fail(f"package pin failed validation: {name}=={wanted}")
    # This check is intentionally performed BEFORE any model download.
    # A missing transitive dependency is a failed preflight, not grounds
    # to trigger an uncontrolled, costly pip reinstall on a GPU session.
    import_script = ("from transformers import AutoProcessor, "
                     "Qwen3VLForConditionalGeneration; "
                     "from peft import LoraConfig, PeftModel")
    checked = subprocess.run([sys.executable, "-c", import_script],
                             capture_output=True, text=True, timeout=120)
    if checked.returncode != 0:
        raise fail("pinned training imports unavailable in fresh Python process")
    return versions


def qualify_stage_source(stage_sha: str, source_sha: str) -> str:
    """Require separately pinned data provenance whenever CPU stage is provided."""
    if bool(stage_sha) != bool(source_sha):
        raise fail("staged TRAIN data requires BOTH ZIP digest and original source Git SHA")
    if stage_sha and (re.fullmatch(r"[0-9a-f]{64}", stage_sha) is None
                      or SHA_PATTERN.fullmatch(source_sha) is None):
        raise fail("invalid H67 stage digest or original source Git SHA")
    return source_sha


def qualify_h75_audit(stage_sha: str, steps: str, max_steps: str,
                       audit_records: str) -> int:
    """Validate the explicit bounded processor audit BEFORE dependency installs."""
    if audit_records == "0":
        return 0
    if (not audit_records.isascii() or not audit_records.isdigit()
            or not 1 <= int(audit_records) <= 64
            or not max_steps.isascii() or not max_steps.isdigit()
            or not 1 <= int(max_steps) <= 3 or not stage_sha
            or "train" not in [s.strip() for s in steps.split(",")]):
        raise fail("H75 bounded processor audit only permitted on staged <=3-step TRAIN")
    return int(audit_records)


def main() -> int:
    started = time.time()
    work = Path(os.environ.get("H62_WORK", "/content/hala-h62-work"))
    work.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "work": str(work),
        "paid_compute_units_consumed": 0,
    }
    try:
        receipt["runtime"] = require_free_colab_runtime()
        pinned = os.environ.get("HALA_PINNED_SOURCE_SHA", "")
        receipt["pinned_source_sha"] = pinned
        receipt["source_branch"] = os.environ.get("HALA_REMOTE_SOURCE_BRANCH", "")
        repo = work / "repo"
        receipt["repo_head"] = fetch_pinned_source(repo, pinned)
        receipt["repo_source_sha_verified"] = True
        stage_sha = os.environ.get("H67_STAGE_SHA256", "")
        stage_source_sha = qualify_stage_source(
            stage_sha, os.environ.get("H67_STAGE_SOURCE_SHA", ""))
        if stage_sha:
            receipt["h67_source_git_sha"] = stage_source_sha
            receipt["h67_zip_sha256"] = stage_sha
        steps = os.environ.get("H61_STEPS", "export,firstparty,curriculum,baseline,train,evaluate")
        max_steps = os.environ.get("H61_MAX_STEPS", "600")
        audit_records = os.environ.get("H61_PROCESSOR_AUDIT_RECORDS", "0")
        audit_count = qualify_h75_audit(stage_sha, steps, max_steps, audit_records)
        if audit_count:
            receipt["processor_audit_records"] = audit_count
        # This entry script itself was uploaded from the host. Only import
        # training/staging code from the SHA-verified freshly fetched checkout.
        sys.path.insert(0, str(repo))
        receipt["packages_before"] = package_versions()
        receipt["packages_after_install"] = ensure_packages()

        if stage_sha:
            # CPU-created TRAIN bundle is uploaded before this T4 process.
            # Never repeat full data preparation inside a short-lived GPU.
            if any(s.strip() in {"export", "firstparty", "curriculum", "baseline"}
                   for s in steps.split(",")):
                raise fail("CPU-only steps forbidden with an H67 staged dataset")
            from h67.stage import restore_stage
            receipt["h67_stage"] = restore_stage(
                Path("/content/hala-h67-stage.zip"), stage_sha, work / "run",
                stage_source_sha, rederive=True)
        images = os.environ.get("H61_IMAGES", "slider=3500,slide_resize=2500,sheet=2500,text_highlight=2500")
        official_limit = os.environ.get("H61_OFFICIAL_LIMIT", "30")
        family_targets = os.environ.get(
            "H61_FAMILY_TARGETS", '{"h61_firstparty_bilingual_drag_train": 2000}')
        family_targets_path = work / "family-targets.json"
        family_targets_path.write_text(family_targets, encoding="utf8")

        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repo)
        environment["HALA_REMOTE_TRAINING_ALLOWED"] = "1"
        environment["HALA_REMOTE_BASELINE_ALLOWED"] = "1"
        environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
        environment["DO_NOT_TRACK"] = "1"
        command = [
            sys.executable, "-m", "h61.colab_run",
            "--repo", str(repo),
            "--work", str(work / "run"),
            "--steps", steps,
            "--images", images,
            "--max-steps", max_steps,
            "--official-limit", official_limit,
            "--family-targets-file", str(family_targets_path),
        ]
        if audit_count:
            command.extend(["--processor-audit-records", str(audit_count)])
        receipt["command"] = command
        result = subprocess.run(command, cwd=str(repo), env=environment, text=True)
        receipt["colab_run_exit"] = result.returncode
        if result.returncode != 0:
            raise fail(f"the H61 pipeline child failed with exit {result.returncode}")
        receipt["run_receipt"] = str(work / "run" / "run-receipt.json")
    except BaseException as error:  # noqa: BLE001
        receipt["failure"] = f"{type(error).__name__}: {error}"[:400]
        receipt["elapsed_seconds"] = round(time.time() - started, 3)
        (work / "h62-remote-launch-receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf8")
        print(f"H62_ENTRY_FAILED {type(error).__name__}: {error}", flush=True)
        raise
    receipt["elapsed_seconds"] = round(time.time() - started, 3)
    receipt["status"] = "completed"
    (work / "h62-remote-launch-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf8")
    print("H62_ENTRY_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
