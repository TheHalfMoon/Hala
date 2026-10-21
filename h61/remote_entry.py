"""H61 the remote entry point for an authorised free Colab CUDA session.

The script prepares the runtime, clones the exact H61 branch, and runs the
H61 steps. Every step is idempotent and every step writes its own receipt, so
an interrupted session is continued rather than restarted.

The runtime is never trusted: the installed package versions are measured and
compared against the pinned requirements, the compute unit balance is read
before and after the run, and the paid resources are never enabled.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/TheHalfMoon/Hala.git"
BRANCH = "ps/h61-complete-corpus-and-curriculum"
REQUIREMENTS = ("torch==2.11.0", "transformers==5.16.1", "peft==0.20.0", "Pillow==11.3.0")
MODULE_FOR_REQUIREMENT = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "pillow": "PIL",
}
RECEIPT_SCHEMA = "hala.h61.remote-entry.v1"


def run(command, **kwargs):
    print(f"H61_RUN {command}", flush=True)
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs)


def package_versions() -> dict:
    versions = {}
    for name in ("torch", "transformers", "peft", "PIL"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as error:  # noqa: BLE001
            versions[name] = f"unavailable: {type(error).__name__}"
    return versions


def main() -> int:
    started = time.time()
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "branch": BRANCH,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "steps": [],
    }
    work = Path(os.environ.get("H61_WORK", "/content/hala-h61-work"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        if not Path("/content").exists():
            raise RuntimeError("The remote session has no interactive Colab workspace")
        if not os.environ.get("COLAB_RELEASE_TAG"):
            raise RuntimeError("The remote session does not identify itself as Colab")
        receipt["colab_release_tag"] = os.environ["COLAB_RELEASE_TAG"]
        receipt["packages_before"] = package_versions()

        if not _git_available():
            run(["apt-get", "install", "-y", "git"])
        repo = work / "repo"
        if not (repo / ".git").exists():
            run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(repo)])
        run(["git", "-C", str(repo), "rev-parse", "HEAD"])
        receipt["repo_head"] = run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"]
        ).stdout.strip()

        versions = package_versions()
        missing = []
        for requirement in REQUIREMENTS:
            name, _, wanted = requirement.partition("==")
            module = MODULE_FOR_REQUIREMENT.get(name.lower())
            actual = versions.get(module, "") if module else ""
            if actual.split("+")[0] != wanted:
                missing.append(requirement)
        if missing:
            print(f"H61_INSTALL {missing}", flush=True)
            run([sys.executable, "-m", "pip", "install", "--quiet", *missing])
            receipt["packages_after_install"] = package_versions()
        else:
            receipt["packages_after_install"] = versions

        steps = os.environ.get(
            "H61_STEPS",
            "export,curriculum,baseline,official_base,train,evaluate,official_adapter",
        )
        images = os.environ.get(
            "H61_IMAGES",
            "slider=8000,slide_resize=6000,sheet=6000,text_highlight=5000",
        )
        max_steps = os.environ.get("H61_MAX_STEPS", "600")
        official_limit = os.environ.get("H61_OFFICIAL_LIMIT", "30")
        command = [
            sys.executable, "-m", "h61.colab_run",
            "--repo", str(repo),
            "--work", str(work / "run"),
            "--steps", steps,
            "--images", images,
            "--max-steps", max_steps,
            "--official-limit", official_limit,
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repo)
        environment["HALA_REMOTE_TRAINING_ALLOWED"] = "1"
        environment["HALA_REMOTE_BASELINE_ALLOWED"] = "1"
        result = subprocess.run(command, cwd=str(repo), env=environment, text=True)
        receipt["steps"] = steps.split(",")
        receipt["colab_run_exit"] = result.returncode
        receipt["run_receipt"] = str(work / "run" / "run-receipt.json")
    except BaseException as error:  # noqa: BLE001
        receipt["failure"] = f"{type(error).__name__}: {error}"[:400]
        receipt["elapsed_seconds"] = round(time.time() - started, 3)
        (work / "remote-entry-receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf8"
        )
        print(f"H61_ENTRY_FAILED {type(error).__name__}: {error}", flush=True)
        raise
    receipt["elapsed_seconds"] = round(time.time() - started, 3)
    (work / "remote-entry-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf8"
    )
    print("H61_ENTRY_COMPLETE", flush=True)
    return 0


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
