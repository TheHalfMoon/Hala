"""H94 owner-private Kaggle T4 LoRA study. No release, no paid compute."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA = "c6aa0e6bef4e87638eaaf22af15af77d875ddd29"
SOURCE_URL = "https://github.com/TheHalfMoon/Hala.git"
REPO = Path("/kaggle/tmp/hala-h94-source")
SCRATCH = Path("/kaggle/tmp/hala-h94-study")
OUTPUT = Path("/kaggle/working/hala-h94-private-checkpoint")
ATTEMPT = Path("/kaggle/working/hala-h94-attempt.json")


def record(value):
    ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    tmp = ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), "utf8")
    tmp.replace(ATTEMPT)


def command(*args, timeout=150):
    subprocess.run(list(args), check=True, capture_output=True,
                   text=True, timeout=timeout)


def main():
    state = {
        "schema": "hala.h94.private-gpu-attempt.v1",
        "source_sha": SOURCE_SHA,
        "status": "STARTED", "stage": "PREFLIGHT",
        "commercial_use": False, "public_model_released": False,
        "checkpoint_independently_verified": False,
        "new_optimizer_steps_verified": 0
    }
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("genuine private Kaggle notebook required")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("real Kaggle T4 CUDA GPU required")
        state["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        state["stage"] = "EXACT_DEPENDENCIES"
        command(sys.executable, "-m", "pip", "uninstall", "-y", "torchao")
        command(sys.executable, "-m", "pip", "install", "-q",
                "transformers==5.16.1", "peft==0.20.0", "Pillow==11.3.0",
                timeout=480)
        expected = {"torch": "2.11.0", "transformers": "5.16.1",
                    "peft": "0.20.0", "Pillow": "11.3.0"}
        actual = {k: importlib.metadata.version(k) for k in expected}
        if any(actual[k].split("+")[0] != v for k, v in expected.items()):
            raise ValueError("pinned real Qwen training dependencies mismatch")
        try:
            importlib.metadata.version("torchao")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise ValueError("incompatible optional torchao still installed")
        state["stage"] = "EXACT_PUBLIC_GIT_SOURCE"
        if REPO.exists():
            raise ValueError("unverified pre-existing source checkout")
        REPO.parent.mkdir(parents=True, exist_ok=True)
        command("git", "init", "-q", str(REPO))
        command("git", "-C", str(REPO), "remote", "add", "origin", SOURCE_URL)
        command("git", "-C", str(REPO), "fetch", "--depth", "1",
                "origin", SOURCE_SHA, timeout=180)
        command("git", "-C", str(REPO), "checkout", "--detach", "-q", "FETCH_HEAD")
        actual_sha = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        if actual_sha != SOURCE_SHA:
            raise ValueError("Qwen source commit identity mismatch")
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",
                          DO_NOT_TRACK="1")
        sys.path.insert(0, str(REPO))
        state["stage"] = "H92_SEEDED_64_REAL_QWEN_UPDATES"
        record(state)
        from h94.research_train import run
        receipt = run(repo=REPO, scratch=SCRATCH, output=OUTPUT)
        if (receipt.get("status") != "H94_REAL_64_STEP_CHECKPOINT_WRITTEN"
                or receipt.get("steps") != 64
                or not receipt.get("checkpoint_sha256")):
            raise ValueError("no H94 final immutable optimizer checkpoint")
        state.update(
            status="H94_REAL_64_STEP_CHECKPOINT_WRITTEN",
            stage="FINAL_CHECKPOINT_WRITTEN",
            new_optimizer_steps_verified=64,
            checkpoint_sha256=receipt["checkpoint_sha256"],
            checkpoint_independently_verified=False
        )
        record(state)
        print("H94_GENUINE_QWEN_PRIVATE_RESEARCH_CHECKPOINT_WRITTEN", flush=True)
    except Exception as error:
        state.update(
            status="FAILED_NO_QUALIFIED_64_STEP_RESULT",
            failure_class=type(error).__name__
        )
        record(state)
        print("H94_STUDY_STOPPED", type(error).__name__, flush=True)
        raise


if __name__ == "__main__":
    main()
