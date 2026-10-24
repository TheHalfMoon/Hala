"""H88: replayable, source-pinned, PRIVATE Kaggle H87 research notebook.

Run as a private Kaggle GPU script only, with H71 dataset attached. One
research LoRA step may run; never treat a submitted job as a checkpoint.
The commercial/deployment boundary is unchanged.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA = "5a8396881b655736277b25255d7101f6c602213e"
SOURCE_REPO = "https://github.com/TheHalfMoon/Hala.git"
DATASET_REF = "abdulazizshehri/hala-h71-drag-train-10000-private"
REPO = Path("/kaggle/tmp/hala-h88-verified-code")
SCRATCH = Path("/kaggle/tmp/hala-h88-research")
CHECKPOINT = Path("/kaggle/working/hala-h88-research-checkpoint")
ATTEMPT = Path("/kaggle/working/hala-h88-research-attempt.json")
SCHEMA = "hala.h88.noncommercial-private-kaggle-research-attempt.v1"
PACKAGES = {"torch": "2.11.0", "transformers": "5.16.1",
            "peft": "0.20.0", "Pillow": "11.3.0"}


def require_private_research_environment(env: dict, *, is_kaggle_working: bool,
                                         cuda_devices: int) -> None:
    if (not env.get("KAGGLE_KERNEL_RUN_TYPE") or not is_kaggle_working
            or cuda_devices < 1):
        raise ValueError("genuine private Kaggle GPU research runtime required")
    if env.get("HALA_NONCOMMERCIAL_RESEARCH") not in (None, "1"):
        raise ValueError("incompatible research setting")


def safe_failure(stage: str, error: Exception) -> dict:
    """Never print arbitrary provider exception payloads or private inputs."""
    return {"schema": SCHEMA, "status": "FAILED_NO_QUALIFIED_MODEL_CHECKPOINT",
            "stage": stage, "failure_class": type(error).__name__,
            "source_sha": SOURCE_SHA, "model_trained_verified": False,
            "provider_private_readback_verified": False,
            "commercial_use": False}


def save_attempt(value: dict, path: Path = ATTEMPT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # One concise metadata-only receipt. No optimizer or private training text.
    if set(value) & {"kaggle_token", "prompt", "completion", "password"}:
        raise ValueError("private data disallowed in research attempt receipt")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), "utf8")
    tmp.replace(path)


def checked_process(*args: str, timeout: int = 120) -> None:
    subprocess.run(list(args), check=True, timeout=timeout,
                   capture_output=True, text=True)


def install_pinned_dependencies() -> dict:
    # Kaggle's stock torchao==0.10.0 was observed to crash PEFT==0.20.0 LoRA
    # dispatch after Qwen loaded. It is OPTIONAL for this FP16 train path.
    checked_process(sys.executable, "-m", "pip", "uninstall", "-y", "torchao",
                    timeout=90)
    checked_process(sys.executable, "-m", "pip", "install", "-q",
                    "transformers==5.16.1", "peft==0.20.0", "Pillow==11.3.0",
                    timeout=480)
    found = {name: importlib.metadata.version(name) for name in PACKAGES}
    if any(found[name].split("+")[0] != version
           for name, version in PACKAGES.items()):
        raise ValueError("pinned model runtime versions did not match")
    try:
        importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise ValueError("optional torchao is still installed and unqualified")
    return found


def fetch_exact_source() -> None:
    REPO.parent.mkdir(parents=True, exist_ok=True)
    if REPO.exists():
        raise FileExistsError("refusing an unqualified existing checkout")
    checked_process("git", "init", "-q", str(REPO), timeout=30)
    checked_process("git", "-C", str(REPO), "remote", "add", "origin", SOURCE_REPO)
    checked_process("git", "-C", str(REPO), "fetch", "--depth", "1",
                    "origin", SOURCE_SHA, timeout=180)
    checked_process("git", "-C", str(REPO), "checkout", "--detach", "-q",
                    "FETCH_HEAD", timeout=30)
    actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                     text=True, timeout=15).strip()
    if actual != SOURCE_SHA:
        raise ValueError("fetched research source SHA mismatch")


def main() -> None:
    stage = "KAGGLE_GPU_PREFLIGHT"
    record = {"schema": SCHEMA, "status": "STARTED", "stage": stage,
              "source_sha": SOURCE_SHA, "model_trained_verified": False,
              "provider_private_readback_verified": False, "commercial_use": False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise ValueError("not a Kaggle runtime")
        import torch
        require_private_research_environment(os.environ,
            is_kaggle_working=Path("/kaggle/working").is_dir(),
            cuda_devices=torch.cuda.device_count() if torch.cuda.is_available() else 0)
        record["gpu_count"] = torch.cuda.device_count()
        record["gpu_names"] = [torch.cuda.get_device_name(i)
                               for i in range(torch.cuda.device_count())]
        stage = "PINNED_DEPENDENCIES"
        record["stage"] = stage
        record["versions"] = install_pinned_dependencies()
        stage = "PINNED_SOURCE"
        record["stage"] = stage
        fetch_exact_source()
        sys.path.insert(0, str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        stage = "ORIGINAL_H71_QWEN_LORA_STEP"
        record["stage"] = stage
        save_attempt(record)
        from h87.kaggle_research_train import run
        receipt = run(repo=REPO, scratch=SCRATCH, output=CHECKPOINT)
        if (receipt.get("status") != "REAL_QWEN_LORA_STEP_CHECKPOINT_VERIFIED"
                or receipt.get("step") != 1
                or not receipt.get("checkpoint_sha256")):
            raise ValueError("no independently reloaded model optimizer checkpoint")
        record.update(status="REAL_QWEN_LORA_STEP_CHECKPOINT_VERIFIED",
                      model_trained_verified=True,
                      checkpoint_sha256=receipt["checkpoint_sha256"],
                      optimizer_step=1, stage="CHECKPOINT_RELOADED")
        save_attempt(record)
    except Exception as error:
        save_attempt(safe_failure(stage, error))
        raise


if __name__ == "__main__":
    main()
