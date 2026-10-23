"""H73: verify owner-private H71 TRAIN in a private Kaggle CPU-only research notebook.

This script NEVER trains or downloads model weights. Kaggle terms restrict
services to personal non-commercial use: commercial model training is NOT
authorized by this admission test. There is no GPU acceleration or model
evaluation. Outputs are a small source-integrity receipt only, no raw data.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_URL = "https://github.com/TheHalfMoon/Hala.git"
H71_CODE_SHA = "acd694fbae10752ff4ee3fe6cc3e52142ab7f0fe"
DATASET_REF = "abdulazizshehri/hala-h71-drag-train-10000-private"
ARCHIVE = Path("/kaggle/input/hala-h71-drag-train-10000-private/hala-h71-train-10000.hala.bin")
ARCHIVE_BYTES = 1_759_484_545
ARCHIVE_SHA = "4a8e602c785c2b90a60e49f78381a7af983401cb99b16c53f3df69e930cc073a"
TRAIN_SHA = "9888ac9d3b5ec7b8efe677287d54dc350c222a1804619e6217a84ec8f95eddcd"
EXPECTED_MEMBERS = 15224
OUTPUT = Path("/kaggle/working/hala-h73-cpu-admission.json")
SCHEMA = "hala.h73.kaggle-cpu-source-admission.v1"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_attached_train(root: Path = Path("/kaggle/input")) -> Path:
    """Locate only the attached owner/slug H71 binary, not arbitrary inputs.

    Modern Kaggle mounts use /kaggle/input/datasets/<owner>/<slug>.
    Older kernels flatten the dataset slug at /kaggle/input/<slug>.
    Any candidate is still required to pass the full immutable SHA-256.
    """
    owner = "abdulazizshehri"
    slug = "hala-h71-drag-train-10000-private"
    name = "hala-h71-train-10000.hala.bin"
    candidates = [
        root / "datasets" / owner / slug / name,
        root / slug / name,
        root / owner / slug / name,
    ]
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            return path
    raise ValueError("exact attached owner-private TRAIN binary not mounted")


def verify_input(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("private attached Kaggle H71 TRAIN binary missing or linked")
    size = path.stat().st_size
    if size != ARCHIVE_BYTES:
        raise ValueError("attached original source binary has wrong length")
    sha = file_hash(path)
    if sha != ARCHIVE_SHA:
        raise ValueError("attached original TRAIN archive SHA256 mismatch")
    return {"archive_sha256": sha, "archive_bytes": size, "dataset_ref": DATASET_REF}


def require_cpu():
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ValueError("must be an actual Kaggle CPU notebook session")
    if shutil.which("nvidia-smi"):
        state = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
        if state.returncode == 0 and "GPU " in state.stdout:
            raise ValueError("GPU detected: CPU-only notebook may not run on accelerator")


def fetch_pinned_source(target: Path):
    subprocess.run(["git", "init", "-q", str(target)], check=True, timeout=25, capture_output=True)
    subprocess.run(["git", "-C", str(target), "remote", "add", "origin", REPO_URL],
                   check=True, timeout=15, capture_output=True)
    subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", H71_CODE_SHA],
                   check=True, timeout=120, capture_output=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", "-q", "FETCH_HEAD"],
                   check=True, timeout=30, capture_output=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"],
                                     text=True, timeout=10).strip()
    if actual != H71_CODE_SHA:
        raise ValueError("fetched verification source SHA differs from pinned H71")
    return actual


def verify_source(root: Path, path: Path):
    repo = root / "source"
    sha = fetch_pinned_source(repo)
    sys.path.insert(0, str(repo))
    from h67.stage import restore_stage
    qualified = restore_stage(path, ARCHIVE_SHA, root / "qualified", sha, rederive=True)
    if (qualified.get("status") != "RESTORED_TRAIN_STAGE"
            or qualified.get("dataset_sha256") != TRAIN_SHA
            or qualified.get("sha256") != ARCHIVE_SHA
            or qualified.get("files") != EXPECTED_MEMBERS):
        raise ValueError("pinned H67 TRAIN source restore failed integrity admission")
    return {"source_git_sha": sha, "dataset_sha256": TRAIN_SHA,
            "archive_members": qualified["files"], "h67": qualified["status"]}


def write_receipt(value):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_name("." + OUTPUT.name + ".tmp")
    if temp.exists():
        raise FileExistsError("refusing an existing uncommitted admission receipt")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, OUTPUT)


def main():
    started = time.monotonic()
    status = {"schema": SCHEMA, "execution": "CPU_ONLY",
              "gpu_training_executed": False, "private_input": True,
              "model_weights_downloaded": False, "admission": "NOT_QUALIFIED"}
    try:
        require_cpu()
        attached = locate_attached_train()
        status.update(verify_input(attached))
        scratch_parent = Path("/kaggle/temp")
        if not scratch_parent.is_dir():
            raise ValueError("Kaggle scratch workspace unavailable")
        if shutil.disk_usage(scratch_parent).free < 5 * 1024**3:
            raise OSError("insufficient scratch disk for full source rederivation")
        with tempfile.TemporaryDirectory(prefix="hala-h73-verify-", dir=scratch_parent) as temp:
            status.update(verify_source(Path(temp), attached))
        status["admission"] = "VERIFIED_CPU_TRAIN_SOURCE"
    except Exception as error:
        status["failure"] = f"{type(error).__name__}: {error}"[:260]
        # A metadata-only bounded locator distinguishes provider mount
        # names from missing credentials. Never output file content or
        # mount absolute paths to the notebook status.
        root = Path("/kaggle/input")
        if root.is_dir():
            status["attached_dataset_names"] = sorted(
                child.name for child in list(root.iterdir())[:32])
            dataset_mount = root / "datasets" / "abdulazizshehri"
            if dataset_mount.is_dir():
                status["attached_owner_names"] = sorted(
                    child.name for child in list(dataset_mount.iterdir())[:32])
                desired = dataset_mount / "hala-h71-drag-train-10000-private"
                if desired.is_dir():
                    status["attached_expected_entries"] = sorted(
                        child.name for child in list(desired.iterdir())[:32])
            status["expected_archive_exists"] = ARCHIVE.exists()
    status["elapsed_seconds"] = round(time.monotonic() - started, 3)
    write_receipt(status)
    print("H73_ADMISSION=" + json.dumps(status, sort_keys=True), flush=True)
    return 0 if status["admission"] == "VERIFIED_CPU_TRAIN_SOURCE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
