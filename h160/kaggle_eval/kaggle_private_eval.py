"""Owner-private H160 free-T4 32-call model inference-only experiment."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_COMMIT = "7a705c920e9fd4a3edf3084069dc764b43d7f087"
WORK = Path("/kaggle/tmp/hala-h160-crossapp-eval-source")
OUT = Path("/kaggle/working/hala-h160-stable-secondapp-inference")
RECEIPT = Path("/kaggle/working/h160-free-t4-inference-receipt.json")


def command(*args, timeout=180):
    result = subprocess.run(args, capture_output=True, text=True,
                            check=False, timeout=timeout)
    if result.returncode:
        raise RuntimeError("H160 pinned provenance setup failed")
    return result.stdout.strip()


def save(doc):
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RECEIPT.with_suffix(".tmp")
    temporary.write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n","utf8")
    temporary.replace(RECEIPT)


def find(root, name):
    matches = list(Path(root).rglob(name)) + list(Path(root).rglob(name+".zip"))
    matches = [p for p in dict.fromkeys(matches) if p.is_file() or p.is_dir()]
    if len(matches) != 1:
        raise ValueError("H160 requires exactly one frozen source/model input")
    return matches[0]


def private_mount(root, inputs):
    root = Path(root)
    folders = [x for x in root.iterdir() if x.is_dir()]
    if len(folders) != 1:
        raise ValueError("H160 one private dataset only")
    parent = folders[0].resolve()
    if (len(set(x.resolve() for x in inputs)) != 3 or
        any(not p.resolve().is_relative_to(parent) for p in inputs)):
        raise ValueError("H160 source/model paths outside one private mount")


def main():
    receipt = {
        "schema": "hala.h160.free-t4-frozen-paired-exploratory-inference.v1",
        "source_commit": SOURCE_COMMIT,
        "status": "H160_NOT_EVALUATED",
        "real_gpu_calls": 0,
        "optimizer_steps": 0, "h140_blind_calls": 0,
        "paid_gpu": False, "Hala35B_unchanged": True,
        "models_published": False,
    }
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or WORK.exists() or OUT.exists():
            raise ValueError("H160 new private kernel execution only")
        import torch
        if (not torch.cuda.is_available() or
            not any("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count()))):
            raise ValueError("H160 no paid GPU fallback")
        mount = Path("/kaggle/input")
        source, old, new = (
            find(mount, "h160-stable-original-todomvc-source"),
            find(mount, "h156-frozen-research-lora"),
            find(mount, "h157-frozen-research-lora"),
        )
        private_mount(mount, (source, old, new))
        receipt["status"] = "H160_PRIVATE_SOURCE_ADMITTED"
        save(receipt)
        command(sys.executable, "-m", "pip", "uninstall", "-y", "torchao", timeout=90)
        command(sys.executable, "-m", "pip", "install", "-q",
                "transformers==5.17.0", "peft==0.21.2", "Pillow==11.3.0",
                timeout=540)
        versions = {
            pkg: importlib.metadata.version(pkg).split("+")[0]
            for pkg in ("torch","transformers","peft","Pillow")
        }
        if versions != {
            "torch": "2.11.0", "transformers": "5.17.0",
            "peft": "0.21.2", "Pillow": "11.3.0"
        }:
            raise ValueError("H160 exact package versions not satisfied")
        WORK.parent.mkdir(parents=True, exist_ok=True)
        command("git", "init", "-q", str(WORK))
        command("git", "-C", str(WORK), "remote", "add", "origin",
                "https://github.com/TheHalfMoon/Hala.git")
        command("git", "-C", str(WORK), "fetch", "--depth", "1", "origin",
                SOURCE_COMMIT)
        command("git", "-C", str(WORK), "checkout", "--detach", "-q", "FETCH_HEAD")
        if command("git", "-C", str(WORK), "rev-parse", "HEAD") != SOURCE_COMMIT:
            raise ValueError("H160 expected frozen source code SHA")
        sys.path.insert(0, str(WORK))
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
                          HALA_NONCOMMERCIAL_RESEARCH="1")
        from h160.gpu_infer import run
        receipt["status"] = "H160_EXECUTING_32_REAL_FREE_GPU_INFERENCES"
        save(receipt)
        answer = run(source, old, new, OUT)
        if (answer.get("status") != "H160_COMPLETED_32_REAL_GPU_CALLS"
            or answer.get("gpu_calls") != 32
            or answer.get("optimizer_steps") != 0
            or answer.get("h140_blind_calls") != 0):
            raise ValueError("H160 actual GPU completion not verified")
        receipt.update(
            status="H160_COMPLETE_32_REAL_GPU_CALLS",
            real_gpu_calls=32,
            model_format_counts=answer["counts"],
        )
        save(receipt)
    except Exception as exc:
        receipt.update(status="H160_FAILED_NO_MODEL_SUCCESS_CLAIM",
                       failure_class=type(exc).__name__)
        save(receipt)
        raise


if __name__ == "__main__":
    main()
