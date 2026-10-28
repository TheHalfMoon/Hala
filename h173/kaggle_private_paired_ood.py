"""H173 private FREE T4 frozen OOD kernel; intentionally cannot start yet.

Before using on a brand-NEW 32/32 qualified OOD, the owner must pin
the gold-free ZIP SHA inside source and pass the final, exact Git commit
via HALA_H173_PINNED_SOURCE_SHA in the private Kaggle notebook environment.
No old H172 DEV media, H164 historical OOD or owner reference gold.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from h173.source_pin import pinned_source_commit

# Do NOT substitute an old DEV/consumed OOD or mutable branch.
# Source SHA is owner-pinned at runtime: cannot embed a commit into itself.
DATA = Path("/kaggle/input")
OUT = Path("/kaggle/working/hala-h173-private-paired-ood")
JOB = Path("/kaggle/working/h173-paired-ood-job.json")
WORK = Path("/kaggle/tmp/h173-immutable-source")


def command(*args):
    subprocess.run(
        args, capture_output=True, text=True, timeout=600, check=True
    )


def receipt(status, **values):
    doc = {
        "schema": "hala.h173.owner-private-free-t4-unseen-ood-job.v1",
        "status": status,
        "source_commit": os.environ.get(
            "HALA_H173_PINNED_SOURCE_SHA", "UNQUALIFIED"
        ),
        "optimizer_steps": 0,
        "real_gpu_calls": 0,
        "owner_gold_uploaded": False,
        "old_dev_ood_used": False,
        "paid_gpu": False,
        "weights_published": False,
        "flagship_35b_unchanged": True,
    }
    doc.update(values)
    JOB.write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )


def main():
    try:
        # Exact commit is supplied by the PRIVATE owner notebook environment.
        # This avoids the impossible self-hashing Git commit SHA cycle.
        sha = pinned_source_commit()
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise ValueError(
                "H173 new OOD source/model code must be immutably pinned first"
            )
        files = [x for x in DATA.rglob("*") if x.is_file()]
        new_ood = [x for x in files if x.name == "h173-ood.pkg"]
        language = [
            x for x in files
            if x.name == "adapter_model.safetensors"
            and "hala-h167-private-frozen-language-only-v1" in x.parts
        ]
        visual = [
            x for x in files
            if x.name == "adapter_model.safetensors"
            and "hala-h173-private-vision-only-v1" in x.parts
        ]
        if (
            len(files) != 6
            or len(new_ood) != 1 or len(language) != 1
            or len(visual) != 1
            or "hala-h173-private-new-ood-only-v1" not in new_ood[0].parts
            or {
                x.name for x in language[0].parent.iterdir()
            } != {"README.md", "adapter_config.json", "adapter_model.safetensors"}
            or {
                x.name for x in visual[0].parent.iterdir()
            } != {"adapter_config.json", "adapter_model.safetensors"}
        ):
            raise ValueError(
                "H173 private gold-free novel OOD, frozen H164/H171 only"
            )
        command(sys.executable, "-m", "pip", "uninstall", "-y", "torchao")
        command(
            sys.executable, "-m", "pip", "install", "-q",
            "transformers==5.17.0", "peft==0.21.2", "Pillow==11.3.0"
        )
        WORK.parent.mkdir(parents=True, exist_ok=True)
        command("git", "init", "-q", str(WORK))
        command(
            "git", "-C", str(WORK), "remote", "add",
            "origin", "https://github.com/TheHalfMoon/Hala.git"
        )
        command(
            "git", "-C", str(WORK), "fetch", "--depth", "1", "origin", sha
        )
        command("git", "-C", str(WORK), "checkout", "-q",
                "--detach", "FETCH_HEAD")
        identity = subprocess.check_output(
            ["git", "-C", str(WORK), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if identity != sha:
            raise ValueError("H173 exact immutable git source not loaded")
        sys.path.insert(0, str(WORK))
        os.environ.update(
            HALA_NONCOMMERCIAL_RESEARCH="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            DO_NOT_TRACK="1",
        )
        from h173.infer_gpu import OOD_ZIP_SHA, run
        if len(OOD_ZIP_SHA) != 64:
            raise ValueError("H173 qualified exact new OOD ZIP SHA not pinned")
        receipt("H173_32_PAIRED_UNSEEN_OOD_FROZEN_GPU_RUNNING")
        result = run(new_ood[0], language[0].parent, visual[0].parent, OUT)
        if result.get("real_gpu_calls") != 64 or result.get("optimizer_steps") != 0:
            raise ValueError("H173 missing 64 frozen T4 model inferences")
        receipt(
            "H173_PRIVATE_64_FROZEN_UNSEEN_OOD_T4_PROPOSALS_PASS",
            real_gpu_calls=64,
            peak_allocated_gib=result["peak_allocated_gib"],
        )
    except Exception as error:
        receipt(
            "H173_NEW_UNSEEN_OOD_GPU_NOT_QUALIFIED",
            failure_class=type(error).__name__,
            failure_message=str(error)[:240],
        )
        raise


if __name__ == "__main__":
    main()
