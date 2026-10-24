"""Run H95 on PRIVATE noncommercial Kaggle: no training, strict paired EVAL."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA = "21bddd6a30aaff536949257db60390e23f1533e7"
SOURCE_REPO = "https://github.com/TheHalfMoon/Hala.git"
REPO = Path("/kaggle/tmp/hala-h95-source")
SCRATCH = Path("/kaggle/tmp/hala-h95-official-new-eval")
OUTPUT = Path("/kaggle/working/hala-h95-private-eval")
ATTEMPT = Path("/kaggle/working/hala-h95-attempt.json")


def record(payload):
    ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    tmp = ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), "utf8")
    tmp.replace(ATTEMPT)


def checked(*args, timeout=120):
    subprocess.run(list(args), capture_output=True, text=True,
                   check=True, timeout=timeout)


def main():
    data = {
        "schema": "hala.h95.private-paired-new-eval-attempt.v1",
        "status": "STARTED", "stage": "GPU_PREFLIGHT",
        "source_sha": SOURCE_SHA,
        "training_executed": False,
        "public_model_released": False, "commercial_use": False,
    }
    try:
        if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
                or not Path("/kaggle/working").is_dir()):
            raise ValueError("owner-private Kaggle notebook required")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise ValueError("genuine free T4 needed for paired research evaluation")
        data["stage"] = "PINNED_PACKAGES"
        checked(sys.executable, "-m", "pip", "uninstall", "-y", "torchao", timeout=90)
        checked(sys.executable, "-m", "pip", "install", "-q",
                "transformers==5.16.1", "peft==0.20.0", "Pillow==11.3.0",
                timeout=480)
        expected = {"torch": "2.11.0", "transformers": "5.16.1",
                    "peft": "0.20.0", "Pillow": "11.3.0"}
        found = {name: importlib.metadata.version(name) for name in expected}
        if any(found[name].split("+")[0] != expected[name] for name in expected):
            raise ValueError("unqualified paired GPU model environment")
        try:
            importlib.metadata.version("torchao")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise ValueError("incompatible optional torchao still installed")
        data["stage"] = "PINNED_GIT_SOURCE"
        if REPO.exists():
            raise ValueError("refusing unqualified previous source checkout")
        REPO.parent.mkdir(parents=True, exist_ok=True)
        checked("git", "init", "-q", str(REPO), timeout=30)
        checked("git", "-C", str(REPO), "remote", "add", "origin", SOURCE_REPO)
        checked("git", "-C", str(REPO), "fetch", "--depth", "1",
                "origin", SOURCE_SHA, timeout=180)
        checked("git", "-C", str(REPO), "checkout", "--detach", "-q", "FETCH_HEAD")
        actual_sha = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        if actual_sha != SOURCE_SHA:
            raise ValueError("exact source SHA mismatch")
        os.environ["HALA_NONCOMMERCIAL_RESEARCH"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"
        sys.path.insert(0, str(REPO))
        data["stage"] = "REAL_H92_VS_H94_NEW_PUBLIC_EVAL"
        record(data)
        from h95.eval import run
        report = run(repo=REPO, work=SCRATCH, output=OUTPUT)
        if (report.get("status") != "REAL_PAIRED_NEW_OFFICIAL_PUBLIC_EVAL_DIAGNOSTIC"
                or report.get("sample_counts") != {"h92": 16, "h94": 16}
                or not report.get("raw_evidence_sha256")):
            raise ValueError("paired new-16-source evaluation not qualified")
        data.update(status="H95_REAL_NEW_PAIRED_EVAL_COMPLETE",
                    stage="FINISHED_STRICT_AND_FORMAT_DIAGNOSTICS",
                    strict_successes=report["strict_successes"],
                    report_sha256=report["raw_evidence_sha256"])
        record(data)
        print("H95_REAL_SOURCE_PINNED_NEW_EVAL_COMPLETE", flush=True)
    except Exception as error:
        data.update(status="FAILED_NO_QUALIFIED_PAIRED_EVAL",
                    failure_class=type(error).__name__)
        record(data)
        print("H95_EVAL_FAILED", type(error).__name__, flush=True)
        raise


if __name__ == "__main__":
    main()
