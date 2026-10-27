"""Owner-private H163 frozen paired inference ONLY on genuinely new GridStack app."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_COMMIT="2d28123d13e4abe0691f7d2cbbda7654bdc17b5b"
WORK=Path("/kaggle/tmp/hala-h163-new-original-app-source")
OUT=Path("/kaggle/working/hala-h163-gridstack-paired-model-inference")
RECEIPT=Path("/kaggle/working/h163-free-t4-original-app-receipt.json")


def command(*argv,timeout=180):
    result=subprocess.run(argv,capture_output=True,text=True,
                          check=False,timeout=timeout)
    if result.returncode:
        raise RuntimeError("H163 exact owner frozen source startup failed")
    return result.stdout.strip()


def record(doc):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=RECEIPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n","utf8")
    tmp.replace(RECEIPT)


def locate(mount,basename):
    items=list(mount.rglob(basename))+list(mount.rglob(basename+".zip"))
    matches=[p for p in dict.fromkeys(items) if p.is_dir() or p.is_file()]
    if len(matches)!=1:
        raise ValueError("H163 exactly one frozen data/model input per type")
    return matches[0]


def main():
    receipt={
        "schema":"hala.h163.free-t4-new-licensed-gui-paired-eval.v1",
        "source_commit":SOURCE_COMMIT,
        "status":"H163_NOT_EVALUATED",
        "gpu_calls":0,"optimizer_steps":0,
        "paid_GPU":False,"h140_blind_calls":0,
        "flagship_Hala35B_unchanged":True,
        "weights_published":False,
    }
    try:
        if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            or WORK.exists() or OUT.exists()):
            raise ValueError("H163 private new FREE T4 kernel only")
        import torch
        if (not torch.cuda.is_available() or
            not any("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count()))):
            raise ValueError("H163 paid GPU fallback is forbidden")
        mount=Path("/kaggle/input")
        roots=[d for d in mount.iterdir() if d.is_dir()]
        if len(roots)!=1:
            raise ValueError("H163 exactly one private dataset expected")
        source,old,new=[locate(mount,x) for x in (
            "h163-new-original-gridstack-scenes",
            "h157-original-frozen-research-lora",
            "h161-original-trained-research-lora",
        )]
        if (len({x.resolve() for x in (source,old,new)})!=3
            or any(not x.resolve().is_relative_to(roots[0].resolve())
                   for x in (source,old,new))):
            raise ValueError("H163 private evaluation mount provenance changed")
        receipt["status"]="H163_SOURCE_MODELS_ADMITTED"
        record(receipt)
        command(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        command(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0",
                timeout=540)
        installed={name:importlib.metadata.version(name).split("+")[0]
                   for name in ("torch","transformers","peft","Pillow")}
        if installed!={"torch":"2.11.0","transformers":"5.17.0",
                      "peft":"0.21.2","Pillow":"11.3.0"}:
            raise ValueError("H163 exact FREE T4 package requirements not available")
        WORK.parent.mkdir(parents=True,exist_ok=True)
        command("git","init","-q",str(WORK))
        command("git","-C",str(WORK),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        command("git","-C",str(WORK),"fetch","--depth","1","origin",SOURCE_COMMIT)
        command("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if command("git","-C",str(WORK),"rev-parse","HEAD")!=SOURCE_COMMIT:
            raise ValueError("H163 pinned GitHub evaluator SHA changed")
        sys.path.insert(0,str(WORK))
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1",
                          HALA_NONCOMMERCIAL_RESEARCH="1")
        from h163.gpu_infer import run
        receipt["status"]="H163_RUNNING_32_REAL_FROZEN_GPU_INFERENCES"
        record(receipt)
        result=run(source,old,new,OUT)
        if (result.get("status")!="H163_COMPLETE_32_REAL_FREE_GPU_INFERENCES"
            or result.get("gpu_calls")!=32 or
            result.get("optimizer_steps")!=0 or
            result.get("h140_blind_calls")!=0):
            raise ValueError("H163 no claim without all genuine model calls")
        receipt.update(
            status="H163_COMPLETED_32_REAL_FROZEN_GPU_CALLS",
            gpu_calls=32,
            model_format_counts=result["counts"])
        record(receipt)
    except Exception as exc:
        receipt.update(status="H163_UNQUALIFIED_GPU_MODEL_RESULT",
                       failure_class=type(exc).__name__)
        record(receipt)
        raise


if __name__=="__main__":
    main()
