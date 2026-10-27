"""H158 owner PRIVATE paired independent-app test on free Kaggle T4.

No training, no H140 blind access, no model or media publication.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_COMMIT="NOT_COMMITTED_YET"
WORK=Path("/kaggle/tmp/hala-h158-crossapp-eval-source")
OUT=Path("/kaggle/working/hala-h158-new-app-inference")
RECEIPT=Path("/kaggle/working/hala-h158-free-t4-inference-receipt.json")


def command(*args,timeout=180):
    completed=subprocess.run(args,capture_output=True,text=True,
                             check=False,timeout=timeout)
    if completed.returncode:
        raise RuntimeError("H158 pinned free GPU provenance setup failed")
    return completed.stdout.strip()


def save(data):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    temp=RECEIPT.with_suffix(".tmp")
    temp.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n",encoding="utf8")
    temp.replace(RECEIPT)


def find(root,basename):
    root=Path(root)
    matches=list(root.rglob(basename))+list(root.rglob(basename+".zip"))
    matches=[p for p in matches if p.is_file() or p.is_dir()]
    if not matches and basename=="h158-original-todomvc-source":
        matches=[p.parent for p in root.rglob("manifest.json")
                 if p.is_file() and
                 json.loads(p.read_text("utf8")).get("schema")==
                 "hala.h158.unmodified-todomvc-crossapp-click-source.v1"]
    if not matches and basename.endswith("-frozen-research-lora"):
        matches=[p.parent for p in root.rglob("adapter_model.safetensors")
                 if p.is_file() and basename.split("-")[0] in str(p.parent)]
    matches=list(dict.fromkeys(matches))
    if len(matches)!=1:
        raise ValueError("H158 exactly one pinned original input required")
    return matches[0]


def one_private_mount(root,inputs):
    root=Path(root)
    if not root.is_dir() or len([p for p in root.iterdir() if p.is_dir()])!=1:
        raise ValueError("H158 Kaggle must mount only one private dataset")
    parent=[p for p in root.iterdir() if p.is_dir()][0].resolve()
    if len(set(map(lambda v:v.resolve(),inputs)))!=3:
        raise ValueError("H158 original source and two models must be distinct")
    if any(not p.resolve().is_relative_to(parent) for p in inputs):
        raise ValueError("H158 all inputs must be inside one private mount")


def main():
    receipt={
        "schema":"hala.h158.private-second-app-two-model-free-t4.v1",
        "source_commit":SOURCE_COMMIT,
        "status":"H158_NOT_EVALUATED",
        "real_gpu_calls":0,"optimizer_steps":0,"paid_gpu":False,
        "h140_sealed_blind_calls":0,"Hala35B_unchanged":True,
        "model_weights_published":False,
    }
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or WORK.exists() or OUT.exists():
            raise ValueError("H158 private new GPU kernel only")
        import torch
        if (not torch.cuda.is_available() or
            not any("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count()))):
            raise ValueError("H158 no paid GPU fallback")
        mount=Path("/kaggle/input")
        names=("h158-original-todomvc-source","h156-frozen-research-lora",
               "h157-frozen-research-lora")
        original,old,new=(find(mount,n) for n in names)
        one_private_mount(mount,(original,old,new))
        receipt["status"]="H158_PRIVATE_SOURCE_PROVENANCE_ADMITTED"
        save(receipt)
        command(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        command(sys.executable,"-m","pip","install","-q","transformers==5.17.0",
                "peft==0.21.2","Pillow==11.3.0",timeout=540)
        versions={x:importlib.metadata.version(x).split("+")[0]
                  for x in ("torch","transformers","peft","Pillow")}
        if versions!={"torch":"2.11.0","transformers":"5.17.0",
                     "peft":"0.21.2","Pillow":"11.3.0"}:
            raise ValueError("H158 pinned GPU dependencies not available")
        WORK.parent.mkdir(parents=True,exist_ok=True)
        command("git","init","-q",str(WORK))
        command("git","-C",str(WORK),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        command("git","-C",str(WORK),"fetch","--depth","1","origin",SOURCE_COMMIT)
        command("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if command("git","-C",str(WORK),"rev-parse","HEAD")!=SOURCE_COMMIT:
            raise ValueError("H158 private eval source commit changed")
        sys.path.insert(0,str(WORK))
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",
                          DO_NOT_TRACK="1",HALA_NONCOMMERCIAL_RESEARCH="1")
        from h158.gpu_infer import run
        receipt["status"]="H158_EXECUTING_32_GENUINE_CROSSAPP_MODEL_CALLS"
        save(receipt)
        result=run(original,old,new,OUT)
        if (result.get("status")!="H158_COMPLETE_REAL_32_GPU_NEW_APP_INFERENCES" or
            result.get("total_gpu_calls")!=32 or
            result.get("optimizer_steps")!=0 or
            result.get("h140_blind_calls")!=0):
            raise ValueError("H158 model paired inference incomplete")
        receipt.update(status="H158_COMPLETE_32_REAL_NEW_APP_GPU_CALLS",
                       real_gpu_calls=32,model_format_counts=result["counts"])
        save(receipt)
    except Exception as exc:
        receipt.update(status="FAIL_H158_NOT_QUALIFIED",
                       failure_class=type(exc).__name__)
        save(receipt)
        raise


if __name__=="__main__":
    main()
