"""H162 owner-private independent third original MIT app frozen LoRA paired inference."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_COMMIT="152b64c1b40c112af718761f87afa6d25934e0bb"
WORK=Path("/kaggle/tmp/hala-h162-independent-html5-app-eval")
OUT=Path("/kaggle/working/hala-h162-third-original-app-paired-inference")
RECEIPT=Path("/kaggle/working/h162-free-t4-crossapp-receipt.json")


def command(*args,timeout=180):
    r=subprocess.run(args,capture_output=True,text=True,
                     check=False,timeout=timeout)
    if r.returncode:
        raise RuntimeError("H162 pinned third-app provenance setup failed")
    return r.stdout.strip()


def save(doc):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=RECEIPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n","utf8")
    tmp.replace(RECEIPT)


def find(root,base):
    entries=list(Path(root).rglob(base))+list(Path(root).rglob(base+".zip"))
    matching=[x for x in dict.fromkeys(entries) if x.is_file() or x.is_dir()]
    if len(matching)!=1:
        raise ValueError("H162 one original source/model expected")
    return matching[0]


def main():
    receipt={
        "schema":"hala.h162.private-frozen-two-lora-third-app-t4.v1",
        "source_commit":SOURCE_COMMIT,
        "status":"H162_NOT_EVALUATED",
        "real_gpu_calls":0,"optimizer_steps":0,
        "h140_blind_calls":0,"paid_GPU":False,
        "Hala35B_unchanged":True,
        "model_weights_published":False,
    }
    try:
        if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or
            WORK.exists() or OUT.exists()):
            raise ValueError("H162 private new free GPU kernel only")
        import torch
        if (not torch.cuda.is_available() or
            not any("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count()))):
            raise ValueError("H162 paid GPU is prohibited")
        mount=Path("/kaggle/input")
        subfolders=[x for x in mount.iterdir() if x.is_dir()]
        if len(subfolders)!=1:
            raise ValueError("H162 only one owner PRIVATE data mount")
        names=("h162-third-html5-original-source",
               "h157-starting-research-lora",
               "h161-mixed-trained-research-lora")
        source,old,new=(find(mount,name) for name in names)
        if (len({x.resolve() for x in (source,old,new)})!=3 or
            any(not p.resolve().is_relative_to(subfolders[0].resolve())
                for p in (source,old,new))):
            raise ValueError("H162 original app/test/model source paths drift")
        receipt["status"]="H162_OWNER_SOURCE_AND_MODELS_ADMITTED"
        save(receipt)
        command(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        command(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0",
                timeout=540)
        versions={p:importlib.metadata.version(p).split("+")[0]
                  for p in ("torch","transformers","peft","Pillow")}
        if versions!={"torch":"2.11.0","transformers":"5.17.0",
                     "peft":"0.21.2","Pillow":"11.3.0"}:
            raise ValueError("H162 free T4 pinned dependencies not reproducible")
        WORK.parent.mkdir(parents=True,exist_ok=True)
        command("git","init","-q",str(WORK))
        command("git","-C",str(WORK),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        command("git","-C",str(WORK),"fetch","--depth","1","origin",SOURCE_COMMIT)
        command("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if command("git","-C",str(WORK),"rev-parse","HEAD")!=SOURCE_COMMIT:
            raise ValueError("H162 source commit SHA changed after freeze")
        sys.path.insert(0,str(WORK))
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1",
                          HALA_NONCOMMERCIAL_RESEARCH="1")
        from h162.gpu_infer import run
        receipt["status"]="H162_RUNNING_32_REAL_GPU_THIRD_APP_CALLS"
        save(receipt)
        result=run(source,old,new,OUT)
        if (result.get("status")!="H162_COMPLETED_32_REAL_FREE_T4_THIRD_APP_CALLS" or
            result.get("gpu_inference_calls")!=32 or
            result.get("optimizer_steps")!=0 or result.get("h140_blind_calls")!=0):
            raise ValueError("H162 third app independent inference incomplete")
        receipt.update(
            status="H162_COMPLETED_32_GENUINE_THIRD_APP_GPU_CALLS",
            real_gpu_calls=32,
            model_format_counts=result["counts"])
        save(receipt)
    except Exception as exc:
        receipt.update(status="H162_NO_QUALIFIED_CROSSAPP_RESULT",
                       failure_class=type(exc).__name__)
        save(receipt)
        raise


if __name__=="__main__":
    main()
