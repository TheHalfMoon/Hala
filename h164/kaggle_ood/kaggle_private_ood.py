"""Private H164 one-shot, paired genuine Pell/Pikaday OOD inference, free T4 only.
The owner gold/DOM labels are never sent to Kaggle. No optimizer steps.
"""
import importlib.metadata
import json,os,subprocess,sys
from pathlib import Path

SOURCE_COMMIT="644ba511a629863bd7575ac8032122df8445a281"
WORK=Path("/kaggle/tmp/hala-h164-frozen-ood-eval")
OUT=Path("/kaggle/working/hala-h164-two-new-app-ood-paired-inference")
RECEIPT=Path("/kaggle/working/h164-ood-kernel-receipt.json")

def command(*argv,timeout=210):
    p=subprocess.run(argv,capture_output=True,text=True,check=False,timeout=timeout)
    if p.returncode:
        raise RuntimeError("H164 exact pinned research source setup failed")
    return p.stdout.strip()

def record(doc):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=RECEIPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n","utf8")
    tmp.replace(RECEIPT)

def locate(mount,name):
    found=list(mount.rglob(name))+list(mount.rglob(name+".zip"))
    found=[p for p in dict.fromkeys(found) if p.is_file() or p.is_dir()]
    if len(found)!=1:
        raise ValueError("H164 requires exactly one frozen private OOD input "+name)
    return found[0]

def main():
    receipt={"schema":"hala.h164.private-ood-64-paired-free-t4.v1",
      "source_commit":SOURCE_COMMIT,"status":"H164_OOD_NOT_EVALUATED",
      "gpu_calls":0,"optimizer_steps":0,"paid_GPU":False,
      "h140_blind_calls":0,"flagship_Hala35B_unchanged":True,
      "model_weights_published":False,"owner_gold_uploaded":False}
    try:
        if (len(SOURCE_COMMIT)!=40 or WORK.exists() or OUT.exists() or
            not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")):
            raise ValueError("H164 source SHA not pinned or not private Kaggle kernel")
        import torch
        if (not torch.cuda.is_available() or
            not any("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count()))):
            raise ValueError("H164 FREE Nvidia T4 required; no paid fallback")
        mount=Path("/kaggle/input")
        roots=[p for p in mount.iterdir() if p.is_dir()]
        if len(roots)!=1:
            raise ValueError("H164 exactly one owner-private source dataset required")
        source,old,new=[locate(mount,n) for n in (
            "h164-ood-real-original-ui",
            "h164-parent-h161-frozen-lora",
            "h164-trained-384-lora")]
        if (len({p.resolve() for p in (source,old,new)})!=3 or
            any(not p.resolve().is_relative_to(roots[0].resolve())
                for p in (source,old,new))):
            raise ValueError("H164 private Kaggle input mount changed")
        receipt["status"]="H164_FROZEN_OOD_AND_ADAPTERS_ADMITTED"
        record(receipt)
        command(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        command(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0",
                timeout=600)
        versions={n:importlib.metadata.version(n).split("+")[0] for n in
                  ("torch","transformers","peft","Pillow")}
        if versions!={"torch":"2.11.0","transformers":"5.17.0",
                      "peft":"0.21.2","Pillow":"11.3.0"}:
            raise ValueError("H164 exact free T4 dependencies unavailable")
        WORK.parent.mkdir(parents=True,exist_ok=True)
        command("git","init","-q",str(WORK))
        command("git","-C",str(WORK),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        command("git","-C",str(WORK),"fetch","--depth","1","origin",SOURCE_COMMIT)
        command("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if command("git","-C",str(WORK),"rev-parse","HEAD")!=SOURCE_COMMIT:
            raise ValueError("H164 pinned source Git commit changed")
        sys.path.insert(0,str(WORK))
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1",
                          HALA_NONCOMMERCIAL_RESEARCH="1")
        from h164.ood_gpu_infer import run
        receipt["status"]="H164_128_REAL_PAIRED_OOD_GPU_INFERENCE_RUNNING"
        record(receipt)
        result=run(source,old,new,OUT)
        if (result.get("status")!="H164_COMPLETED_128_REAL_GPU_OOD_CALLS" or
            result.get("real_gpu_calls")!=128 or
            result.get("optimizer_steps")!=0 or
            result.get("private_owner_gold_on_gpu") is not False):
            raise ValueError("H164 paired real GPU OOD calls not qualified")
        receipt.update(status="H164_COMPLETED_128_REAL_GPU_OOD_CALLS",
                       gpu_calls=128,
                       new_H164_adapter_sha256=result["new_H164_adapter_sha256"],
                       output_counts=result["counts"])
        record(receipt)
    except Exception as e:
        receipt.update(status="H164_OOD_NOT_QUALIFIED",
                       failure_class=type(e).__name__)
        record(receipt)
        raise

if __name__=="__main__":
    main()
