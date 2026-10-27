"""Private H164 384/96 free-T4 native source training launcher.

Fails closed until the exact GitHub commit and dataset/source SHA identities
have been pinned. Never reads OOD source. No paid compute or weight release.
"""
import importlib.metadata
import json,os,subprocess,sys
from pathlib import Path

SOURCE_COMMIT="UNQUALIFIED"
ROOT=Path("/kaggle/tmp/hala-h164-pinned-training-source")
OUT=Path("/kaggle/working/hala-h164-new-original-ui-384-trained-lora")
RECEIPT=Path("/kaggle/working/h164-free-T4-train-job-receipt.json")

def command(*args,timeout=200):
    result=subprocess.run(args,capture_output=True,text=True,
                          check=False,timeout=timeout)
    if result.returncode:
        raise RuntimeError("H164 pinned external train source setup failed")
    return result.stdout.strip()

def record(doc):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    file=RECEIPT.with_suffix(".tmp")
    file.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n","utf8")
    file.replace(RECEIPT)

def locate(mount,name):
    found=list(mount.rglob(name))+list(mount.rglob(name+".zip"))
    filtered=[f for f in dict.fromkeys(found) if f.is_file() or f.is_dir()]
    if len(filtered)!=1:raise ValueError("H164 exactly one owner-private frozen source")
    return filtered[0]

def main():
    receipt={"schema":"hala.h164.private-new-original-ui-free-t4.v1",
             "source_commit":SOURCE_COMMIT,
             "status":"H164_NOT_TRAINED",
             "optimizer_steps":0,"h140_blind_calls":0,
             "paid_GPU":False,"flagship_Hala35B_unchanged":True,
             "model_weights_published":False}
    try:
        if (len(SOURCE_COMMIT)!=40 or ROOT.exists() or OUT.exists() or
            not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")):
            raise ValueError("H164 private pinned new free T4 kernel required")
        import torch
        if (not torch.cuda.is_available() or
            not any("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count()))):
            raise ValueError("H164 require FREE Kaggle T4; never paid GPU")
        mount=Path("/kaggle/input")
        roots=[x for x in mount.iterdir() if x.is_dir()]
        if len(roots)!=1:
            raise ValueError("H164 exactly one private dataset source required")
        train,dev,ood,parent=[
            locate(mount,x) for x in (
                "h164-train-real-original-ui",
                "h164-dev-real-original-ui",
                "h164-ood-real-original-ui",
                "h164-parent-h161-frozen-lora")]
        if (len({x.resolve() for x in (train,dev,ood,parent)})!=4 or
            any(not x.resolve().is_relative_to(roots[0].resolve())
                for x in (train,dev,ood,parent))):
            raise ValueError("H164 owner private source input paths changed")
        receipt["status"]="H164_PRIVATE_SOURCE_MOUNT_ACCEPTED"
        record(receipt)
        command(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        command(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0",
                timeout=600)
        versions={name:importlib.metadata.version(name).split("+")[0]
                  for name in ("torch","transformers","peft","Pillow")}
        if versions!={"torch":"2.11.0","transformers":"5.17.0",
                     "peft":"0.21.2","Pillow":"11.3.0"}:
            raise ValueError("H164 only exact free-T4 research packages permitted")
        ROOT.parent.mkdir(parents=True,exist_ok=True)
        command("git","init","-q",str(ROOT))
        command("git","-C",str(ROOT),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        command("git","-C",str(ROOT),"fetch","--depth","1","origin",SOURCE_COMMIT)
        command("git","-C",str(ROOT),"checkout","--detach","-q","FETCH_HEAD")
        if command("git","-C",str(ROOT),"rev-parse","HEAD")!=SOURCE_COMMIT:
            raise ValueError("H164 actual repository source commit differs")
        sys.path.insert(0,str(ROOT))
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1",
                          HALA_NONCOMMERCIAL_RESEARCH="1")
        from h164.train_gpu import run,STEPS
        receipt["status"]="H164_TRAINING_REAL_384_EXAMPLES"
        record(receipt)
        # OOD is intentionally never opened by training model or helper.
        result=run(train,dev,parent,OUT)
        if (result.get("status")!=
            "H164_192_GENUINE_FREE_T4_OPTIMIZER_STEPS_COMPLETED"
            or result.get("optimizer_steps")!=STEPS
            or result.get("unique_train_images")!=384
            or result.get("dev_reference_labels_seen_by_gpu") is not False):
            raise ValueError("H164 real free T4 native learning not complete")
        receipt.update(
            status="H164_COMPLETED_GENUINE_192_T4_OPTIMIZER_STEPS",
            optimizer_steps=STEPS,
            trained_adapter_sha256=result["new_H164_adapter_sha256"])
        record(receipt)
    except Exception as error:
        receipt.update(status="H164_NOT_QUALIFIED_FOR_RELEASE",
                       failure_class=type(error).__name__)
        record(receipt)
        raise

if __name__=="__main__":
    main()
