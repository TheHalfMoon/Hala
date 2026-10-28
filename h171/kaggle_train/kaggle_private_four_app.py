"""H171 private free Tesla T4 original 4-app bilingual SFT, TRAIN-only."""
import json,os,subprocess,sys
from pathlib import Path

SHA="d2de3c9cc3cee75e8de55fe73f533fbf5d819c4f"
OUT=Path("/kaggle/working/hala-h171-private-four-original-apps")
JOB=Path("/kaggle/working/h171-private-train-job.json")
SOURCE=Path("/kaggle/tmp/h171-pinned-code")

def cmd(*args):
    subprocess.run(args,capture_output=True,text=True,check=True,timeout=600)

def record(status,**info):
    doc=dict(schema="hala.h171.owner-private-free-t4-four-app-job.v1",
        status=status,source_commit=SHA,
        optimizer_steps=0,dev_ood_model_calls=0,
        owner_dev_gold_used=False,previous_consumed_ood_used=False,
        paid_gpu=False,weights_published=False,
        flagship_35b_unchanged=True)
    doc.update(info)
    JOB.write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n","utf8")

def main():
    try:
        root=Path("/kaggle/input")
        files=[p for p in root.rglob("*") if p.is_file()]
        old=[p for p in files if p.name=="h168-training.pkg"]
        new=[p for p in files if p.name=="h171-training.pkg"]
        frozen=[p for p in files if p.name=="adapter_model.safetensors"
                and "hala-h167-private-frozen-language-only-v1" in p.parts]
        if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or len(SHA)!=40
            or len(files)!=5 or len(old)!=1 or len(new)!=1 or len(frozen)!=1
            or "hala-h168-private-train-32-v1" not in old[0].parts
            or "hala-h171-private-new-click-train-32-v1" not in new[0].parts
            or {p.name for p in frozen[0].parent.iterdir()}!={
                "README.md","adapter_config.json","adapter_model.safetensors"}):
            raise ValueError("H171 only new and old private TRAIN plus frozen H164 LoRA")
        import torch
        if not torch.cuda.is_available() or any(
             "T4" not in torch.cuda.get_device_name(i)
             for i in range(torch.cuda.device_count())):
            raise ValueError("H171 strict FREE Nvidia Tesla T4 only")
        cmd(sys.executable,"-m","pip","uninstall","-y","torchao")
        cmd(sys.executable,"-m","pip","install","-q",
            "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0")
        SOURCE.parent.mkdir(parents=True,exist_ok=True)
        cmd("git","init","-q",str(SOURCE))
        cmd("git","-C",str(SOURCE),"remote","add","origin",
            "https://github.com/TheHalfMoon/Hala.git")
        cmd("git","-C",str(SOURCE),"fetch","--depth","1","origin",SHA)
        cmd("git","-C",str(SOURCE),"checkout","--detach","-q","FETCH_HEAD")
        if subprocess.check_output(["git","-C",str(SOURCE),"rev-parse","HEAD"],
                 text=True).strip()!=SHA:
            raise ValueError("H171 source SHA differs from frozen commit")
        sys.path.insert(0,str(SOURCE))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        record("H171_FOUR_APP_TRAINING_RUNNING")
        from h171.train_gpu import run
        result=run(old[0],new[0],frozen[0].parent,OUT)
        if result["real_optimizer_steps"]!=128 or result["unique_original_train_cases"]!=64:
            raise ValueError("H171 four-app full 128 optimizer steps unqualified")
        record("H171_128_REAL_FOUR_APP_ORIGINAL_T4_TRAIN_PASS",
               optimizer_steps=128,
               peak_allocated_gib=result["peak_allocated_gib"],
               private_vision_sha256=result["private_vision_weights_sha256"])
    except Exception as e:
        record("H171_FOUR_APP_GPU_NOT_QUALIFIED",
            failure_class=type(e).__name__,failure_message=str(e)[:220])
        raise

if __name__=="__main__":
    main()
