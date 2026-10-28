"""H172 one-shot private frozen H164/H168 two-adapter DEV inference; no gold."""
import json,os,subprocess,sys
from pathlib import Path
SHA="UNQUALIFIED"
OUT=Path("/kaggle/working/hala-h172-private-paired-dev")
JOB=Path("/kaggle/working/h172-paired-dev-job.json")
WORK=Path("/kaggle/tmp/h172-source")
DATA=Path("/kaggle/input")
def command(*args):
    subprocess.run(args,capture_output=True,text=True,check=True,timeout=600)
def receipt(status,**kw):
    doc=dict(schema="hala.h172.owner-private-paired-dev-kaggle-job.v1",
             status=status,source_commit=SHA,optimizer_steps=0,
             owner_gold_uploaded=False,old_ood_used=False,paid_gpu=False,
             weights_published=False,flagship_35b_unchanged=True,**kw)
    JOB.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n","utf8")
def main():
    try:
        if len(SHA)!=40 or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise ValueError("H172 private pinned Kaggle only")
        files=[x for x in DATA.rglob("*") if x.is_file()]
        dev=[x for x in files if x.name=="h172-dev.pkg"]
        lang=[x for x in files if x.name=="adapter_model.safetensors"
              and "hala-h167-private-frozen-language-only-v1" in x.parts]
        visual=[x for x in files if x.name=="adapter_model.safetensors"
              and "hala-h172-private-vision-only-v1" in x.parts]
        if (len(dev)!=1 or len(lang)!=1 or len(visual)!=1
            or len(files)!=6 or
            "hala-h172-private-new-dev-only-v1" not in dev[0].parts
            or {x.name for x in lang[0].parent.iterdir()}!={"README.md",
                                "adapter_config.json","adapter_model.safetensors"}
            or {x.name for x in visual[0].parent.iterdir()}!={
                                "adapter_config.json","adapter_model.safetensors"}):
            raise ValueError("H172 owner-only DEV, frozen language and vision source mounts invalid")
        command(sys.executable,"-m","pip","uninstall","-y","torchao")
        command(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0")
        WORK.parent.mkdir(parents=True,exist_ok=True)
        command("git","init","-q",str(WORK))
        command("git","-C",str(WORK),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        command("git","-C",str(WORK),"fetch","--depth","1","origin",SHA)
        command("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if subprocess.check_output(
                ["git","-C",str(WORK),"rev-parse","HEAD"],
                text=True).strip()!=SHA:
            raise ValueError("H172 pinned source SHA did not match")
        sys.path.insert(0,str(WORK))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        receipt("H172_PAIRED_DEV_EVAL_RUNNING",real_gpu_calls=0)
        from h172.infer_gpu import run
        done=run(dev[0],lang[0].parent,visual[0].parent,OUT)
        if done["real_gpu_calls"]!=64 or done["optimizer_steps"]!=0:
            raise ValueError("H172 32 paired DEV model proposals incomplete")
        receipt("H172_PAIRED_32_NEW_DEV_REAL_GPU_PREDICTIONS_PASS",
                real_gpu_calls=64,peak_allocated_gib=done["peak_allocated_gib"])
    except Exception as exc:
        receipt("H172_NEW_DEV_GPU_NOT_QUALIFIED",real_gpu_calls=0,
                failure_class=type(exc).__name__,
                failure_message=str(exc)[:240])
        raise
if __name__=="__main__":
    main()
